// Host-only contract and independent-oracle checks; NOT HIP kernel validation.
#include "strata/kernels/qsa_evo_topk.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
namespace e = strata::kernels::evo_topk;
using strata::kernels::QsaTopkScratch;

static uint32_t key(float s) {
    const float v = s + 0.0f;
    if (std::isnan(v)) return 0;
    uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}
// Oracle: sort token scores numerically, NaNs last, lower token ID on ties;
// then sort the chosen IDs ascending, as required by attention (not score order).
static std::vector<int> oracle(const std::vector<float>& scores, int tokens, int width) {
    std::vector<int> ids(tokens);
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        float x = scores[a / 4], y = scores[b / 4];
        if (std::isnan(x)) return false;
        if (std::isnan(y)) return true;
        return x > y;
    });
    ids.resize(std::min(tokens, width));
    std::sort(ids.begin(), ids.end());
    return ids;
}
// Model of the weighted four-pass radix and chunk prefix counts. This tests the
// algorithm and ID contract, not the HIP implementation's synchronization/codegen.
static std::vector<int> split_model(const std::vector<float>& scores, int tokens, int width) {
    if (tokens <= width) { std::vector<int> r(tokens); std::iota(r.begin(), r.end(), 0); return r; }
    const int nb = tokens / 4 + 1;
    auto weight = [&](int b) { return b < tokens / 4 ? 4 : tokens % 4; };
    uint32_t prefix = 0;
    int above = 0;
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        const uint32_t mask = p == 0 ? 0 : ~uint32_t(0) << (shift + 8);
        std::array<int, 256> hist{};
        for (int b = 0; b < nb; ++b)
            if ((key(scores[b]) & mask) == (prefix & mask)) hist[(key(scores[b]) >> shift) & 255] += weight(b);
        int d = 255;
        for (; d > 0; --d) { if (above + hist[d] >= width) break; above += hist[d]; }
        prefix |= uint32_t(d) << shift;
    }
    const int chunks = (nb + e::kChunkBlocks - 1) / e::kChunkBlocks;
    std::vector<int> gt(chunks), eq(chunks), out(width, -1);
    for (int b = 0; b < nb; ++b) {
        if (key(scores[b]) > prefix) gt[b / e::kChunkBlocks] += weight(b);
        if (key(scores[b]) == prefix) eq[b / e::kChunkBlocks] += weight(b);
    }
    for (int c = 0; c < chunks; ++c) {
        int before_eq = 0, pos = 0;
        for (int j = 0; j < c; ++j) {
            pos += gt[j] + std::clamp(width - above - before_eq, 0, eq[j]);
            before_eq += eq[j];
        }
        int left = std::max(0, width - above - before_eq);
        for (int b = c * e::kChunkBlocks; b < std::min(nb, (c + 1) * e::kChunkBlocks); ++b) {
            int take = key(scores[b]) > prefix ? weight(b) : key(scores[b]) == prefix ? std::min(weight(b), left) : 0;
            if (key(scores[b]) == prefix) left -= take;
            for (int t = 0; t < take; ++t) out.at(pos++) = 4 * b + t;
        }
    }
    return out;
}
int main() {
    for (int q : {1, 2, 8, 16}) for (int b : {1, 2048, 2049, 32768, 32770, 1048576}) {
        const auto a = e::layout(q, b);
        assert(a && a.chunks == (b + 2047) / 2048);
        assert(a.logical_bytes == std::size_t(4 * q) * (1034 + 2 * a.chunks));
        assert(a.state_offset % 256 == 0 && a.count_offset % 256 == 0 && a.bytes % 256 == 0);
        assert(a.state_offset >= std::size_t(q) * 4096);
        assert(a.count_offset >= a.state_offset + std::size_t(5 * q * 2 * 4));
        assert(a.bytes >= a.count_offset + std::size_t(q * a.chunks * 2 * 4));
    }
    for (int64_t q : std::array<int64_t, 4>{-1, 0, 17, INT64_MAX}) assert(!e::layout(q, 32770));
    for (int64_t b : std::array<int64_t, 4>{-1, 0, 1048577, INT64_MAX}) assert(!e::layout(16, b));
    auto a = e::layout(16, 32770);
    alignas(256) std::array<int32_t, 16 * (1034 + 2 * 17) + 256> data{};
    alignas(256) std::array<int32_t, 16 * (1034 + 2 * 17) + 256> other{};
    QsaTopkScratch scratch{data.data(), a}, scratch2{other.data(), a};
    assert(scratch.fits(1, 2049) && scratch.fits(16, 32770));
    assert(!scratch.fits(17, 32770) && !scratch.fits(1, 32771));
    assert(scratch.data != scratch2.data);
    scratch2.data[0] = 99; assert(scratch.data[0] == 0);
    scratch.data++; assert(!scratch.fits(1, 1)); scratch.data--;
    scratch.capacity.count_offset -= 256; assert(!scratch.fits(1, 1));
    assert(!e::parse(nullptr, nullptr).split);
    for (auto bad : {"SPLIT", "1", "split ", "unknown"}) assert(!e::parse(bad, nullptr).valid);
    for (auto bad : {"0", "65536x", "65537", "-1", "999999999999999999999"}) assert(!e::parse("split", bad).split);
    for (auto t : {"65536", "98304", "114688"}) {
        auto c = e::parse("split", t);
        assert(c.valid && c.split && !e::wants_split(c, c.min_tokens - 1));
        assert(e::wants_split(c, c.min_tokens) && e::wants_split(c, c.min_tokens + 1));
    }
    assert(e::device_matches(true, true, "gfx1151:sramecc-:xnack-", true, 32, 1, false));
    assert(!e::device_matches(true, true, "gfx11510", true, 32, 1, false));
    assert(!e::device_matches(true, true, "gfx1100", true, 32, 1, false));
    assert(!e::device_matches(false, true, "gfx1151", true, 32, 1, false));
    assert(!e::device_matches(true, false, "gfx1151", true, 32, 1, false));
    assert(!e::device_matches(true, true, "gfx1151", false, 32, 1, false));
    assert(!e::device_matches(true, true, "gfx1151", true, 64, 1, false));
    assert(!e::device_matches(true, true, "gfx1151", true, 32, 2, false));
    assert(!e::device_matches(true, true, "gfx1151", true, 32, 1, true));
    std::mt19937 rng(0x5411151);
    int checks = 0;
    for (int n : {1, 3, 4, 5, 8191, 8192, 8193, 65535, 65536, 65537, 98303, 98304, 98305, 114687, 114688, 114689, 131071}) {
        std::vector<float> scores(n / 4 + 1);
        for (int kind = 0; kind < 5; ++kind) {
            for (auto& f : scores) {
                if (kind == 0) f = float(int(rng() % 65) - 32);
                if (kind == 1) f = .25f;
                if (kind == 2) f = (rng() & 1) ? 0.f : -0.f;
                if (kind == 3) {
                    const float special[] = {NAN, INFINITY, -INFINITY, 0.f, -0.f, -1.f, 1.f};
                    f = special[rng() % 7];
                }
                if (kind == 4) f = -NAN;
            }
            for (int width : {1, 7, 2048, 2051, n}) {
                assert(split_model(scores, n, width) == oracle(scores, n, width));
                ++checks;
            }
        }
    }
    std::cout << "PASS host contract/layout/dispatch and " << checks << " independent oracle cases; HIP NOT RUN\n";
}
