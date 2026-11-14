// EVO QSA split top-k: host-side contract, allocation layout and strict opt-in parsing.
// No device allocation, mutable global state or GPU runtime dependency in this header.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "strata/kernels/gfx_arch.hpp"

namespace strata::kernels {
namespace evo_topk {
constexpr int kMaxQueries = 16, kChunkBlocks = 2048, kMaxChunks = 512;
constexpr std::size_t kAlignment = 256;
inline std::size_t aligned(std::size_t n) { return (n + kAlignment - 1) & ~(kAlignment - 1); }

struct Layout {
    int queries = 0, chunks = 0;
    int64_t max_blocks = 0;
    std::size_t state_offset = 0, count_offset = 0, logical_bytes = 0, bytes = 0;
    explicit operator bool() const { return bytes != 0; }
};
inline Layout layout(int64_t queries, int64_t max_blocks) {
    Layout r;
    // Check before rounding/multiplying, including INT64_MAX inputs.
    if (queries < 1 || queries > kMaxQueries || max_blocks < 1 ||
        max_blocks > int64_t(kChunkBlocks) * kMaxChunks) return r;
    r.queries = int(queries);
    r.chunks = int((max_blocks + kChunkBlocks - 1) / kChunkBlocks);
    r.max_blocks = max_blocks;
    const std::size_t q = std::size_t(queries), c = std::size_t(r.chunks);
    r.state_offset = aligned(4 * q * 4 * 256);
    r.count_offset = r.state_offset + aligned(4 * 5 * q * 2);
    r.logical_bytes = 4 * q * (1034 + 2 * c);
    r.bytes = r.count_offset + aligned(4 * q * c * 2);
    return r;
}

struct Config {
    bool split = false;
    bool valid = true;
    int64_t min_tokens = 65536;
};
inline Config parse(const char* mode, const char* threshold) {
    Config r;
    if (mode && *mode) {
        if (std::strcmp(mode, "split") == 0) r.split = true;
        else if (std::strcmp(mode, "register") != 0) r.valid = false;
    }
    if (threshold && *threshold) {
        if (std::strcmp(threshold, "65536") == 0) r.min_tokens = 65536;
        else if (std::strcmp(threshold, "98304") == 0) r.min_tokens = 98304;
        else if (std::strcmp(threshold, "114688") == 0) r.min_tokens = 114688;
        else r.valid = false;
    }
    if (!r.valid) r.split = false;
    return r;
}
inline bool device_matches(bool hip, bool linux_os, const char* arch, bool integrated,
                           int wave_size, int device_count, bool arch_override) {
    return hip && linux_os && gfx_arch_is_gfx1151(arch) && integrated && wave_size == 32 &&
           device_count == 1 && !arch_override;
}
inline bool wants_split(const Config& config, int64_t pos0) {
    return config.valid && config.split && pos0 >= config.min_tokens;
}
}  // namespace evo_topk

// Non-owning view of a Verifier's arena allocation. The arena, not this view, owns its
// lifetime. One in-flight call per view; do not serialize it into session snapshots.
struct QsaTopkScratch {
    int32_t* data = nullptr;
    evo_topk::Layout capacity;
    bool fits(int64_t nq, int64_t blocks) const {
        if (!data || reinterpret_cast<std::uintptr_t>(data) % evo_topk::kAlignment != 0) return false;
        const auto expected = evo_topk::layout(capacity.queries, capacity.max_blocks);
        return expected && expected.bytes == capacity.bytes && expected.chunks == capacity.chunks &&
               expected.state_offset == capacity.state_offset && expected.count_offset == capacity.count_offset &&
               expected.logical_bytes == capacity.logical_bytes &&
               nq >= 1 && nq <= capacity.queries && blocks >= 1 && blocks <= capacity.max_blocks;
    }
};
}  // namespace strata::kernels
