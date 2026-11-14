// Host tests only. A synthetic built=true below exercises policy, not HIP compilation or numerical correctness.
#include "strata/prefill/evo_fp16_plan.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cstdio>
#include <string>

using namespace strata::prefill::evo_fp16;
static int failures = 0, checks = 0;
static void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
static bool reason(const Plan& p, const char* value) { return std::string(p.reason) == value; }
static Input fixture() {
    Input i;
    i.requested = Mode::pair128;
    i.device = {true, true, true, "gfx1151", 32};
    i.actual_tokens = i.upstream_priced_tokens = 8192;
    i.ring = 128;
    i.has_source = i.native_layout = i.no_peer = i.mmq_or_fused_only_plan = true;
    i.direct64_built = i.pair128_built = true;
    i.scratch_aligned = i.scratch_context_owned = true;
    const auto& g = i.geometry;
    i.format.gu_type = GGML_TYPE_IQ3_XXS; i.format.d_type = GGML_TYPE_IQ4_NL;
    i.format.gu_row = size_t(g.n_embd / 256) * 98;
    i.format.d_row = size_t(g.n_ff / 32) * 18;
    i.format.up_off = size_t(g.n_ff) * i.format.gu_row;
    i.format.down_off = 2 * i.format.up_off;
    i.min_blob_bytes = i.format.down_off + size_t(g.n_embd) * i.format.d_row;
    i.common_blob_alignment = 16;
    // A deliberately generous independently priced full-chunk fallback; direct must not shrink it away.
    const size_t rows = i.actual_tokens * size_t(g.n_expert_used);
    i.upstream_required = {rows * size_t(2 * g.n_ff) * 4, rows * size_t(g.n_ff) * 4,
                           rows * size_t(g.n_embd) * 2, rows * size_t(g.n_ff) * 2,
                           rows * size_t(g.n_embd) * 4};
    Workspace w;
    if (!workspace(i.actual_tokens, g, i.upstream_required, w)) std::abort();
    i.capacity = w.envelope;
    return i;
}
int main() {
    Mode m = Mode::pair128;
    expect(parse_mode(nullptr, m) && m == Mode::upstream, "unset keeps upstream");
    for (const char* v : {"upstream", "direct64", "pair128"}) expect(parse_mode(v, m) && std::string(mode_name(m)) == v, "exact modes");
    for (const char* v : {"", "DIRECT64", "pair256", "pair128 ", "2", "upstreamgarbage"}) expect(!parse_mode(v, m), "unknown mode rejected");
    // Derive reference row extents from the vendored ggml structs, not the planner's constants.
    const std::pair<int, size_t> formats[] = {
        {GGML_TYPE_IQ2_XXS, sizeof(block_iq2_xxs)}, {GGML_TYPE_IQ2_XS, sizeof(block_iq2_xs)},
        {GGML_TYPE_IQ2_S, sizeof(block_iq2_s)}, {GGML_TYPE_IQ3_XXS, sizeof(block_iq3_xxs)},
        {GGML_TYPE_IQ3_S, sizeof(block_iq3_s)}, {GGML_TYPE_IQ4_XS, sizeof(block_iq4_xs)}};
    for (const auto& gu : formats) for (int down : {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL}) {
        auto i = fixture();
        i.format.gu_type = gu.first; i.format.d_type = down;
        i.format.gu_row = size_t(i.geometry.n_embd / QK_K) * gu.second;
        i.format.d_row = down == GGML_TYPE_Q2_0 ? size_t(i.geometry.n_ff / QK2_0) * sizeof(block_q2_0)
                                               : size_t(i.geometry.n_ff / QK4_NL) * sizeof(block_iq4_nl);
        i.format.up_off = size_t(i.geometry.n_ff) * i.format.gu_row;
        i.format.down_off = 2 * i.format.up_off;
        i.min_blob_bytes = i.format.down_off + size_t(i.geometry.n_embd) * i.format.d_row;
        expect(plan(i).outcome == Outcome::direct_candidate, "GGML struct-based row size for every initial qtype pair");
    }
    Input good = fixture();
    auto p = plan(good);
    expect(p.outcome == Outcome::direct_candidate && p.selected == Mode::pair128 && !p.dm16_used, "explicit synthetic capability: FP32 Dm candidate");
    expect(fits(p.bytes.envelope, good.upstream_required) && fits(p.bytes.envelope, p.bytes.direct), "retain direct and full fallback extents");
    expect(p.bytes.total == p.bytes.envelope.gu + p.bytes.envelope.h + p.bytes.envelope.xq + p.bytes.envelope.hq + p.bytes.envelope.dm, "one aligned envelope, not double counted");
    if (good.geometry.n_embd == 2560 && good.geometry.n_ff == 640 && good.geometry.n_expert_used == 10) {
        expect(p.bytes.fp16_xh_bytes == 146800640, "8192-token exact FP16 X/H payload");
        expect(p.bytes.old_int8_xh_bytes == 91750400, "8192-token prior X/H payload");
        expect(p.bytes.fp16_xh_bytes - p.bytes.old_int8_xh_bytes == 55050240, "52.5 MiB delta, not total RAM");
        Workspace larger;
        expect(workspace(16384, good.geometry, {}, larger) && larger.fp16_xh_bytes - larger.old_int8_xh_bytes == 110100480, "16384-token 105 MiB delta");
        expect(p.bytes.direct.gu == 22544, "upstream group table bound including partial tiles");
    }
    for (size_t n : {size_t(1), size_t(127), size_t(1023), size_t(1024), size_t(1025)}) {
        auto i = good; i.actual_tokens = i.upstream_priced_tokens = n;
        expect(plan(i).outcome == (n >= 1024 ? Outcome::direct_candidate : Outcome::upstream), "actual chunk boundary, not configured chunk");
    }
    { auto i = good; i.stream_all_min = 16384; expect(reason(plan(i), "actual-chunk-below-threshold"), "existing higher stream threshold preserved"); }
    { auto i = good; i.actual_tokens = i.upstream_priced_tokens = 512; i.stream_all_min = 0; expect(plan(i).outcome == Outcome::upstream, "stream override cannot lower new 1024 floor"); }
    { auto i = good; i.upstream_priced_tokens = 1023; expect(plan(i).outcome == Outcome::reject, "shrunken small-tail fallback price refused"); }
    { auto i = good; i.direct64_built = false; expect(plan(i).outcome == Outcome::upstream, "pair requires direct64 kernel even with full tiles"); }
    { auto i = good; i.upstream_supported = true; i.pair128_built = false; expect(reason(plan(i), "direct-kernel-not-built"), "upstream support cannot authorize FP16 representation"); }
    { auto i = good; i.pair128_built = false; i.requested = Mode::direct64; expect(plan(i).outcome == Outcome::direct_candidate, "direct64 does not require pair built"); }
    for (int field = 0; field != 3; ++field) {
        auto i = good;
        if (field == 0) i.flags.fused = 0;
        if (field == 1) i.flags.fused_native = 0;
        if (field == 2) i.flags.mmq = 0;
        expect(plan(i).outcome == Outcome::reject && reason(plan(i), "direct-mode-flag-conflict"), "explicit disable is profile error");
        i.requested = Mode::upstream;
        expect(plan(i).outcome == Outcome::upstream, "upstream keeps existing disable meaning");
    }
    { auto i = good; i.flags.mmq = 2; expect(plan(i).outcome == Outcome::reject, "bad tri-state rejected"); }
    { auto i = good; i.requested = static_cast<Mode>(999); expect(plan(i).outcome == Outcome::reject, "invalid enum rejected"); }
    for (const char* arch : {"gfx1100", "gfx1201", "gfx11510", "gfx1151garbage", ""}) {
        auto i = good; i.device.arch = arch; expect(reason(plan(i), "device-unsupported"), "exact architecture guard");
    }
    { auto i = good; i.device.arch = "gfx1151:sramecc+:xnack-"; expect(plan(i).outcome == Outcome::direct_candidate, "standard architecture suffix allowed"); }
    for (int field = 0; field != 5; ++field) {
        auto i = good;
        if (field == 0) i.device.linux_host = false;
        if (field == 1) i.device.hip = false;
        if (field == 2) i.device.integrated = false;
        if (field == 3) i.device.wave_size = 64;
        if (field == 4) i.device.arch = nullptr;
        expect(plan(i).outcome == Outcome::upstream, "device facts all mandatory");
    }
    for (int field = 0; field != 4; ++field) {
        auto i = good;
        if (field == 0) i.geometry.n_embd += 256;
        if (field == 1) i.geometry.n_ff += 64;
        if (field == 2) i.geometry.n_expert_used += 1;
        if (field == 3) i.geometry.n_expert = 0;
        expect(reason(plan(i), "geometry-unsupported"), "compiled geometry or expert extent mismatch");
    }
    { auto i = good; i.geometry.n_expert = 32; expect(plan(i).outcome == Outcome::direct_candidate, "valid pruned expert count"); }
    for (int type = -1; type < GGML_TYPE_COUNT + 1; ++type) {
        auto i = good; i.format.d_type = type;
        if (type != GGML_TYPE_Q2_0 && type != GGML_TYPE_IQ4_NL) expect(reason(plan(i), "qtype-unsupported"), "unknown down never treated as IQ4_NL");
    }
    { auto i = good; i.format.gu_type = GGML_TYPE_IQ1_M; expect(reason(plan(i), "qtype-unsupported"), "IQ1_M gate remains upstream"); }
    { auto i = good; i.format.gu_type = GGML_TYPE_Q4_K; expect(reason(plan(i), "qtype-unsupported"), "additional upstream qtype remains upstream"); }
    { auto i = good; i.format.d_type = GGML_TYPE_Q2_0; i.format.d_row /= 2; expect(plan(i).outcome == Outcome::direct_candidate, "explicit Q2_0 row geometry"); }
    for (int field = 0; field != 6; ++field) {
        auto i = good;
        if (field == 0) ++i.format.gu_row;
        if (field == 1) ++i.format.d_row;
        if (field == 2) ++i.format.up_off;
        if (field == 3) ++i.format.down_off;
        if (field == 4) --i.min_blob_bytes;
        if (field == 5) i.common_blob_alignment = 1;
        expect(reason(plan(i), "weight-layout-unsupported"), "stride offset extent alignment guard");
    }
    for (int field = 0; field != 10; ++field) {
        auto i = good;
        if (field == 0) i.ring = i.stage;
        if (field == 1) i.has_source = false;
        if (field == 2) i.native_layout = false;
        if (field == 3) i.no_peer = false;
        if (field == 4) i.had_gu = true;
        if (field == 5) i.had_d = true;
        if (field == 6) i.grouped_scratch = true;
        if (field == 7) i.fused_override = 0;
        if (field == 8) i.mmq_or_fused_only_plan = false;
        if (field == 9) i.upstream_supported = true, i.had_gu = true;
        expect(plan(i).outcome == Outcome::upstream, "OR capability does not bypass outer guards");
    }
    for (int field = 0; field != 2; ++field) {
        auto i = good;
        if (field == 0) i.scratch_aligned = false;
        if (field == 1) i.scratch_context_owned = false;
        expect(reason(plan(i), "direct-workspace-unavailable"), "context-owned aligned scratch required");
    }
    { auto i = good; i.capacity = {}; expect(plan(i).outcome == Outcome::reject, "cannot fall back into undersized workspace"); }
    { auto i = good; i.format.d_type = 999; i.capacity.dm = 0; expect(plan(i).outcome == Outcome::reject, "unsupported down with no fallback capacity rejects before conversion"); }
    { auto i = good; i.upstream_required = {}; Workspace w; expect(workspace(i.actual_tokens, i.geometry, {}, w), "direct minimum pricing"); i.capacity = w.envelope; --i.capacity.xq; expect(plan(i).outcome == Outcome::upstream, "one-byte-short direct input region refused"); }
    { Workspace w; Buffers huge; huge.gu = SIZE_MAX; expect(!workspace(8192, good.geometry, huge, w), "alignment overflow refused"); expect(!workspace(SIZE_MAX, good.geometry, {}, w), "token multiply overflow refused"); expect(!workspace(size_t(INT32_MAX), good.geometry, {}, w), "int32 grouping overflow refused"); }
    // Independent exhaustive row coverage oracle. No rounding-up tail may create a padded pair read.
    for (Mode mode : {Mode::direct64, Mode::pair128}) for (size_t rows = 0; rows < 10000; ++rows) {
        Tiles t; expect(tiles(mode, rows, t), "direct partition exists");
        expect(t.pair128 * 128 + t.direct64 * 64 + t.tail_rows == rows && t.tail_rows < 64,
               "every real row covered once, no padding");
        if (mode == Mode::pair128) expect(t.direct64 <= 1 && t.pair128 * 128 <= rows, "pair accepts only complete 128 rows");
        else expect(t.pair128 == 0, "direct64 reference never uses pair");
    }
    Tiles t; expect(!tiles(Mode::upstream, 128, t), "upstream representation cannot enter direct tail scheduler");
    // A second context/request changes neither the first input nor its deterministic decision.
    auto second = good; second.actual_tokens = second.upstream_priced_tokens = 513; second.format.d_type = -1;
    (void) plan(second);
    expect(plan(good).outcome == Outcome::direct_candidate && plan(good).bytes.total == p.bytes.total, "no process-global plan state");
    std::printf("evo_fp16_plan_test: %d checks, %d failures (host policy only)\n", checks, failures);
    return failures ? 1 : 0;
}
