// Host-only preparation for a direct-FP16 native-expert port. NOT an executable GPU path.
// Copyright (c) 2026 Niko1221 and the Strata contributors. SPDX-License-Identifier: MIT
// See docs/evo_x2/direct_fp16_preflight.md for source provenance and remaining gates.
#pragma once

#include "strata/prefill/moe_fused_iq.hpp"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strata::prefill::evo_fp16 {

enum class Mode { upstream, direct64, pair128 };
// This parser does not read an environment variable. Registration belongs to the future runtime integration.
inline bool parse_mode(const char* value, Mode& mode) {
    if (!value || std::strcmp(value, "upstream") == 0) { mode = Mode::upstream; return true; }
    if (std::strcmp(value, "direct64") == 0) { mode = Mode::direct64; return true; }
    if (std::strcmp(value, "pair128") == 0) { mode = Mode::pair128; return true; }
    return false;
}
inline bool valid_mode(Mode mode) {
    return mode == Mode::upstream || mode == Mode::direct64 || mode == Mode::pair128;
}
inline const char* mode_name(Mode mode) {
    return mode == Mode::upstream ? "upstream" : mode == Mode::direct64 ? "direct64" :
           mode == Mode::pair128 ? "pair128" : "invalid";
}

namespace detail {
inline bool add(size_t a, size_t b, size_t& out) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    out = a + b; return true;
}
inline bool mul(size_t a, size_t b, size_t& out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b; return true;
}
inline bool align(size_t n, size_t& out) {
    if (!add(n, 255, out)) return false;
    out &= ~size_t{255}; return true;
}
}  // namespace detail

// Bytes, not float elements. These are only the five shared MoE regions; routing arrays, ring, graphs and the
// rest of Prefill remain the caller's responsibility. Each region must have its own aligned, non-overlapping span.
struct Buffers {
    size_t gu = 0, h = 0, xq = 0, hq = 0, dm = 0;
};
inline bool fits(const Buffers& have, const Buffers& need) {
    return have.gu >= need.gu && have.h >= need.h && have.xq >= need.xq &&
           have.hq >= need.hq && have.dm >= need.dm;
}
struct Workspace {
    Buffers direct;
    Buffers envelope;  // max(direct, exact full-chunk upstream fallback), rounded per region
    size_t total = 0;
    size_t fp16_xh_bytes = 0;
    size_t old_int8_xh_bytes = 0;
};

// Runtime expert count can be pruned. Widths and routed experts are the compiled kernel contract.
inline bool geometry_supported(const core::ModelGeometry& g) {
    const auto& c = core::kCompiledGeometry;
    return g.family == c.family && g.n_embd == c.n_embd && g.n_ff == c.n_ff &&
           g.n_expert_used == c.n_expert_used && g.n_expert > 0 && g.n_expert <= c.n_expert &&
           g.n_expert % 32 == 0 && g.n_expert_used <= g.n_expert &&
           g.n_embd % 256 == 0 && g.n_ff % 64 == 0;
}

// The upstream fallback MUST be priced for these actual tokens, not stream_all_min()-1. Zero-byte hq is valid
// only if the selected upstream plan genuinely needs no hq. No allocation, device pointer or global state here.
inline bool workspace(size_t tokens, const core::ModelGeometry& g, const Buffers& upstream, Workspace& out) {
    if (!tokens || !geometry_supported(g)) return false;
    Workspace w;
    size_t rows, x_values, h_values, all_values, padded_rows, tiles, table_bytes, tile_bytes;
    if (!detail::mul(tokens, size_t(g.n_expert_used), rows) || rows > size_t(INT32_MAX) ||
        !detail::mul(tokens, size_t(g.n_embd), x_values) || !detail::mul(rows, size_t(g.n_ff), h_values) ||
        !detail::add(x_values, h_values, all_values) || !detail::mul(all_values, 2, w.fp16_xh_bytes) ||
        !detail::mul(all_values / 64, 80, w.old_int8_xh_bytes) ||
        !detail::mul(x_values, 2, w.direct.xq) || !detail::mul(h_values, 2, w.direct.h) ||
        !detail::mul(rows, size_t(g.n_embd), w.direct.dm) || !detail::mul(w.direct.dm, 4, w.direct.dm)) return false;
    // Same worst-case grouping-table extent as fused::group_bytes: int32 counts/offsets/fill/tile starts,
    // followed by one int2 per 64-row tile plus one possible partial tile per expert. Checked before arithmetic.
    if (!detail::add(rows, 63, padded_rows) || !detail::add(padded_rows / 64, size_t(g.n_expert), tiles) ||
        !detail::mul(tiles, 8, tile_bytes)) return false;
    table_bytes = ((4 * size_t(g.n_expert) + 2) * 4 + 15) / 16 * 16;
    if (!detail::add(table_bytes, tile_bytes, w.direct.gu)) return false;
    auto region = [&](size_t direct, size_t fallback, size_t& result) {
        return detail::align(std::max(direct, fallback), result) && detail::add(w.total, result, w.total);
    };
    if (!region(w.direct.gu, upstream.gu, w.envelope.gu) || !region(w.direct.h, upstream.h, w.envelope.h) ||
        !region(w.direct.xq, upstream.xq, w.envelope.xq) || !region(w.direct.hq, upstream.hq, w.envelope.hq) ||
        !region(w.direct.dm, upstream.dm, w.envelope.dm)) return false;
    out = w; return true;
}

// Explicit initial allowlist. In particular an unknown down type is NEVER read as IQ4_NL. Q4_K/Q5_K and other
// upstream-supported types remain upstream until their direct64 AND pair128 instances have separate GPU tests.
inline size_t gate_block_bytes(int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_XS: return 74;
        case GGML_TYPE_IQ2_S: return 82;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ3_S: return 110;
        case GGML_TYPE_IQ4_XS: return 136;
        default: return 0;
    }
}
inline bool types_supported(int gu, int down) {
    return gate_block_bytes(gu) != 0 && (down == GGML_TYPE_Q2_0 || down == GGML_TYPE_IQ4_NL);
}
// Contiguous native GGUF rows, full [gate | up | down] coverage, and 2-byte weight loads only. Do not infer any
// of these fields from a model's marketing quantization label. The caller proves this extent for EVERY expert.
inline bool layout_supported(const core::ModelGeometry& g, const fused::NativeGeom& f,
                             size_t min_blob_bytes, size_t common_blob_alignment) {
    if (!geometry_supported(g) || !types_supported(f.gu_type, f.d_type) ||
        common_blob_alignment < 2 || common_blob_alignment % 2 != 0) return false;
    const size_t gu_row = size_t(g.n_embd / 256) * gate_block_bytes(f.gu_type);
    const size_t d_row = size_t(g.n_ff / 64) * 18; // Q2_0 and IQ4_NL are 18 bytes / 64 and / 32 respectively below
    const size_t expected_down_row = f.d_type == GGML_TYPE_IQ4_NL ? 2 * d_row : d_row;
    size_t gu_size, down_end, down_size;
    if (!detail::mul(size_t(g.n_ff), gu_row, gu_size) || !detail::mul(size_t(g.n_embd), expected_down_row, down_size) ||
        !detail::add(f.down_off, down_size, down_end)) return false;
    return f.gu_row == gu_row && f.d_row == expected_down_row && f.up_off == gu_size &&
           f.down_off == 2 * gu_size && down_end <= min_blob_bytes;
}

struct Flags {
    // -1 unset, 0 off, 1 on. A direct request is the opt-in if these are unset.
    int fused = -1, fused_native = -1, mmq = -1;
};
inline bool flags_valid(const Flags& f) {
    return f.fused >= -1 && f.fused <= 1 && f.fused_native >= -1 && f.fused_native <= 1 && f.mmq >= -1 && f.mmq <= 1;
}
struct Device {
    bool linux_host = false, hip = false, integrated = false;
    const char* arch = nullptr;
    int wave_size = 0;
};
inline bool device_supported(const Device& d) {
    return d.linux_host && d.hip && d.integrated && d.wave_size == 32 && d.arch &&
           std::strncmp(d.arch, "gfx1151", 7) == 0 && (d.arch[7] == '\0' || d.arch[7] == ':');
}
struct Input {
    Mode requested = Mode::upstream;
    Flags flags;
    Device device;
    core::ModelGeometry geometry = core::kCompiledGeometry;
    fused::NativeGeom format;
    size_t min_blob_bytes = 0, common_blob_alignment = 0;
    size_t actual_tokens = 0, stream_all_min = 1024;
    size_t ring = 0, stage = 8;
    bool has_source = false, native_layout = false, no_peer = false;
    bool had_gu = false, had_d = false, grouped_scratch = false;
    bool upstream_supported = false;
    bool mmq_or_fused_only_plan = false;
    int fused_override = -1;
    // Future kernel translation unit supplies these facts. They default false; an env flag is not proof.
    bool direct64_built = false, pair128_built = false;
    // Allocator must supply exact upstream needs at actual_tokens and actual capacity BEFORE any conversion.
    size_t upstream_priced_tokens = 0;
    Buffers upstream_required, capacity;
    bool scratch_aligned = false, scratch_context_owned = false;
};
enum class Outcome { upstream, direct_candidate, reject };
struct Plan {
    Outcome outcome = Outcome::upstream;
    Mode selected = Mode::upstream;
    const char* reason = "upstream-requested";
    Workspace bytes;
    // Direct X/H always use FP16; Dm and combine always use FP32. STRATA_PF_F16/DM16 are not read or changed.
    bool dm16_used = false;
};
// Pure pre-conversion decision. This does NOT launch a kernel or attest qualification. Safety conditions stay
// outside (upstream_supported || evo_direct_supported); a direct request cannot override them.
inline Plan plan(const Input& i) {
    Plan p;
    auto reject = [&](const char* reason) { p.outcome = Outcome::reject; p.reason = reason; return p; };
    auto fallback = [&](const char* reason) {
        p.reason = reason;
        if (!fits(i.capacity, i.upstream_required)) { p.outcome = Outcome::reject; p.reason = "upstream-workspace-too-small"; }
        return p;
    };
    if (!valid_mode(i.requested) || !flags_valid(i.flags)) return reject("invalid-mode-or-flag");
    if (i.requested == Mode::upstream) return fallback("upstream-requested");
    if (!i.actual_tokens || i.upstream_priced_tokens != i.actual_tokens) return reject("full-chunk-upstream-price-required");
    if (i.flags.fused == 0 || i.flags.fused_native == 0 || i.flags.mmq == 0) return reject("direct-mode-flag-conflict");
    if (!device_supported(i.device)) return fallback("device-unsupported");
    if (!geometry_supported(i.geometry)) return fallback("geometry-unsupported");
    if (!types_supported(i.format.gu_type, i.format.d_type)) return fallback("qtype-unsupported");
    if (!layout_supported(i.geometry, i.format, i.min_blob_bytes, i.common_blob_alignment)) return fallback("weight-layout-unsupported");
    if (i.actual_tokens < 1024 || i.actual_tokens < i.stream_all_min) return fallback("actual-chunk-below-threshold");
    if (i.ring <= i.stage || !i.has_source) return fallback("stream-all-unavailable");
    if (!i.native_layout || !i.no_peer || i.had_gu || i.had_d || i.grouped_scratch || i.fused_override == 0)
        return fallback("outer-safety-guard");
    const bool evo_supported = i.direct64_built && (i.requested != Mode::pair128 || i.pair128_built);
    if (!(i.upstream_supported || evo_supported) || !i.mmq_or_fused_only_plan) return fallback("native-plan-unavailable");
    // upstream-supported alone can never select the direct numerical representation.
    if (!evo_supported) return fallback("direct-kernel-not-built");
    if (!workspace(i.actual_tokens, i.geometry, i.upstream_required, p.bytes)) return fallback("workspace-overflow-or-index-limit");
    if (!i.scratch_aligned || !i.scratch_context_owned || !fits(i.capacity, p.bytes.envelope)) return fallback("direct-workspace-unavailable");
    p.outcome = Outcome::direct_candidate; p.selected = i.requested; p.reason = "direct-candidate-unqualified";
    return p;
}

// Per-expert partition, not per request: only FULL 128-row blocks use pair128. Every remainder, including a
// partial 64-row tile, reads the same FP16 X/H through direct64 with an explicit valid-row bound.
struct Tiles { size_t pair128 = 0, direct64 = 0, tail_rows = 0; };
inline bool tiles(Mode mode, size_t rows, Tiles& out) {
    if (mode != Mode::direct64 && mode != Mode::pair128) return false;
    Tiles t;
    if (mode == Mode::pair128) { t.pair128 = rows / 128; rows %= 128; }
    t.direct64 = rows / 64; t.tail_rows = rows % 64;
    out = t; return true;
}

}  // namespace strata::prefill::evo_fp16
