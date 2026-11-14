// Typed, context-local host binding contract for the unimplemented direct-FP16 GPU path.
// Copyright (c) 2026 Niko1221 and the Strata contributors. SPDX-License-Identifier: MIT
#pragma once

#include "strata/prefill/evo_fp16_plan.hpp"

namespace strata::prefill::evo_fp16 {

// Mirrors pinned Prefill::Alloc::take's extra 256-byte guard, then rounds to 256.
// The planner's envelope contains payload capacity, NOT this allocator overhead.
inline bool allocation_take_bytes(size_t payload, size_t& bytes) {
    size_t guarded;
    return detail::add(payload, 256, guarded) && detail::align(guarded, bytes);
}
inline bool allocation_bytes(const Buffers& payload, size_t& bytes) {
    size_t total = 0, take;
    for (size_t n : {payload.gu, payload.h, payload.xq, payload.hq, payload.dm}) {
        if (!allocation_take_bytes(n, take) || !detail::add(total, take, total)) return false;
    }
    bytes = total; return true;
}

// Device addresses are opaque. This code never dereferences or frees them. capacity excludes Alloc's guard.
struct Region { void* address = nullptr; size_t capacity = 0; };
struct Binding { Region gu, h, xq, hq, dm; };
inline Buffers capacities(const Binding& b) { return {b.gu.capacity, b.h.capacity, b.xq.capacity, b.hq.capacity, b.dm.capacity}; }

namespace detail {
inline bool range(const Region& r, uintptr_t& first, uintptr_t& end) {
    first = reinterpret_cast<uintptr_t>(r.address);
    if (!r.capacity) { end = first; return true; }
    if (!r.address || first % 256 || r.capacity > UINTPTR_MAX - first) return false;
    end = first + r.capacity; return true;
}
}  // namespace detail

// Pairwise disjoint regions are mandatory even when an individual layer uses fewer bytes than its fallback.
// They can alias attention scratch OUTSIDE this binding only when the integration proves stream ordering.
inline bool binding_valid(const Binding& b, const Buffers& required) {
    if (!fits(capacities(b), required)) return false;
    const Region all[] = {b.gu, b.h, b.xq, b.hq, b.dm};
    uintptr_t first[5], end[5];
    for (size_t a = 0; a < 5; ++a) {
        if (!detail::range(all[a], first[a], end[a])) return false;
        for (size_t z = 0; z < a; ++z)
            if (all[a].capacity && all[z].capacity && first[a] < end[z] && first[z] < end[a]) return false;
    }
    return true;
}

// Distinct types prevent accidental direct-X/H arguments in an int8 upstream kernel adapter. The integration
// must write a deliberate typed adapter; these do not implicitly convert to void* or to one another.
struct Fp16Input { const uint16_t* data = nullptr; size_t values = 0; };
struct Fp16Output { uint16_t* data = nullptr; size_t values = 0; };
struct Fp32Output { float* data = nullptr; size_t values = 0; };
struct GroupTables { void* data = nullptr; size_t bytes = 0; };
struct Views {
    Fp16Output x;
    Fp16Output h;
    Fp32Output dm;
    GroupTables groups;
    Mode mode = Mode::upstream;
    size_t tokens = 0;
    void* stream = nullptr;
    Fp16Input x_input() const { return {x.data, x.values}; }
    Fp16Input h_input() const { return {h.data, h.values}; }
};

// retire invokes the adapter's record callback on the bound stream AFTER the caller's last read/write of this
// layer (including combine and telemetry copies). Query must report the backend event's status, never merely
// "cancel requested". The event must be created before capture. This class neither allocates nor destroys it.
// Keep event and user alive until poll returns complete. A failed record leaves the layer acquired.
enum class Completion { pending, complete, error };
struct Fence {
    const void* event = nullptr;
    void* user = nullptr;
    bool (*record)(void* user, const void* event, void* stream) noexcept = nullptr;
    Completion (*query)(void* user, const void* event) noexcept = nullptr;
};
enum class State { unbound, ready, acquired, pending, poisoned };

// Host-thread confined, noncopyable, and nonmovable. Place one inside each actual Prefill execution context.
// No allocator ownership is implied: the caller's allocation owner must outlive this guard AND GPU completion.
// This is preparatory code only; no production caller exists yet.
class WorkspaceGuard {
public:
    WorkspaceGuard() = default;
    WorkspaceGuard(const WorkspaceGuard&) = delete;
    WorkspaceGuard& operator=(const WorkspaceGuard&) = delete;
    WorkspaceGuard(WorkspaceGuard&&) = delete;
    WorkspaceGuard& operator=(WorkspaceGuard&&) = delete;

    State state() const { return state_; }
    size_t retained_graphs() const { return graphs_; }
    bool can_release_owner() const { return (state_ == State::unbound || state_ == State::ready) && !graphs_; }

    bool bind(Input input, const Binding& binding, void* stream) {
        if (!can_release_owner()) return false;
        // Do not trust a caller's capacity/alignment assertions over the actual supplied spans.
        input.capacity = capacities(binding);
        input.scratch_aligned = true;
        const Plan candidate = plan(input);
        if (candidate.outcome != Outcome::direct_candidate || !binding_valid(binding, candidate.bytes.envelope)) return false;
        plan_ = candidate; binding_ = binding; tokens_ = input.actual_tokens; stream_ = stream; state_ = State::ready;
        return true;
    }

    // Locks the representation and addresses before exposing any writable GPU pointer. There is intentionally
    // no "fallback after acquire" or unconditional cancel/reset operation. Even cancellation retires a fence.
    bool acquire(Views& out) {
        if (state_ != State::ready) return false;
        Views v;
        v.x = {static_cast<uint16_t*>(binding_.xq.address), plan_.bytes.direct.xq / sizeof(uint16_t)};
        v.h = {static_cast<uint16_t*>(binding_.h.address), plan_.bytes.direct.h / sizeof(uint16_t)};
        v.dm = {static_cast<float*>(binding_.dm.address), plan_.bytes.direct.dm / sizeof(float)};
        v.groups = {binding_.gu.address, plan_.bytes.direct.gu};
        v.mode = plan_.selected; v.tokens = tokens_; v.stream = stream_;
        state_ = State::acquired; out = v; return true;
    }

    bool retire(const Fence& done) {
        if (state_ != State::acquired || !done.event || !done.record || !done.query) return false;
        if (!done.record(done.user, done.event, stream_)) return false;
        fence_ = done; state_ = State::pending; return true;
    }

    Completion poll() {
        if (state_ == State::poisoned) return Completion::error;
        if (state_ != State::pending) return state_ == State::ready ? Completion::complete : Completion::pending;
        const Completion result = fence_.query(fence_.user, fence_.event);
        if (result == Completion::complete) { fence_ = {}; state_ = State::ready; }
        else if (result != Completion::pending) state_ = State::poisoned;
        return state_ == State::poisoned ? Completion::error : result;
    }

    // Call before capture, retaining one reference for every graph that embeds these addresses. release_graph
    // is permitted only after all tracked GPU use is complete. Capture/replay still needs the real HIP adapter.
    bool retain_graph() {
        if (state_ != State::ready || graphs_ == SIZE_MAX) return false;
        ++graphs_; return true;
    }
    bool release_graph() {
        if (state_ != State::ready || !graphs_) return false;
        --graphs_; return true;
    }

    // Restore must create/rebind the DESTINATION context from its own allocator. Never serialize this object,
    // Views, Binding or Fence. Drop is only a host bookkeeping release; it does not free device allocations.
    bool drop() {
        if (!can_release_owner()) return false;
        plan_ = {}; binding_ = {}; fence_ = {}; tokens_ = 0; stream_ = nullptr; state_ = State::unbound;
        return true;
    }

private:
    State state_ = State::unbound;
    size_t graphs_ = 0, tokens_ = 0;
    void* stream_ = nullptr;
    Plan plan_;
    Binding binding_;
    Fence fence_;
};

}  // namespace strata::prefill::evo_fp16
