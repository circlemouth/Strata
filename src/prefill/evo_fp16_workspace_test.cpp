// CPU-only tests with opaque synthetic device addresses. No pointer is dereferenced or sent to a GPU.
#include "strata/prefill/evo_fp16_workspace.hpp"
#include <cstdio>
#include <type_traits>

using namespace strata::prefill::evo_fp16;
static int failures = 0, checks = 0;
static void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
static Input fixture() {
    Input i;
    i.requested = Mode::pair128;
    i.device = {true, true, true, "gfx1151", 32};
    i.actual_tokens = i.upstream_priced_tokens = 1024;
    i.ring = 128;
    i.has_source = i.native_layout = i.no_peer = i.mmq_or_fused_only_plan = true;
    i.direct64_built = i.pair128_built = true;
    i.scratch_context_owned = true;
    const auto& g = i.geometry;
    i.format.gu_type = GGML_TYPE_IQ3_XXS; i.format.d_type = GGML_TYPE_IQ4_NL;
    i.format.gu_row = size_t(g.n_embd / 256) * 98;
    i.format.d_row = size_t(g.n_ff / 32) * 18;
    i.format.up_off = size_t(g.n_ff) * i.format.gu_row;
    i.format.down_off = 2 * i.format.up_off;
    i.min_blob_bytes = i.format.down_off + size_t(g.n_embd) * i.format.d_row;
    i.common_blob_alignment = 16;
    const size_t rows = i.actual_tokens * size_t(g.n_expert_used);
    i.upstream_required = {rows * size_t(2 * g.n_ff) * 4, rows * size_t(g.n_ff) * 4,
                           rows * size_t(g.n_embd) * 2, rows * size_t(g.n_ff) * 2,
                           rows * size_t(g.n_embd) * 4};
    return i;
}
static Binding fake_binding(const Input& i, uintptr_t base) {
    Workspace w;
    if (!workspace(i.actual_tokens, i.geometry, i.upstream_required, w)) std::abort();
    auto region = [&](size_t n) {
        Region r{reinterpret_cast<void*>(base), n};
        size_t span;
        if (!allocation_take_bytes(n, span) || span > UINTPTR_MAX - base) std::abort();
        base += span; return r;
    };
    Binding b;
    b.gu = region(w.envelope.gu); b.h = region(w.envelope.h); b.xq = region(w.envelope.xq);
    b.hq = region(w.envelope.hq); b.dm = region(w.envelope.dm);
    return b;
}
static void* const stream1 = reinterpret_cast<void*>(uintptr_t(0x10100));
static void* const stream2 = reinterpret_cast<void*>(uintptr_t(0x20200));
struct Event {
    Completion status = Completion::pending;
    unsigned queries = 0, records = 0;
    void* expected_stream = stream1;
    bool record_ok = true;
};
static bool record(void* user, const void* event, void* stream) noexcept {
    if (user != event) return false;
    auto* e = static_cast<Event*>(user); ++e->records;
    return e->record_ok && stream == e->expected_stream;
}
static Completion query(void* user, const void* event) noexcept {
    if (user != event) return Completion::error;
    auto* e = static_cast<Event*>(user); ++e->queries; return e->status;
}
static Fence fence(Event& e) { return {&e, &e, record, query}; }

static_assert(!std::is_copy_constructible_v<WorkspaceGuard> && !std::is_move_constructible_v<WorkspaceGuard>);
static_assert(!std::is_convertible_v<Fp16Input, const void*> && !std::is_convertible_v<Fp16Output, void*>);
static_assert(!std::is_convertible_v<Fp16Output, Fp32Output>);
static_assert(std::is_same_v<decltype(Views{}.dm.data), float*>);
static_assert(std::is_same_v<decltype(Views{}.x_input().data), const uint16_t*>);

int main() {
    size_t bytes = 777;
    for (size_t payload : {size_t(0), size_t(1), size_t(255), size_t(256), size_t(257), size_t(8192)}) {
        expect(allocation_take_bytes(payload, bytes), "normal allocation footprint");
        // Independently repeated literal pinned Alloc::take formula, deliberately not the planner's align helper.
        expect(bytes == ((uint64_t(payload) + 256 + 255) & ~uint64_t(255)), "matches pinned Alloc extra guard exactly");
    }
    expect(!allocation_take_bytes(SIZE_MAX, bytes), "guard overflow rejected");
    expect(!allocation_take_bytes(SIZE_MAX - 300, bytes), "rounding overflow rejected");
    expect(allocation_bytes({}, bytes) && bytes == 1280, "five zero-sized takes still have five guards");
    Buffers huge; huge.gu = SIZE_MAX - 511;
    expect(!allocation_bytes(huge, bytes), "aggregate allocation overflow rejected");
    auto i = fixture();
    const auto b = fake_binding(i, uintptr_t(0x10000000));
    const auto b2 = fake_binding(i, uintptr_t(0x90000000));
    Workspace w;
    expect(workspace(i.actual_tokens, i.geometry, i.upstream_required, w), "host workspace prices");
    expect(allocation_bytes(w.envelope, bytes) && bytes == w.total + 1280, "allocator guards counted beyond logical envelope");
    expect(binding_valid(b, w.envelope), "disjoint exact capacities bind");
    { auto bad = b; --bad.xq.capacity; expect(!binding_valid(bad, w.envelope), "one byte short"); }
    { auto bad = b; bad.dm.address = nullptr; expect(!binding_valid(bad, w.envelope), "nonempty null span"); }
    { auto bad = b; bad.h.address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(b.h.address) + 1); expect(!binding_valid(bad, w.envelope), "misaligned actual pointer"); }
    { auto bad = b; bad.xq.address = bad.h.address; expect(!binding_valid(bad, w.envelope), "exact overlap"); }
    { auto bad = b; bad.xq.address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(b.h.address) + 256); expect(!binding_valid(bad, w.envelope), "partial overlap"); }
    { auto bad = b; bad.dm.address = reinterpret_cast<void*>(UINTPTR_MAX & ~uintptr_t(255)); expect(!binding_valid(bad, w.envelope), "address end wraps"); }
    { Binding empty; expect(binding_valid(empty, {}), "zero capacity is not an address range"); empty.hq.address = reinterpret_cast<void*>(1); expect(binding_valid(empty, {}), "empty hq pointer never used"); }
    // Pairwise exhaustive aliases: failure is independent of declared member order.
    for (size_t a = 0; a < 5; ++a) for (size_t z = 0; z < 5; ++z) if (a != z) {
        auto bad = b; Region* fields[] = {&bad.gu, &bad.h, &bad.xq, &bad.hq, &bad.dm};
        fields[a]->address = fields[z]->address;
        expect(!binding_valid(bad, w.envelope), "all region alias pairs rejected");
    }
    WorkspaceGuard first, second;
    Views v;
    expect(first.can_release_owner() && first.state() == State::unbound, "initially no GPU lifetime");
    expect(!first.acquire(v) && !first.retain_graph(), "unbound cannot expose views or capture");
    { auto bad = i; bad.scratch_context_owned = false; expect(!first.bind(bad, b, stream1), "ownership is separately mandatory"); }
    { auto bad = i; bad.capacity = w.envelope; auto short_b = b; short_b.dm.capacity = 0; expect(!first.bind(bad, short_b, stream1), "actual spans override optimistic capacity assertion"); }
    { auto bad = i; bad.pair128_built = false; expect(!first.bind(bad, b, stream1), "no unavailable kernel can bind"); }
    { auto bad = i; bad.requested = Mode::upstream; expect(!first.bind(bad, b, stream1), "upstream cannot receive direct views"); }
    expect(first.bind(i, b, stream1) && second.bind(i, b2, stream2), "two distinct context-local bindings");
    expect(first.retain_graph() && first.retain_graph() && first.retained_graphs() == 2, "capture retains stable addresses");
    expect(!first.bind(i, b2, stream1) && !first.drop() && !first.can_release_owner(), "graph pins prevent rebind release and drop");
    expect(first.acquire(v), "acquire exposes typed direct views");
    expect(v.x.data == b.xq.address && v.h.data == b.h.address && v.dm.data == b.dm.address, "views preserve actual bound addresses");
    expect(v.x.values == w.direct.xq / 2 && v.h.values == w.direct.h / 2 && v.dm.values == w.direct.dm / 4, "views bounded by direct payload, not oversized fallback");
    expect(v.groups.bytes == w.direct.gu && v.mode == Mode::pair128 && v.tokens == i.actual_tokens && v.stream == stream1, "views retain fixed layer representation");
    expect(!first.acquire(v) && !first.drop() && !first.bind(i, b2, stream1), "active layer cannot be overlapped or rebound");
    expect(!first.retire({}) && first.state() == State::acquired, "missing recorded-event interface fails closed");
    Event e1, e2; e2.expected_stream = stream2;
    e1.record_ok = false;
    expect(!first.retire(fence(e1)) && first.state() == State::acquired && !first.can_release_owner(), "failed event record does not unlock scratch");
    e1.record_ok = true;
    expect(!first.retire(fence(e2)) && first.state() == State::acquired, "record receives exactly the bound stream");
    expect(first.retire(fence(e1)) && !first.retire(fence(e1)), "retirement exactly once");
    expect(e1.records == 2, "retire invokes record, never trusts an already-complete event alone");
    expect(first.poll() == Completion::pending && e1.queries == 1, "real adapter must decide completion");
    expect(!first.acquire(v) && !first.drop() && !first.release_graph(), "cancel does not release pending GPU buffers");
    Views v2;
    expect(second.acquire(v2) && v2.x.data == b2.xq.address && second.retire(fence(e2)), "second context independent");
    e2.status = Completion::complete;
    expect(second.poll() == Completion::complete && second.drop(), "completed second context may release");
    expect(first.poll() == Completion::pending && first.retained_graphs() == 2, "second completion cannot release first context");
    e1.status = Completion::complete;
    expect(first.poll() == Completion::complete && first.state() == State::ready, "completion returns same bindings to ready");
    expect(!first.can_release_owner(), "completed work does not destroy retained graph addresses");
    expect(first.release_graph() && first.release_graph() && !first.release_graph(), "release every graph exactly once");
    expect(first.can_release_owner() && first.drop(), "only now owner may free or resize");
    // Destination restore uses its allocator, not a pointer copied from a source snapshot.
    expect(first.bind(i, b2, stream1) && first.acquire(v) && v.x.data == b2.xq.address, "restore destination binding replaces old addresses");
    e1.status = Completion::complete;
    expect(first.retire(fence(e1)) && first.poll() == Completion::complete && first.drop(), "restored binding retires normally");
    WorkspaceGuard error;
    expect(error.bind(i, b, stream1) && error.acquire(v) && error.retire(fence(e1)), "error path acquired");
    e1.status = Completion::error;
    expect(error.poll() == Completion::error && error.state() == State::poisoned, "event-query error is sticky");
    e1.status = Completion::complete;
    expect(error.poll() == Completion::error && !error.drop() && !error.bind(i, b2, stream1) && !error.can_release_owner(), "error cannot silently authorize reuse");
    // No allocations are owned in these tests, including error. Production must quarantine failed contexts and
    // resolve backend completion before destruction/free; this guard cannot synchronize an absent HIP backend.
    std::printf("evo_fp16_workspace_test: %d checks, %d failures (host binding contract only)\n", checks, failures);
    return failures ? 1 : 0;
}
