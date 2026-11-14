// Executes the real ExpertCache allocation path using CPU fault-injected CUDA
// calls. No device, HIP compilation, UMA observation or GPU ordering is tested.
#include "strata/core/expert_cache.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <functional>

using namespace strata::core;
namespace {
int checks = 0, allocations = 0, frees = 0;
bool fail_info = false, fail_alloc = false, fail_memset = false, fail_free = false;
size_t free_bytes = 1 << 20, total_bytes = 1 << 20;
std::function<void(size_t)> before_allocate;
void check(bool pass, const char* why) {
    ++checks;
    if (!pass) throw std::runtime_error(why);
}
std::shared_ptr<EvoMemoryPlan> plan(uint64_t cap = 1024) {
    EvoBudget b;
    b.host_total_bytes = 16ull << 30;
    b.host_other_peak_known = true;
    b.device_addressable_bytes = cap;
    return std::make_shared<EvoMemoryPlan>(b);
}
EvoBacking backing(std::string id, uint64_t bytes) {
    EvoBacking b;
    b.backing_id = id;
    b.owner = "fake-expert-cache";
    b.purpose = "host-test-arena";
    b.lifetime = "cache-close";
    b.physical_bytes = bytes;
    b.device_bytes = bytes;
    return b;
}
bool bind_plan(ExpertCache& c, const std::shared_ptr<EvoMemoryPlan>& p,
          const char* id, uint64_t bytes, std::string& err) {
    return c.set_evo_memory_plan(p, backing(id, bytes), id, err);
}
}
cudaError_t cudaMalloc(void** p, size_t n) {
    ++allocations;
    if (before_allocate) before_allocate(n);
    *p = fail_alloc ? nullptr : std::malloc(n);
    return *p ? cudaSuccess : 1;
}
cudaError_t cudaFree(void* p) {
    ++frees;
    if (fail_free) return 1;
    std::free(p); return cudaSuccess;
}
cudaError_t cudaMemGetInfo(size_t* f, size_t* t) {
    *f = free_bytes; *t = total_bytes; return fail_info ? 1 : cudaSuccess;
}
cudaError_t cudaMemset(void* p, int v, size_t n) {
    if (fail_memset) return 1;
    std::memset(p, v, n); return cudaSuccess;
}
cudaError_t cudaGetLastError() { return 1; }
const char* cudaGetErrorString(cudaError_t) { return "injected failure"; }
cudaError_t cudaMemcpy(void* to, const void* from, size_t n, int) {
    std::memcpy(to, from, n); return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* to, const void* from, size_t n, int k, cudaStream_t) {
    return cudaMemcpy(to, from, n, k);
}
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

int main() {
    std::string err;
    {
        ExpertCache c;
        check(!c.set_evo_memory_plan(nullptr, backing("x", 512), "x", err), "null plan");
        auto p = plan();
        check(!c.set_evo_memory_plan(p, backing("x", 512), "", err), "empty reference");
        check(bind_plan(c, p, "x", 512, err), "bind");
        before_allocate = [&](size_t n) {
            check(n == 512 && p->snapshot().reserved_bytes == 512, "reserve precedes allocation");
            check(p->snapshot().allocated_bytes == 0, "not committed before allocation");
        };
        check(c.open(2, 2, 2, 256, err), "guarded open");
        before_allocate = {};
        check(p->snapshot().allocated_bytes == 512 && p->snapshot().reserved_bytes == 0, "committed");
        check(!bind_plan(c, p, "y", 512, err), "cannot rebind live cache");
        check(c.bytes() == 512 && c.slots() == 2, "correct arena");
        c.close();
        check(p->snapshot().physical_bytes == 0 && !c.valid(), "release after free");
        check(c.open(2, 2, 2, 256, err), "reopen after close");
    }
    {
        auto p = plan(); ExpertCache c;
        check(bind_plan(c, p, "x", 512, err), "bind errors");
        int n = allocations;
        check(!c.open(3, 2, 2, 256, err) && allocations == n, "exact footprint required");
        fail_info = true;
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "unknown capacity rejected");
        fail_info = false;
        total_bytes = 0;
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "unknown total rejected");
        total_bytes = 1 << 20;
        free_bytes = total_bytes + 1;
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "inconsistent capacity rejected");
        free_bytes = 511;
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "insufficient device bytes");
        free_bytes = 1 << 20;
        c.set_segment_bytes(256);
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "segmented ownership rejected");
        c.set_segment_bytes(0); ExpertCache::set_vmm(true);
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "VMM ownership rejected");
        ExpertCache::set_vmm(false);
        check(!c.open(std::numeric_limits<int64_t>::max(), 2, 2, 2, err) && allocations == n, "byte overflow");
        check(!c.open(2, std::numeric_limits<int64_t>::max(), 2, 256, err) && allocations == n, "residency overflow");
        check(!c.open(0, 2, 2, 256, err) && allocations == n, "zero slots");
        check(!c.open(2, 2, 2, -1, err) && allocations == n, "negative blob");
        check(p->snapshot().physical_bytes == 0, "failures never reserve");
        before_allocate = [](size_t) { throw std::bad_alloc(); };
        bool threw = false;
        try { c.open(2, 2, 2, 256, err); } catch (const std::bad_alloc&) { threw = true; }
        before_allocate = {};
        check(threw && p->snapshot().physical_bytes == 0 && !c.valid(), "exception cancels reservation");
        fail_alloc = true;
        check(!c.open(2, 2, 2, 256, err), "allocation failure"); fail_alloc = false;
        check(p->snapshot().physical_bytes == 0 && !c.valid(), "allocation failure cancels reservation");
        fail_memset = true;
        check(!c.open(2, 2, 2, 256, err), "zeroing failure"); fail_memset = false;
        check(p->snapshot().physical_bytes == 0 && !c.valid(), "zeroing failure frees arena");
        check(c.open(2, 2, 2, 256, err), "open before failed free");
        fail_free = true; c.close();
        check(c.valid() && p->snapshot().allocated_bytes == 512, "unknown release retains charge");
        n = allocations;
        check(!c.open(2, 2, 2, 256, err) && allocations == n, "failed release blocks reopen");
        fail_free = false; c.close();
        check(p->snapshot().physical_bytes == 0 && !c.valid(), "confirmed free releases charge");
    }
    {
        auto p = plan(768); ExpertCache a, b, duplicate;
        check(bind_plan(a, p, "a", 512, err) && bind_plan(b, p, "b", 512, err), "shared plan");
        before_allocate = [&](size_t) {
            const int n = allocations;
            check(!b.open(2, 1, 1, 256, err) && allocations == n,
                  "pending reservation prevents overlapping allocation");
        };
        check(a.open(2, 1, 1, 256, err), "first shared allocation");
        before_allocate = {};
        int n = allocations;
        check(!b.open(2, 1, 1, 256, err) && allocations == n, "no double booking between caches");
        check(bind_plan(duplicate, p, "a", 512, err), "duplicate bind");
        check(!duplicate.open(2, 1, 1, 256, err) && allocations == n, "duplicate backing cannot undercount");
        a.close();
        check(b.open(2, 1, 1, 256, err), "released capacity reusable");
    }
    {
        auto p = plan(2048); ExpertCache c;
        c.set_slot_guard(256);
        check(bind_plan(c, p, "sized", 1280, err), "sized bind");
        check(c.open_sized({100, 257}, 1, 1, err), "sized open");
        check(c.bytes() == 1280 && c.slots() == 2 && c.slot_offsets()[1] == 512, "padding and guards charged");
        check(p->snapshot().allocated_bytes == 1280, "exact sized arena committed");
        c.close();
        const int n = allocations;
        check(!c.open_sized({-1}, 1, 1, err) && allocations == n, "negative sized blob");
        check(!c.open_sized({std::numeric_limits<int64_t>::max()}, 1, 1, err) && allocations == n, "sized overflow");
        c.set_slot_guard(std::numeric_limits<int64_t>::max());
        check(!c.open_sized({1}, 1, 1, err) && allocations == n, "guard overflow");
        c.set_slot_guard(0);
        const auto half = std::numeric_limits<int64_t>::max() / 2;
        check(!c.open_sized({half, half, half}, 1, 1, err) && allocations == n, "summed size overflow");
    }
    {
        auto p = plan(); ExpertCache c;
        check(bind_plan(c, p, "pressure", 512, err), "pressure bind");
        p->observe_headroom(false, 0, 0);
        int n = allocations;
        check(!c.open(2, 1, 1, 256, err) && allocations == n, "unknown host headroom blocks allocation");
    }
    {
        ExpertCache c;
        fail_info = true;
        check(c.open(2, 1, 1, 256, err), "unbound upstream unknown-query fallback unchanged");
        fail_info = false;
    }
    check(frees > 0, "real close path exercised");
    std::cout << checks << " ExpertCache CPU-injected allocation checks; GPU NOT RUN\n";
}
