#include "strata/core/evo_memory_plan.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace strata::core;
constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t MiB = 1ull << 20;
int checks = 0;
void check(bool yes, const char* label) {
    ++checks;
    if (!yes) { std::cerr << "FAIL: " << label << '\n'; std::exit(1); }
}
template<class F> void rejects(F f, const char* label) {
    try { f(); } catch (const std::exception&) { check(true, label); return; }
    check(false, label);
}
EvoBudget budget() {
    return {124 * GiB, 16 * GiB, true, 100 * GiB,
            {{"parent", 100 * GiB, 20 * GiB, true}, {"child", 90 * GiB, 0, true}}};
}
EvoBacking backing(std::string id, uint64_t physical, uint64_t device = 0) {
    return {id, "session-0", "KV", "request", physical, device,
            {{"parent", physical}, {"child", physical}}, false};
}
int main() {
    check(EvoMemoryPlan::host_reserve(124 * GiB) == 12 * GiB + 512 * MiB, "124 GiB reserve is 12.5 GiB");
    check(EvoMemoryPlan::host_reserve(120 * GiB + 1) == 12 * GiB + 256 * MiB, "fraction rounded up not down");
    check(EvoMemoryPlan::cgroup_reserve(30 * GiB) == GiB + 512 * MiB, "cgroup uses independent five percent");
    check(EvoMemoryPlan::cgroup_reserve(GiB) == GiB, "cgroup one GiB floor");
    auto b = budget();
    EvoMemoryPlan plan(b);
    check(plan.host_limit() == 95 * GiB + 512 * MiB, "host excludes no extra cgroup reserve");
    rejects([] { auto b = budget(); b.host_other_peak_known = false; EvoMemoryPlan p(b); }, "unknown other services rejected");
    rejects([] { auto b = budget(); b.host_other_peak_bytes = 124 * GiB; EvoMemoryPlan p(b); }, "negative budget rejected");
    rejects([] { auto b = budget(); b.cgroups[1].other_peak_known = false; EvoMemoryPlan p(b); }, "unknown cgroup demand rejected");
    rejects([] { auto b = budget(); b.cgroups.push_back(b.cgroups[0]); EvoMemoryPlan p(b); }, "duplicate ancestor rejected");
    std::string why;
    check(plan.reserve("main", backing("kv", 40 * GiB, 40 * GiB), why), "reserve before allocation");
    check(plan.snapshot().reserved_bytes == 40 * GiB, "reservation already consumes budget");
    check(plan.commit("main", why), "allocation commits existing reservation");
    check(!plan.commit("main", why), "double commit rejected");
    check(plan.alias("device-view", "main", why), "explicit alias shares same backing");
    check(plan.snapshot().physical_bytes == 40 * GiB && plan.snapshot().references == 2, "host device alias counted once");
    check(!plan.reserve("duplicate", backing("kv", 40 * GiB), why), "same backing cannot be counted independently");
    check(!plan.reserve("sibling", backing("too-large", 36 * GiB), why), "parent sibling charge prevents otherwise fitting child");
    check(why.find("parent") != std::string::npos, "limiting ancestor identified");
    check(plan.snapshot().backings == 1, "failed reservation is atomic");
    auto uncharged = backing("driver", GiB);
    uncharged.cgroup_charge_bytes.erase("child");
    check(!plan.reserve("driver", uncharged, why), "missing charge is not zero");
    uncharged.cgroup_charge_bytes = {{"parent", 0}, {"child", 0}};
    check(plan.reserve("driver", uncharged, why), "observed zero charge does not erase host bytes");
    check(plan.reserve("empty", backing("cpu-complement", 0), why), "zero-byte complement valid");
    check(plan.commit("empty", why), "zero-byte complement commits");
    check(plan.release("main"), "release first shared owner");
    check(plan.snapshot().physical_bytes == 41 * GiB, "last owner keeps backing alive");
    check(plan.release("device-view"), "release final shared owner");
    check(plan.snapshot().physical_bytes == GiB, "last owner releases backing");
    check(!plan.release("main"), "double release detected");
    check(!plan.reserve("overflow", backing("overflow", std::numeric_limits<uint64_t>::max()), why), "overflow rejected");
    check(!plan.observe_headroom(true, 12 * GiB, 1), "admission denied immediately below reserve");
    check(!plan.reserve("during-pressure", backing("future-work", 1), why), "no new allocation during pressure");
    check(!plan.sustained_pressure(5000000000ull), "five-second threshold is not admission grace");
    check(plan.sustained_pressure(5000000001ull), "five-second sustained pressure");
    check(plan.observe_headroom(true, 14 * GiB, 6000000000ull), "recovery clears pressure with future room");
    check(!plan.sustained_pressure(20000000000ull), "no stale pressure after recovery");
    check(!plan.observe_headroom(false, 124 * GiB, 20000000000ull), "unknown headroom cannot admit");
    check(!plan.observe_headroom(true, plan.host_reserve_bytes(), 21000000000ull), "pending reservations need future headroom");
    check(!plan.reserve("unfunded", backing("unfunded", 1), why), "reserve cannot spend final headroom byte");
    check(plan.observe_headroom(true, plan.host_reserve_bytes() + 2 * GiB, 22000000000ull), "fresh measured growth room");
    check(plan.reserve("future", backing("future", GiB), why), "reserve future workspace");
    check(plan.commit("future", why), "commit future workspace");
    check(!plan.reserve("stale", backing("stale", GiB + 1), why), "commit does not recycle stale measured availability");
    check(plan.release("future"), "release actual workspace");
    check(!plan.reserve("still-stale", backing("still-stale", GiB + 1), why), "free waits for refreshed host observation");
    // Independent host and addressable limits, not RAM + VRAM.
    auto small = budget(); small.device_addressable_bytes = GiB;
    EvoMemoryPlan device(small);
    check(!device.reserve("gpu", backing("gpu", 2 * GiB, 2 * GiB), why), "addressable limit separate from host budget");
    auto host = budget(); host.host_other_peak_bytes = 110 * GiB;
    EvoMemoryPlan narrow(host);
    check(!narrow.reserve("host", backing("host", 2 * GiB), why), "physical memory limit independent of charges");
    // All 32 concurrent reservations cannot each observe the same free space.
    auto race_budget = budget(); race_budget.host_other_peak_bytes = 110 * GiB;
    EvoMemoryPlan race(race_budget);
    std::vector<std::thread> threads;
    for (int i = 0; i < 32; ++i) threads.emplace_back([&, i] {
        std::string reason;
        race.reserve(std::to_string(i), backing(std::to_string(i), 256 * MiB), reason);
    });
    for (auto& t : threads) t.join();
    check(race.snapshot().physical_bytes == 1536 * MiB && race.snapshot().backings == 6, "atomic concurrent admission cannot overbook");
    std::cout << checks << " EVO memory checks passed\n";
}
