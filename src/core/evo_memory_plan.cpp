#include "strata/core/evo_memory_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::core {
namespace {
constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t quantum = 256ull << 20;
uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("EVO byte count overflow");
    return a + b;
}
uint64_t round_reserve(uint64_t n) {
    const auto units = n / quantum + (n % quantum != 0);
    if (units > std::numeric_limits<uint64_t>::max() / quantum) throw std::overflow_error("EVO reserve overflow");
    return units * quantum;
}
uint64_t ceiling_fraction(uint64_t n, uint64_t denominator) {
    return n / denominator + (n % denominator != 0);
}
uint64_t remaining(uint64_t limit, uint64_t other, uint64_t reserve) {
    if (other > limit || reserve > limit - other) throw std::invalid_argument("EVO budget is negative");
    return limit - other - reserve;
}
}

uint64_t EvoMemoryPlan::host_reserve(uint64_t total) {
    return round_reserve(std::max(12 * GiB, ceiling_fraction(total, 10)));
}
uint64_t EvoMemoryPlan::cgroup_reserve(uint64_t limit) {
    return round_reserve(std::max(GiB, ceiling_fraction(limit, 20)));
}

EvoMemoryPlan::EvoMemoryPlan(const EvoBudget& b)
    : host_reserve_(host_reserve(b.host_total_bytes)),
      host_limit_(remaining(b.host_total_bytes, b.host_other_peak_bytes, host_reserve_)),
      device_limit_(b.device_addressable_bytes) {
    if (!b.host_other_peak_known || !b.host_total_bytes || !device_limit_)
        throw std::invalid_argument("EVO host and device budgets require observed values");
    for (const auto& group : b.cgroups) {
        if (group.id.empty() || !group.limit_bytes || !group.other_peak_known)
            throw std::invalid_argument("EVO cgroup budget requires identity and observed values");
        const auto cap = remaining(group.limit_bytes, group.other_peak_bytes, cgroup_reserve(group.limit_bytes));
        if (!cgroup_limits_.emplace(group.id, cap).second)
            throw std::invalid_argument("duplicate EVO cgroup identity");
    }
}

EvoLedgerSnapshot EvoMemoryPlan::snapshot_locked() const {
    EvoLedgerSnapshot s;
    s.backings = entries_.size();
    s.references = references_.size();
    for (const auto& group : cgroup_limits_) s.cgroup_charge_bytes[group.first] = 0;
    for (const auto& item : entries_) {
        const auto& e = item.second;
        const auto& b = e.backing;
        s.physical_bytes = add(s.physical_bytes, b.physical_bytes);
        s.device_bytes = add(s.device_bytes, b.device_bytes);
        auto& phase = e.allocated ? s.allocated_bytes : s.reserved_bytes;
        phase = add(phase, b.physical_bytes);
        for (const auto& group : b.cgroup_charge_bytes)
            s.cgroup_charge_bytes[group.first] = add(s.cgroup_charge_bytes[group.first], group.second);
    }
    return s;
}

bool EvoMemoryPlan::reserve(const std::string& ref, const EvoBacking& b, std::string& why) {
    std::lock_guard<std::mutex> lock(mutex_);
    why.clear();
    if (ref.empty() || b.backing_id.empty() || b.owner.empty() || b.purpose.empty() || b.lifetime.empty()) {
        why = "allocation identity, owner, purpose and lifetime are required"; return false;
    }
    if (pressure_) { why = "host headroom is unknown or below reserve"; return false; }
    if (references_.count(ref) || entries_.count(b.backing_id)) {
        why = "duplicate reference or backing; use alias for shared ownership"; return false;
    }
    if (b.device_bytes > b.physical_bytes || b.cgroup_charge_bytes.size() != cgroup_limits_.size()) {
        why = "incomplete or inconsistent allocation accounting"; return false;
    }
    try {
        auto s = snapshot_locked();
        if (headroom_observed_ && (incremental_reserved_ > observed_room_ ||
                b.physical_bytes > observed_room_ - incremental_reserved_)) {
            why = "new allocation would consume observed host reserve"; return false;
        }
        if (add(s.physical_bytes, b.physical_bytes) > host_limit_) { why = "host physical budget exceeded"; return false; }
        if (add(s.device_bytes, b.device_bytes) > device_limit_) { why = "GPU addressable budget exceeded"; return false; }
        for (const auto& group : cgroup_limits_) {
            auto charge = b.cgroup_charge_bytes.find(group.first);
            if (charge == b.cgroup_charge_bytes.end()) { why = "missing cgroup ancestor charge"; return false; }
            if (add(s.cgroup_charge_bytes.at(group.first), charge->second) > group.second) {
                why = "cgroup budget exceeded: " + group.first; return false;
            }
        }
    } catch (const std::overflow_error&) { why = "allocation byte count overflow"; return false; }
    entries_.emplace(b.backing_id, Entry{b, false, 1});
    try {
        references_.emplace(ref, b.backing_id);
    } catch (...) {
        entries_.erase(b.backing_id);
        throw;
    }
    if (headroom_observed_) incremental_reserved_ += b.physical_bytes;
    return true;
}

bool EvoMemoryPlan::commit(const std::string& ref, std::string& why) {
    std::lock_guard<std::mutex> lock(mutex_);
    why.clear();
    auto found = references_.find(ref);
    if (found == references_.end()) { why = "unknown reservation"; return false; }
    auto& entry = entries_.at(found->second);
    if (entry.allocated) { why = "reservation is already committed"; return false; }
    entry.allocated = true;
    return true;
}

bool EvoMemoryPlan::alias(const std::string& ref, const std::string& existing, std::string& why) {
    std::lock_guard<std::mutex> lock(mutex_);
    why.clear();
    auto found = references_.find(existing);
    if (ref.empty() || references_.count(ref) || found == references_.end()) {
        why = "alias requires a new reference and an existing backing"; return false;
    }
    references_.emplace(ref, found->second);
    ++entries_.at(found->second).refs;
    return true;
}

bool EvoMemoryPlan::release(const std::string& ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = references_.find(ref);
    if (found == references_.end()) return false;
    auto entry = entries_.find(found->second);
    if (--entry->second.refs == 0) {
        // Never assume freeing an allocated buffer immediately increases the
        // OS's observed availability. Reclaim that allowance at the next sample.
        if (headroom_observed_ && !entry->second.allocated)
            incremental_reserved_ -= entry->second.backing.physical_bytes;
        entries_.erase(entry);
    }
    references_.erase(found);
    return true;
}

EvoLedgerSnapshot EvoMemoryPlan::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_locked();
}

bool EvoMemoryPlan::observe_headroom(bool known, uint64_t available, uint64_t now) {
    std::lock_guard<std::mutex> lock(mutex_);
    headroom_observed_ = true;
    // Existing allocations are already reflected in MemAvailable. Only pending
    // future allocations are charged again; committed growth after this sample
    // remains reserved until the next reading.
    incremental_reserved_ = snapshot_locked().reserved_bytes;
    observed_room_ = known && available >= host_reserve_ ? available - host_reserve_ : 0;
    if (!known || available < host_reserve_) {
        if (!pressure_ || now < pressure_since_) pressure_since_ = now;
        pressure_ = true;
        return false;
    }
    pressure_ = false;
    return incremental_reserved_ <= observed_room_;
}

bool EvoMemoryPlan::sustained_pressure(uint64_t now) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pressure_ && now >= pressure_since_ && now - pressure_since_ >= 5000000000ull;
}
}  // namespace strata::core
