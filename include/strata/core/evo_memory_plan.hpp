// Byte-only UMA planning primitive. Not an allocator or a hardware qualification.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

struct EvoCgroupBudget {
    std::string id;  // Each finite ancestor is an independent constraint.
    uint64_t limit_bytes = 0;
    uint64_t other_peak_bytes = 0;
    bool other_peak_known = false;
};

struct EvoBudget {
    uint64_t host_total_bytes = 0;
    uint64_t host_other_peak_bytes = 0;
    bool host_other_peak_known = false;
    uint64_t device_addressable_bytes = 0;
    std::vector<EvoCgroupBudget> cgroups;
};

struct EvoBacking {
    std::string backing_id;
    std::string owner;
    std::string purpose;
    std::string lifetime;
    uint64_t physical_bytes = 0;
    uint64_t device_bytes = 0;
    // Every finite ancestor must be named, including measured zero charges.
    // Device allocation bytes cannot substitute for measured cgroup charges.
    std::map<std::string, uint64_t> cgroup_charge_bytes;
    bool reclaimable = false;
};

struct EvoLedgerSnapshot {
    uint64_t physical_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t reserved_bytes = 0;
    uint64_t allocated_bytes = 0;
    std::map<std::string, uint64_t> cgroup_charge_bytes;
    size_t backings = 0;
    size_t references = 0;
};

// Callers must connect every relevant allocation and its future workspace before
// claiming enforcement. The optional ExpertCache binding covers only its device arena; it is not
// enabled by a runtime caller and is not whole-engine enforcement.
class EvoMemoryPlan {
public:
    explicit EvoMemoryPlan(const EvoBudget& budget);
    static uint64_t host_reserve(uint64_t total);
    static uint64_t cgroup_reserve(uint64_t limit);
    uint64_t host_limit() const { return host_limit_; }
    uint64_t host_reserve_bytes() const { return host_reserve_; }

    // Atomic check and reservation; failure leaves the ledger unchanged.
    bool reserve(const std::string& reference, const EvoBacking& backing, std::string& why);
    bool commit(const std::string& reference, std::string& why);
    bool alias(const std::string& reference, const std::string& existing_reference, std::string& why);
    bool release(const std::string& reference);
    EvoLedgerSnapshot snapshot() const;

    // Live admission is denied immediately below reserve, never after a grace
    // period. Five seconds only classifies sustained pressure. Unknown readings
    // (known=false) also deny admission without inventing a measurement.
    bool observe_headroom(bool known, uint64_t available, uint64_t monotonic_ns);
    bool sustained_pressure(uint64_t monotonic_ns) const;

private:
    struct Entry { EvoBacking backing; bool allocated = false; size_t refs = 1; };
    EvoLedgerSnapshot snapshot_locked() const;
    mutable std::mutex mutex_;
    uint64_t host_reserve_, host_limit_, device_limit_;
    std::map<std::string, uint64_t> cgroup_limits_;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::string> references_;
    bool pressure_ = false;
    uint64_t pressure_since_ = 0;
    bool headroom_observed_ = false;
    uint64_t observed_room_ = 0;
    uint64_t incremental_reserved_ = 0;
};

}  // namespace strata::core
