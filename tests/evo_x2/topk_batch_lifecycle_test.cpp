// CPU-only reference model for the proposed batch wiring. It is NOT connected to
// Verifier and does not prove HIP graph, P2 server, cancellation or restore safety.
#include "strata/core/batch_rows.hpp"
#include "strata/kernels/qsa_evo_topk.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace e = strata::kernels::evo_topk;
namespace model {
using BaseKey = std::vector<int>;
enum class Variant { Register, Split };
using ForwardKey = std::pair<BaseKey, Variant>;
// Mirrors the current Verifier::batch_key + bkey representation, deliberately
// excluding top-k mode. Never add mode to the key used by commit/save maps.
BaseKey base_key(const std::vector<int>& rows, int handoff_base, bool doorbell) {
    BaseKey r{handoff_base};
    r.insert(r.end(), rows.begin(), rows.end());
    r.push_back(doorbell ? 1 : 0);
    return r;
}
struct Snapshot { int slot; int64_t position; };  // no scratch address or graph handle
struct Fence { const void* owner; uint64_t generation; uint64_t sequence; };
struct Ticket { BaseKey base; Variant variant; Fence fence; bool cancelled = false; };
struct Binding { const void* owner; const void* scratch; uint64_t generation; };

// A conservative lifecycle oracle: one forward in flight per context; cancellation
// is a requested disposition, never a completed GPU fence. Uncertain/errors quarantine.
class Context {
public:
    Context() : bytes_(e::layout(16, 32770).bytes + e::kAlignment - 1) {
        auto p = reinterpret_cast<std::uintptr_t>(bytes_.data());
        scratch_ = reinterpret_cast<void*>((p + e::kAlignment - 1) & ~(e::kAlignment - 1));
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
    const void* scratch() const { return scratch_; }
    uint64_t generation() const { return generation_; }
    bool warm(const BaseKey& base, Variant variant, bool allocation_ok = true) {
        if (phase_ != Phase::Warmup || !allocation_ok) return false;
        forwards_.emplace(ForwardKey{base, variant}, Binding{this, scratch_, generation_});
        commits_.insert(base);  // variants share the SAME existing commit identity
        return true;
    }
    void end_warmup() { assert(phase_ == Phase::Warmup); phase_ = Phase::Idle; }
    bool has(const BaseKey& base, Variant variant) const {
        auto it = forwards_.find({base, variant});
        return it != forwards_.end() && it->second.owner == this && it->second.scratch == scratch_ &&
               it->second.generation == generation_;
    }
    bool ready_for_reuse() const { return phase_ == Phase::Idle; }
    // No graph creation and no mutable global mode here. All positions must qualify:
    // mixed short/long slots use register; a group straddling the threshold is short.
    std::optional<Ticket> launch(const e::Config& cfg, const std::vector<int>& rows,
                                 const std::vector<int64_t>& pos, int hbase, bool doorbell) {
        if (!cfg.valid || !ready_for_reuse() || rows.size() != pos.size() || rows.empty() || rows.size() > 16)
            return std::nullopt;
        if (strata::core::batch_rows_error(rows.data(), int(rows.size()), pos.data(), 2, 16,
                                            hbase, false, 0)) return std::nullopt;
        if (std::any_of(pos.begin(), pos.end(), [](int64_t p) { return p < 0 || p >= 131072; })) return std::nullopt;
        const BaseKey base = base_key(rows, hbase, doorbell);
        if (!has(base, Variant::Register) || !commits_.count(base)) return std::nullopt;
        // Even after the forward fence, rows with an outstanding verdict are not
        // reusable. A different layout must not hide overlap of the same slot.
        for (const auto& pending : pending_)
            for (std::size_t i = 1; i + 1 < base.size(); ++i)
                for (std::size_t j = 1; j + 1 < pending.first.size(); ++j)
                    if (base[i] == pending.first[j]) return std::nullopt;
        const bool all_long = std::all_of(pos.begin(), pos.end(), [&](int64_t p) { return e::wants_split(cfg, p); });
        const Variant selected = all_long && has(base, Variant::Split) ? Variant::Split : Variant::Register;
        active_ = Ticket{base, selected, {this, generation_, ++sequence_}, false};
        phase_ = Phase::InFlight;
        return active_;
    }
    bool cancel() {
        if (!active_ || phase_ != Phase::InFlight) return false;
        active_->cancelled = true;
        return true;
    }
    bool complete(const Fence& fence, bool success = true) {
        if (!active_ || phase_ != Phase::InFlight || fence.owner != this || fence.generation != generation_ ||
            fence.sequence != active_->fence.sequence) return false;
        if (!success) { phase_ = Phase::Quarantined; return false; }
        if (!active_->cancelled) pending_.emplace(active_->base, active_->fence);  // verdict/commit retains its immutable launch key
        active_.reset(); phase_ = Phase::Idle;
        return true;
    }
    bool consume_verdict(const Ticket& ticket) {
        if (ticket.fence.owner != this || ticket.fence.generation != generation_ || !ready_for_reuse()) return false;
        const auto found = pending_.find(ticket.base);
        if (found == pending_.end() || found->second.sequence != ticket.fence.sequence || !commits_.count(ticket.base))
            return false;
        pending_.erase(found);
        return true;
    }
    bool evict(const BaseKey& base) {
        if (!ready_for_reuse() || pending_.count(base)) return false;
        forwards_.erase({base, Variant::Register}); forwards_.erase({base, Variant::Split});
        commits_.erase(base);
        return true;
    }
    bool restore(const Snapshot& snapshot) {
        if (!ready_for_reuse() || !pending_.empty() || snapshot.slot < 0 || snapshot.slot >= 2 ||
            snapshot.position < 0 || snapshot.position >= 131072) return false;
        restored_position_ = snapshot.position;
        return true;  // scratch stays owned by destination; graph addresses are not restored
    }
    bool rebind_storage() {
        if (!ready_for_reuse() || !pending_.empty()) return false;
        // Real implementation must destroy native graph handles before changing backing.
        forwards_.clear(); commits_.clear(); ++generation_; phase_ = Phase::Warmup;
        return true;
    }
    size_t forward_count() const { return forwards_.size(); }
    size_t commit_count() const { return commits_.size(); }
private:
    enum class Phase { Warmup, Idle, InFlight, Quarantined };
    Phase phase_ = Phase::Warmup;
    std::vector<unsigned char> bytes_;
    void* scratch_ = nullptr;
    uint64_t generation_ = 1, sequence_ = 0;
    int64_t restored_position_ = 0;
    std::map<ForwardKey, Binding> forwards_;
    std::set<BaseKey> commits_;
    std::map<BaseKey, Fence> pending_;
    std::optional<Ticket> active_;
};
}  // namespace model

int main() {
    using namespace model;
    const std::vector<int> rows{0, 1}, grouped{0, 0, 1, 1};
    const auto resident = base_key(rows, 0, false), doorbell = base_key(rows, 0, true);
    assert(resident != doorbell && resident != base_key({1, 0}, 0, false));
    assert(resident != base_key(rows, 1, false) && resident != base_key({0, 0}, 0, false));
    // Demonstrate the exact key collision a naive native toggle would introduce.
    std::map<BaseKey, Variant> naive;
    naive[resident] = Variant::Register; naive[resident] = Variant::Split;
    assert(naive.size() == 1 && naive[resident] != Variant::Register);
    std::map<ForwardKey, int> split_namespace;
    split_namespace[{resident, Variant::Register}] = 1; split_namespace[{resident, Variant::Split}] = 2;
    assert(split_namespace.size() == 2);

    Context a, b;
    assert(a.scratch() != b.scratch());
    for (auto* ctx : {&a, &b}) {
        assert(ctx->warm(resident, Variant::Register));
        assert(ctx->warm(resident, Variant::Split));
        assert(ctx->warm(doorbell, Variant::Register));
        // Deliberate failed split warmup must NOT publish a ready split graph.
        assert(!ctx->warm(doorbell, Variant::Split, false));
        assert(ctx->warm(base_key(grouped, 0, false), Variant::Register));
        assert(ctx->warm(base_key(grouped, 0, false), Variant::Split));
        assert(ctx->forward_count() == 5 && ctx->commit_count() == 3);
        ctx->end_warmup();
    }
    for (auto threshold : {"65536", "98304", "114688"}) {
        const auto cfg = e::parse("split", threshold); const int64_t t = cfg.min_tokens;
        for (const auto& positions : std::vector<std::vector<int64_t>>{{t-1,t+10}, {t,t}, {t+1,t+2}}) {
            auto x = a.launch(cfg, rows, positions, 0, false); assert(x);
            const bool want = positions[0] >= t && positions[1] >= t;
            assert(x->variant == (want ? Variant::Split : Variant::Register));
            assert(!a.warm(resident, Variant::Split));  // no request-time capture
            assert(!a.launch(cfg, rows, positions, 0, false));  // no simultaneous use of one scratch
            assert(a.complete(x->fence) && a.consume_verdict(*x));
        }
        auto mixed = a.launch(cfg, grouped, {t-1,t,t+2,t+3}, 0, false); assert(mixed);
        assert(mixed->variant == Variant::Register); assert(a.complete(mixed->fence) && a.consume_verdict(*mixed));
        auto missing = a.launch(cfg, rows, {t,t}, 0, true); assert(missing);
        assert(missing->variant == Variant::Register); assert(a.complete(missing->fence) && a.consume_verdict(*missing));
    }
    const auto cfg = e::parse("split", "65536");
    auto x = a.launch(cfg, rows, {100000, 110000}, 0, false);
    auto y = b.launch(cfg, rows, {70000, 80000}, 0, false); assert(x && y);
    assert(a.cancel() && !a.ready_for_reuse());
    assert(!a.complete(y->fence));  // another context's event cannot release this scratch
    assert(!a.evict(resident) && !a.restore({0, 30}) && !a.rebind_storage());
    assert(b.complete(y->fence) && b.consume_verdict(*y));
    assert(a.complete(x->fence) && a.ready_for_reuse());
    assert(!a.consume_verdict(*x));  // cancelled output has no accepted verdict
    const void* before_restore = a.scratch();
    assert(a.restore({0, 12}) && a.scratch() == before_restore && a.scratch() != b.scratch());
    auto z = a.launch(cfg, rows, {100000, 110000}, 0, false); assert(z);
    assert(!a.complete(x->fence));  // a completed older window must not finish the new one
    assert(a.complete(z->fence));
    assert(!a.consume_verdict(*x));  // stale cancelled ticket cannot consume a newer verdict with the same base key
    assert(!a.launch(cfg, rows, {100001,110001}, 0, false));
    assert(!a.launch(cfg, grouped, {100001,100002,110001,110002}, 0, false));
    assert(!a.evict(resident) && !a.rebind_storage());  // outstanding verdict pins its key/bundle
    auto changed_cfg = e::parse("register", "98304");
    (void) changed_cfg;  // commit consumes z's immutable key, not current mode/position/residency
    assert(a.consume_verdict(*z) && a.evict(resident));
    assert(!a.launch(cfg, rows, {100000,110000}, 0, false));  // request cannot recapture evicted pair
    const uint64_t old_generation = a.generation();
    assert(a.rebind_storage() && a.generation() != old_generation);
    assert(!a.has(doorbell, Variant::Register));
    assert(a.warm(resident, Variant::Register) && a.warm(resident, Variant::Split)); a.end_warmup();
    auto n = a.launch(cfg, rows, {100000,110000}, 0, false); assert(n);
    assert(!a.complete(z->fence) && !a.complete(n->fence, false));
    assert(!a.ready_for_reuse() && !a.rebind_storage() && !a.restore({0, 1}));

    assert(!b.launch(cfg, {0,1,0}, {1,2,3}, 0, false)); // actual batch_rows_error checks reused
    assert(!b.launch(cfg, {0,0}, {65536,65538}, 0, false));
    assert(!b.launch(cfg, rows, {-1, 70000}, 0, false));
    assert(!b.launch(e::parse("typo", nullptr), rows, {70000,70000}, 0, false));
    std::cout << "PASS CPU batch lifecycle/key/threshold reference; NATIVE P2 AND HIP NOT RUN\n";
}
