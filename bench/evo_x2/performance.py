"""Offline acceptance math. Inputs are observations, never generated timings."""

import math
import statistics

from .contract import LENGTHS, finite, percentile


def _positive_samples(values, minimum):
    return isinstance(values, list) and len(values) >= minimum and all(finite(x) and x > 0 for x in values)


def _ratio(reference, candidate, minimum=5, quantile=None):
    if not _positive_samples(reference, minimum) or not _positive_samples(candidate, minimum) or len(reference) != len(candidate):
        raise ValueError("missing, unequal, non-finite or insufficient paired observations")
    reducer = (lambda values: percentile(values, quantile)) if quantile else statistics.median
    return reducer(candidate) / reducer(reference)


def speed_gate(data):
    """Equal-weight geometric mean of five cold P2 *completion-time* ratios.

    This arithmetic gate does not certify provenance, readiness, quality, memory,
    thermal conditions, coexistence, lifecycle, or the 60-minute soak. Those
    remain separate evidence requirements. Conservatively, 10% improvement
    means candidate/reference elapsed time <=0.90 (speedup >=1/0.90).
    """
    errors, ratios = [], {}
    try:
        cold = data["cold_p2"]
        if set(cold) != {str(x) for x in LENGTHS}:
            raise ValueError("all five input lengths must be present, once")
        elapsed_ratios = []
        for length in LENGTHS:
            cell = cold[str(length)]
            reference, candidate = cell["reference_pair_s"], cell["candidate_pair_s"]
            _ratio(reference, candidate)
            # Preserve the five paired ratios. Summarize each length by its
            # median ratio; then give each length equal weight.
            elapsed_ratio = statistics.median(b / a for a, b in zip(reference, candidate))
            elapsed_ratios.append(elapsed_ratio)
            ratios[f"cold_p2:{length}"] = elapsed_ratio
            per_request = cell["request_latency"]
            if len(per_request) != 2:
                raise ValueError("both request-latency series are required")
            for slot, row in enumerate(per_request):
                ratio = _ratio(row["reference_s"], row["candidate_s"])
                ratios[f"request:{length}:{slot}"] = ratio
                if ratio > 1.05:
                    errors.append(f"request_latency_regression:{length}:{slot}")
        geometric_elapsed_ratio = math.exp(math.fsum(math.log(x) for x in elapsed_ratios) / len(elapsed_ratios))
        geometric_speedup = 1 / geometric_elapsed_ratio
        if geometric_elapsed_ratio > .90:
            errors.append("cold_p2_elapsed_reduction_below_10_percent")
        short = data["short_latency"]
        ratio = _ratio(short["reference_s"], short["candidate_s"], minimum=100, quantile=.95)
        ratios["short_p95"] = ratio
        if ratio > 1.10:
            errors.append("short_latency_p95_regression")
        mixed = data["mixed"]
        for key in ("decode_interval_p95_s", "short_ttft_s"):
            row = mixed[key]
            ratio = _ratio(row["reference"], row["candidate"])
            ratios[f"mixed:{key}"] = ratio
            if ratio > 1.10:
                errors.append(f"mixed_regression:{key}")
        warm = data["warm_ttft"]
        if set(warm) != {"128", "512", "2048"}:
            raise ValueError("all three warm increment groups required")
        for increment in ("128", "512", "2048"):
            row = warm[increment]
            ratio = _ratio(row["reference_s"], row["candidate_s"])
            ratios[f"warm:{increment}"] = ratio
            if ratio > 1.10:
                errors.append(f"warm_regression:{increment}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        errors.append(f"invalid_or_missing_evidence:{exc}")
        geometric_speedup = None
        geometric_elapsed_ratio = None
    return {"scope": "performance_arithmetic_only", "passed": not errors,
            "errors": errors, "cold_p2_geometric_speedup": geometric_speedup,
            "cold_p2_geometric_elapsed_ratio": geometric_elapsed_ratio,
            "candidate_reference_latency_ratios": ratios}


def memory_gate(data):
    """Keep owned physical bytes distinct from RSS and reclaimable file cache."""
    errors = []
    try:
        if data.get("accounting") != "deduplicated_owned_nonreclaimable_locked_gpu_backing":
            raise ValueError("owned physical backing accounting is required")
        before, after = data["reference_owned_bytes"], data["candidate_owned_bytes"]
        if type(before) is not int or type(after) is not int or min(before, after) < 0:
            raise ValueError("invalid owned bytes")
        saved = before - after
        if saved < 4 * 1024 ** 3:
            errors.append("less_than_4_gib_owned_bytes_saved")
        for side in ("reference", "candidate"):
            for key in ("reclaimable_file_cache_bytes", "cgroup_charge_bytes", "startup_peak_bytes", "warmup_peak_bytes", "coexist_headroom_bytes"):
                value = data.get(f"{side}_{key}")
                if type(value) is not int or value < 0:
                    errors.append(f"missing_or_invalid:{side}_{key}")
        headroom_before, headroom_after = data.get("reference_coexist_headroom_bytes"), data.get("candidate_coexist_headroom_bytes")
        if type(headroom_before) is not int or type(headroom_after) is not int or headroom_after <= headroom_before:
            errors.append("coexist_headroom_not_improved")
        rows = data["major_latencies"]
        expected = {"cold_p2_completion", "short_p95", "mixed_decode_p95", "mixed_short_ttft", "warm_128", "warm_512", "warm_2048"}
        if set(rows) != expected:
            errors.append("missing_major_latency_groups")
        for key, row in rows.items():
            if _ratio(row["reference_s"], row["candidate_s"], minimum=100 if key == "short_p95" else 5,
                      quantile=.95 if key == "short_p95" else None) > 1.03:
                errors.append(f"latency_regression:{key}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"invalid_or_missing_evidence:{exc}")
        saved = None
    return {"scope": "memory_arithmetic_only", "passed": not errors, "errors": errors, "owned_bytes_saved": saved}
