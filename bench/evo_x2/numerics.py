"""Full-vocabulary teacher-forced comparisons; no API top-k approximation."""

import math
from collections import defaultdict

from .contract import finite, percentile

LENGTH_GROUPS = ("short", "32k", "64k", "96k", "near_limit")
EXECUTIONS = ("p1", "p2_slot0", "p2_slot1")
CACHE_STATES = ("cold", "reused")
THRESHOLDS = (65536, 98304, 114688)


def compare_logits(reference, candidate):
    if len(reference) < 2 or len(reference) != len(candidate):
        raise ValueError("vocabulary size mismatch")
    if not all(finite(x) for x in reference + candidate):
        raise ValueError("non-finite logit")

    def log_softmax(values):
        peak = max(values)
        shifted = [x - peak for x in values]
        log_total = math.log(math.fsum(math.exp(x) for x in shifted))
        result = [x - log_total for x in shifted]
        if not all(math.isfinite(x) for x in result):
            raise ValueError("logit dynamic range overflow")
        return result

    a, b = log_softmax(reference), log_softmax(candidate)
    kl = math.fsum(math.exp(x) * (x - y) for x, y in zip(a, b))
    if not math.isfinite(kl):
        raise ValueError("non-finite KL")
    # Equal maxima use the first index in both paths, matching a stable argmax.
    return max(0.0, kl), reference.index(max(reference)) == candidate.index(max(candidate))


def _stats(values):
    kls = [x[0] for x in values]
    result = {"positions": len(values), "kl_mean": math.fsum(kls) / len(kls),
              "kl_p99": percentile(kls, .99), "kl_max": max(kls),
              "top1_agreement": sum(x[1] for x in values) / len(values)}
    result["passed"] = (result["kl_mean"] <= .001 and result["kl_p99"] <= .01
                        and result["kl_max"] <= .05 and result["top1_agreement"] >= .995)
    return result


def numerical_gate(prefixes, *, thresholds=THRESHOLDS):
    """Each prefix carries 32+ *full* vocabulary rows for the same forced history.

    The coverage gate requires all 30 length/execution/cache combinations, two
    lifecycle prefixes, and the positions before/at/after each enabled threshold.
    Externally supplied vocabulary size must match the frozen model manifest.
    """
    if not isinstance(prefixes, list) or any(not isinstance(prefix, dict) for prefix in prefixes):
        raise ValueError("prefixes must be a list of evidence objects")
    errors, all_values = [], []
    grouped = defaultdict(list)
    coverage, lifecycle, threshold_positions, seen = set(), set(), set(), set()
    vocabularies = set()
    for prefix in prefixes:
        name = prefix.get("id")
        if not isinstance(name, str) or not name or name in seen:
            errors.append("missing_or_duplicate_prefix_id")
            continue
        seen.add(name)
        try:
            length, execution, cache = (prefix[key] for key in ("length_group", "execution", "cache_state"))
            if length not in LENGTH_GROUPS or execution not in EXECUTIONS or cache not in CACHE_STATES:
                raise ValueError("unknown coverage stratum")
            if prefix.get("teacher_forced") is not True or prefix.get("full_vocabulary") is not True:
                raise ValueError("full-vocabulary teacher-forcing evidence required")
            if type(prefix.get("new_clip_count")) is not int or prefix["new_clip_count"] != 0:
                raise ValueError("new clipping or unknown clip count")
            tokens = prefix["forced_token_ids"]
            reference, candidate = prefix["reference_logits"], prefix["candidate_logits"]
            histories_a, histories_b = prefix["reference_history_sha256"], prefix["candidate_history_sha256"]
            vocabulary = prefix["vocabulary_size"]
            if type(vocabulary) is not int or vocabulary < 2:
                raise ValueError("invalid frozen vocabulary size")
            vocabularies.add(vocabulary)
            if len(tokens) < 32 or not (len(tokens) == len(reference) == len(candidate) == len(histories_a) == len(histories_b)):
                raise ValueError("at least 32 aligned teacher-forced positions required")
            if any(type(token) is not int or not 0 <= token < vocabulary for token in tokens):
                raise ValueError("invalid forced token ID")
            if histories_a != histories_b or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in histories_a):
                raise ValueError("teacher-forced histories differ or are missing")
            values = []
            for a, b in zip(reference, candidate):
                if len(a) != vocabulary or len(b) != vocabulary:
                    raise ValueError("top-k or incomplete vocabulary rows are not allowed")
                values.append(compare_logits(a, b))
            kind = prefix.get("kind", "matrix")
            if kind == "matrix":
                coverage.add((length, execution, cache))
            elif kind == "lifecycle" and prefix.get("lifecycle") in ("cancel", "restore"):
                lifecycle.add(prefix["lifecycle"])
            elif kind == "threshold" and type(prefix.get("sequence_position")) is int:
                threshold_positions.add(prefix["sequence_position"])
            else:
                raise ValueError("unknown prefix kind")
            all_values.extend(values)
            for key in (f"length:{length}", f"execution:{execution}", f"cache:{cache}", f"prefix:{name}"):
                grouped[key].extend(values)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{name}: {exc}")
    required = {(length, execution, cache) for length in LENGTH_GROUPS for execution in EXECUTIONS for cache in CACHE_STATES}
    if required - coverage:
        errors.append(f"missing_matrix_combinations:{len(required - coverage)}")
    if lifecycle != {"cancel", "restore"}:
        errors.append("missing_cancel_restore_prefixes")
    if {p + offset for p in thresholds for offset in (-1, 0, 1)} - threshold_positions:
        errors.append("missing_threshold_boundary_prefixes")
    if len(vocabularies) != 1:
        errors.append("inconsistent_or_missing_vocabulary_size")
    stats = _stats(all_values) if all_values else None
    strata = {name: _stats(values) for name, values in sorted(grouped.items())}
    if stats and not stats["passed"]:
        errors.append("aggregate_numerical_gate")
    for name, value in strata.items():
        if not value["passed"]:
            errors.append(f"stratum_numerical_gate:{name}")
    return {"scope": "teacher_forced_numerics_only", "passed": not errors,
            "prefixes": len(seen), "errors": errors, "aggregate": stats, "strata": strata}
