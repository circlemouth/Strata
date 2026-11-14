"""Independent contract oracle and validators for recorded benchmark evidence."""

import math
import statistics

CONTEXT = 131072
MAX_OUTPUT = 16384
SLACK = 8
LENGTHS = (4096, 32768, 65536, 98304, 130552)
OUTPUT_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def output_limit(request):
    """Return the strict output limit, or raise ValueError (expected HTTP 400)."""
    values = [request[key] for key in OUTPUT_FIELDS if request.get(key) is not None]
    for value in values:
        if type(value) is not int or not 1 <= value <= MAX_OUTPUT:
            raise ValueError("output limit must be an integer from 1 through 16384")
    if len(set(values)) > 1:
        raise ValueError("conflicting output limits")
    return values[0] if values else MAX_OUTPUT


def accepted_request(encoded_input_tokens, request):
    """Count the final encoded sequence, never a string or whitespace estimate."""
    if type(encoded_input_tokens) is not int or encoded_input_tokens < 0:
        raise ValueError("encoded input count must be a nonnegative integer")
    limit = output_limit(request)
    return encoded_input_tokens + limit + SLACK <= CONTEXT


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def percentile(values, probability):
    """Nearest-rank quantile, explicitly fixed for reproducible reporting."""
    if not values or not 0 < probability <= 1 or not all(finite(x) for x in values):
        raise ValueError("quantile requires finite observations and 0 < p <= 1")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summary(values, include_p95=False):
    if not values or not all(finite(x) for x in values):
        raise ValueError("summary requires finite observations")
    result = {"count": len(values), "median": statistics.median(values),
              "min": min(values), "max": max(values)}
    if include_p95:
        if len(values) < 100:
            raise ValueError("latency p95 qualification requires at least 100 requests")
        result["p95"] = percentile(values, .95)
    return result


def validate_readiness(evidence):
    """A configuration alone does not establish P2: require overlapping progress."""
    errors = []
    for key, value in (("context_tokens", CONTEXT), ("configured_parallel", 2),
                       ("props_total_slots", 2), ("engine_batch_slots", 2)):
        if type(evidence.get(key)) is not int or evidence[key] != value:
            errors.append(key)
    if evidence.get("fit_max_tokens") is not False:
        errors.append("fit_max_tokens")
    progress = evidence.get("progress", [])
    if not isinstance(progress, list) or len(progress) != 2:
        return errors + ["two_request_progress"]
    try:
        if {p["slot"] for p in progress} != {0, 1} or any(type(p["slot"]) is not int for p in progress):
            errors.append("distinct_engine_slots")
        if len({p["request_id"] for p in progress}) != 2:
            errors.append("distinct_requests")
        intervals = []
        for p in progress:
            stamps = p["engine_progress_s"]
            if len(stamps) < 2 or not all(finite(s) for s in stamps) or any(a >= b for a, b in zip(stamps, stamps[1:])):
                errors.append("engine_progress_timestamps")
            else:
                intervals.append((stamps[0], stamps[-1]))
        if len(intervals) != 2 or max(x[0] for x in intervals) >= min(x[1] for x in intervals):
            errors.append("overlapping_engine_progress")
    except (KeyError, TypeError):
        errors.append("malformed_progress")
    return errors


def validate_performance_cell(record, *, cold=True):
    """Return all errors; failed cells must not be silently removed from a report."""
    errors = []
    required_true = ("normal_completion", "engine_resident", "final_sequence_counted")
    for key in required_true:
        if record.get(key) is not True:
            errors.append(key)
    for key, expected in (("status_code", 200), ("completion_tokens", 512),
                          ("requested_output_tokens", 512), ("engine_batch_slots", 2),
                          ("context_tokens", CONTEXT)):
        if type(record.get(key)) is not int or record[key] != expected:
            errors.append(key)
    target = record.get("target_input_tokens")
    if type(target) is not int or target not in LENGTHS:
        errors.append("target_input_tokens")
    if type(record.get("prompt_tokens")) is not int or record.get("prompt_tokens") != target:
        errors.append("prompt_tokens")
    if cold and (type(record.get("engine_cache_n")) is not int or record["engine_cache_n"] != 0):
        errors.append("cold_prefix_reuse")
    for key in ("cancelled", "disconnected", "truncated", "output_limit_adjusted"):
        if record.get(key) is not False:
            errors.append(key)
    for key in ("tokenizer_sha256", "template_sha256", "request_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            errors.append(key)
    start, end = record.get("send_s"), record.get("complete_s")
    if not finite(start) or not finite(end) or start >= end:
        errors.append("request_time_interval")
    return errors


def p2_metrics(records):
    """Compute P2 throughput only from two valid, distinct, complete cold requests."""
    if len(records) != 2:
        raise ValueError("P2 requires exactly two requests")
    errors = [validate_performance_cell(r) for r in records]
    if any(errors):
        raise ValueError(f"invalid performance cell: {errors}")
    if records[0]["target_input_tokens"] != records[1]["target_input_tokens"]:
        raise ValueError("P2 input lengths differ")
    if records[0]["request_sha256"] == records[1]["request_sha256"]:
        raise ValueError("P2 requests must contain different synthetic content")
    if any(r.get("dispatch_mode") != "barrier" for r in records):
        raise ValueError("P2 requires barrier dispatch evidence")
    if max(r["send_s"] for r in records) >= min(r["complete_s"] for r in records):
        raise ValueError("requests did not overlap")
    elapsed = max(r["complete_s"] for r in records) - min(r["send_s"] for r in records)
    return {"elapsed_s": elapsed, "start_skew_s": abs(records[0]["send_s"] - records[1]["send_s"]),
            "output_tokens_per_s": 1024 / elapsed}


def stream_metrics(send_s, chunks, engine_token_timestamps=None):
    """Separate SSE transport timings from engine token timings. Missing is null."""
    if not finite(send_s):
        raise ValueError("invalid send timestamp")
    first_api = first_any = None
    times = []
    for chunk in chunks:
        stamp = chunk.get("time_s")
        if not finite(stamp) or stamp < send_s or (times and stamp < times[-1]):
            raise ValueError("non-monotonic SSE timestamp")
        times.append(stamp)
        content, reasoning = chunk.get("content", ""), chunk.get("reasoning", "")
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError("chunk content must be text")
        if content and first_api is None:
            first_api = stamp - send_s
        if (content or reasoning) and first_any is None:
            first_any = stamp - send_s
    engine = None
    if engine_token_timestamps is not None:
        stamps = engine_token_timestamps
        if not all(finite(x) and x >= send_s for x in stamps) or any(a > b for a, b in zip(stamps, stamps[1:])):
            raise ValueError("invalid engine token timestamps")
        intervals = [b - a for a, b in zip(stamps, stamps[1:])]
        if intervals:
            engine = {"count": len(intervals), "p50_s": percentile(intervals, .5),
                      "p95_s": percentile(intervals, .95), "max_s": max(intervals)}
    return {"ttft_api_s": first_api, "ttft_any_s": first_any,
            "sse_intervals_s": [b - a for a, b in zip(times, times[1:])],
            "engine_token_intervals": engine}


def validate_mixed_trigger(record):
    """A fixed delay is not evidence of prefill or the first response token."""
    expected = {"decode_protection": ("first_content", "api"),
                "short_wait": ("prefill_started", "engine")}
    if record.get("scenario") not in expected:
        return ["unknown_mixed_scenario"]
    kind, source = expected[record["scenario"]]
    errors = []
    if record.get("trigger_kind") != kind or record.get("trigger_source") != source:
        errors.append("observed_event_trigger")
    if record.get("dispatch_mode") != "observed_event":
        errors.append("event_driven_dispatch")
    times = [record.get(key) for key in ("primary_send_s", "trigger_s", "secondary_send_s", "primary_complete_s")]
    if not all(finite(x) for x in times) or not times[0] <= times[1] <= times[2] < times[3]:
        errors.append("trigger_order_or_overlap")
    return errors
