"""Fail-closed joins between captured HTTP streams and explicit engine sidecars.

This module never reads a log directory, guesses request IDs from time order,
uses a slot number as an identity, or invents engine token/cache information.
Sidecar provenance and clock bindings must be established by a local collector.
"""

from .contract import (CONTEXT, finite, p2_metrics, stream_metrics,
                       validate_performance_cell, validate_readiness)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def correlate_record(capture, evidence, manifest_sha256):
    errors = []
    if not isinstance(capture, dict) or not isinstance(evidence, dict):
        return {"correlation_complete": False, "record": None, "errors": ["record_objects_required"]}
    binding, tokenization, engine = (evidence.get(key) for key in ("binding", "tokenization", "engine"))
    if not all(isinstance(value, dict) for value in (binding, tokenization, engine)):
        return {"correlation_complete": False, "record": None, "errors": ["binding_tokenization_engine_required"]}
    if capture.get("api_stream_complete") is not True or capture.get("errors") != []:
        errors.append("api_stream_not_complete")
    if binding.get("method") != "instrumented_request_binding":
        errors.append("explicit_instrumented_binding_required")
    for key, capture_key in (("client_request_id", "client_request_id"), ("request_sha256", "request_sha256"),
                             ("api_response_id", "response_id")):
        if not isinstance(capture.get(capture_key), str) or not capture[capture_key] or binding.get(key) != capture[capture_key]:
            errors.append(f"binding_mismatch:{key}")
    engine_id = binding.get("engine_request_id")
    if not isinstance(engine_id, str) or not engine_id or engine.get("request_id") != engine_id:
        errors.append("engine_request_identity_missing_or_mismatched")
    if not _sha(capture.get("request_sha256")) or tokenization.get("request_sha256") != capture.get("request_sha256"):
        errors.append("tokenization_request_hash_mismatch")
    if tokenization.get("manifest_sha256") != manifest_sha256 or not _sha(manifest_sha256):
        errors.append("tokenization_manifest_mismatch")
    expected_source = "synthetic_tokenizer" if capture.get("evidence_kind") == "synthetic_test" else "runtime_tokenizer"
    if tokenization.get("source") != expected_source or tokenization.get("includes_template_and_expansion") is not True:
        errors.append("final_runtime_tokenization_required")
    for key in ("tokenizer_sha256", "template_sha256"):
        if not _sha(tokenization.get(key)):
            errors.append(f"missing_hash:{key}")
    usage = capture.get("usage")
    if not isinstance(usage, dict):
        return {"correlation_complete": False, "record": None, "errors": errors + ["exact_api_usage_required"]}
    encoded = tokenization.get("encoded_input_tokens")
    if type(encoded) is not int or encoded != capture.get("target_input_tokens") or encoded != usage.get("prompt_tokens"):
        errors.append("final_encoded_input_count_mismatch")
    for key, expected in (("batch_slots", 2), ("context_tokens", CONTEXT), ("cache_n", 0), ("completion_tokens", 512)):
        if type(engine.get(key)) is not int or engine[key] != expected:
            errors.append(f"engine_contract:{key}")
    if type(engine.get("slot")) is not int or engine["slot"] not in (0, 1):
        errors.append("engine_slot_unknown")
    if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != engine.get("completion_tokens"):
        errors.append("api_engine_completion_count_mismatch")
    if type(usage.get("cached_tokens")) is not int or usage["cached_tokens"] != engine.get("cache_n"):
        errors.append("api_engine_cache_count_mismatch")
    if engine.get("terminal_status") != "completed":
        errors.append("engine_not_normally_completed")
    if engine.get("resident") is not True:
        errors.append("engine_residency_unverified")
    for key in ("cancelled", "disconnected", "truncated", "output_limit_adjusted"):
        if engine.get(key) is not False:
            errors.append(f"engine_flag:{key}")
    api_timings = capture.get("api_timings")
    if isinstance(api_timings, dict) and "cache_n" in api_timings:
        if type(api_timings["cache_n"]) is not int or api_timings["cache_n"] != engine.get("cache_n"):
            errors.append("api_timing_engine_cache_mismatch")
    stamps = engine.get("token_timestamps_s")
    send, end = capture.get("send_s"), capture.get("complete_s")
    if (not isinstance(stamps, list) or len(stamps) != 512 or not finite(send) or not finite(end)
            or not all(finite(stamp) and send <= stamp <= end for stamp in stamps)
            or any(a > b for a, b in zip(stamps, stamps[1:]))):
        errors.append("exact_engine_token_timestamps_required")
    prefill = engine.get("prefill_started_s")
    if prefill is not None and (not finite(prefill) or not finite(send) or not isinstance(stamps, list)
                                or not stamps or not finite(stamps[0]) or not send <= prefill <= stamps[0]):
        errors.append("invalid_observed_prefill_timestamp")
    if errors:
        return {"correlation_complete": False, "record": None, "errors": errors}
    record = {"client_request_id": capture["client_request_id"], "engine_request_id": engine_id,
              "status_code": capture["status_code"], "request_sha256": capture["request_sha256"],
              "target_input_tokens": capture["target_input_tokens"], "prompt_tokens": usage["prompt_tokens"],
              "completion_tokens": usage["completion_tokens"], "requested_output_tokens": capture["requested_output_tokens"],
              "context_tokens": engine["context_tokens"], "engine_batch_slots": engine["batch_slots"],
              "engine_slot": engine["slot"], "engine_cache_n": engine["cache_n"],
              "tokenizer_sha256": tokenization["tokenizer_sha256"], "template_sha256": tokenization["template_sha256"],
              "normal_completion": True, "engine_resident": True, "final_sequence_counted": True,
              "cancelled": False, "disconnected": False, "truncated": False, "output_limit_adjusted": False,
              "send_s": send, "complete_s": end, "dispatch_mode": capture.get("dispatch_mode"),
              "ttft_api_s": capture.get("ttft_api_s"), "ttft_any_s": capture.get("ttft_any_s"),
              "engine_token_intervals": stream_metrics(send, [], stamps)["engine_token_intervals"],
              "engine_progress_s": [stamp for i, stamp in enumerate(stamps) if i == 0 or stamp != stamps[i - 1]],
              "engine_prefill_started_s": prefill, "evidence_kind": capture["evidence_kind"]}
    errors = validate_performance_cell(record)
    return {"correlation_complete": not errors, "record": record if not errors else None, "errors": errors}


def correlate_pair(capture, evidence):
    """Only an explicit same-run/same-clock binding may unlock P2 arithmetic."""
    result = {"schema_version": 1, "correlation_complete": False, "production_qualified": False,
              "errors": [], "records": [], "p2_metrics": None,
              "note": "Sidecar provenance is supplied, not authenticated. Full qualification remains separate."}
    errors = result["errors"]
    if not isinstance(capture, dict) or not isinstance(evidence, dict):
        errors.append("capture_and_engine_evidence_objects_required")
        return result
    for document in (capture, evidence):
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            errors.append("unsupported_schema_version")
    for key in ("run_id", "clock_id", "evidence_kind"):
        if not isinstance(capture.get(key), str) or not capture[key] or capture[key] != evidence.get(key):
            errors.append(f"sidecar_mismatch:{key}")
    if capture.get("evidence_kind") not in ("synthetic_test", "captured"):
        errors.append("unknown_evidence_kind")
    result["evidence_kind"] = capture.get("evidence_kind")
    if evidence.get("clock_binding") != "shared_monotonic_domain":
        errors.append("explicit_shared_monotonic_clock_binding_required")
    if capture.get("dispatch_mode") != "barrier":
        errors.append("paired_barrier_dispatch_required")
    manifest = evidence.get("manifest_sha256")
    if not _sha(manifest):
        errors.append("frozen_manifest_hash_required")
    captures, sidecars = capture.get("records"), evidence.get("records")
    if (not isinstance(captures, list) or not isinstance(sidecars, list) or len(captures) != 2 or len(sidecars) != 2
            or not all(isinstance(record, dict) for record in captures + sidecars)):
        errors.append("exactly_two_capture_and_engine_records_required")
        return result
    names = [record.get("client_request_id") for record in captures]
    bindings = [record.get("binding") for record in sidecars]
    if (not all(isinstance(name, str) and name for name in names) or len(set(names)) != 2
            or not all(isinstance(binding, dict) for binding in bindings)):
        errors.append("unique_explicit_request_bindings_required")
        return result
    side_names = [binding.get("client_request_id") for binding in bindings]
    if not all(isinstance(name, str) for name in side_names) or set(side_names) != set(names):
        errors.append("sidecar_request_set_mismatch")
        return result
    if any(record.get("evidence_kind") != capture.get("evidence_kind") for record in captures):
        errors.append("mixed_record_evidence_kinds")
    if errors:
        return result
    indexed = dict(zip(side_names, sidecars))
    for record in captures:
        outcome = correlate_record(record, indexed[record["client_request_id"]], manifest)
        errors.extend(f"{record['client_request_id']}:{error}" for error in outcome["errors"])
        if outcome["record"] is not None:
            result["records"].append(outcome["record"])
    if errors:
        return result
    records = result["records"]
    readiness = {"context_tokens": CONTEXT, "configured_parallel": evidence.get("configured_parallel"),
                 "props_total_slots": evidence.get("props_total_slots"), "engine_batch_slots": 2,
                 "fit_max_tokens": evidence.get("fit_max_tokens"),
                 "progress": [{"slot": row["engine_slot"], "request_id": row["engine_request_id"],
                               "engine_progress_s": row["engine_progress_s"]} for row in records]}
    errors.extend(validate_readiness(readiness))
    if not errors:
        try:
            result["p2_metrics"] = p2_metrics(records)
        except ValueError as exc:
            errors.append(str(exc))
    result["correlation_complete"] = not errors
    return result
