"""Deterministic workload instructions, not tokenized requests or measurements."""

import hashlib
import json

from .contract import LENGTHS, MAX_OUTPUT


def build_plan():
    cold = []
    for length in LENGTHS:
        for concurrency in (1, 2):
            cold.append({"id": f"cold-{length}-p{concurrency}", "kind": "cold",
                         "input_tokens": length, "output_tokens": 512,
                         "requests": concurrency, "server_parallel": 2,
                         "dispatch": "barrier" if concurrency == 2 else "single",
                         "exact_final_tokenization_required": True,
                         "distinct_request_content": True, "engine_cache_n": 0})
    warm = []
    for added in (128, 512, 2048):
        for turn in range(12):
            for conversation in ("A", "B"):
                warm.append({"id": f"warm-{added}-{conversation}-{turn + 1}",
                             "kind": "warm", "conversation": conversation,
                             "turn": turn + 1, "added_input_tokens": added,
                             "exact_final_tokenization_required": True})
    return {"schema_version": 1, "kind": "unexecuted_plan", "synthetic_only": True,
            "cold": cold, "warm": warm,
            "mixed": [{"id": "decode-protection", "primary_input_tokens": 1024,
                       "secondary_input_tokens": 98304, "trigger": "first_nonempty_answer_content"},
                      {"id": "short-wait", "primary_input_tokens": 98304,
                       "secondary_input_tokens": 1024, "trigger": "engine_prefill_started"}],
            "boundary": [{"input_tokens": 114680, "output_tokens": MAX_OUTPUT, "expected_status": 200},
                         {"input_tokens": 114681, "output_tokens": MAX_OUTPUT, "expected_status": 400}],
            "lifecycle": ["cancel", "disconnect", "queue", "restart", "slot_reuse"],
            "coexist": {"separate_results": True, "requires_explicit_local_run": True},
            "screening": {"input_lengths": [4096, 32768], "repeats": 3},
            "final": {"pairs_per_length": 5, "short_requests": 100, "p2_minutes": 60},
            "comparison_order": ["AB", "BA", "AB", "BA", "AB"],
            "comparison_rules": {"fresh_process_per_arm": True, "fixed_warmup": True,
                                 "same_seed_sampling_stop": True, "single_loaded_arm": True}}


def synthetic_segments(request_index):
    """Place different signals at the start, middle and end of an assembled prompt.

    These are content seeds only. An authorized local collector must pad and verify
    the final template/tokenizer sequence; character length is never a token count.
    """
    if type(request_index) is not int or request_index < 0:
        raise ValueError("request index must be nonnegative")
    return {"start": f"架空の検査記録。開始コードはSTART-{request_index:04d}。",
            "middle": f"中央コードはMIDDLE-{request_index:04d}。対象の側は{'左' if request_index % 2 else '右'}。",
            "end": f"末尾コードはEND-{request_index:04d}。3つのコードと側をそのまま答えてください。"}


def frozen_digest(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
