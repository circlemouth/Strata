import copy
import unittest

from bench.evo_x2.contract import (LENGTHS, accepted_request, output_limit, p2_metrics,
                                   stream_metrics, summary, validate_mixed_trigger,
                                   validate_performance_cell, validate_readiness)


def valid_cell(**changes):
    row = dict(normal_completion=True, engine_resident=True, final_sequence_counted=True,
               status_code=200, completion_tokens=512, requested_output_tokens=512,
               engine_batch_slots=2, context_tokens=131072, target_input_tokens=4096,
               prompt_tokens=4096, engine_cache_n=0, cancelled=False, disconnected=False,
               truncated=False, output_limit_adjusted=False, tokenizer_sha256="a" * 64,
               template_sha256="b" * 64, request_sha256="c" * 64, send_s=0.0, complete_s=10.0,
               dispatch_mode="barrier")
    return dict(row, **changes)


def readiness():
    return dict(context_tokens=131072, configured_parallel=2, props_total_slots=2,
                engine_batch_slots=2, fit_max_tokens=False,
                progress=[dict(slot=0, request_id="one", engine_progress_s=[1, 3, 5]),
                          dict(slot=1, request_id="two", engine_progress_s=[2, 4, 6])])


class OutputContract(unittest.TestCase):
    def test_omitted_and_null_default(self):
        for value in ({}, {"max_tokens": None}, {"max_completion_tokens": None}):
            self.assertEqual(output_limit(value), 16384)

    def test_supported_aliases(self):
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            self.assertEqual(output_limit({field: 512}), 512)

    def test_equal_aliases_and_null_are_allowed(self):
        self.assertEqual(output_limit(dict(max_tokens=512, max_output_tokens=None, max_completion_tokens=512)), 512)

    def test_conflicting_aliases_rejected(self):
        with self.assertRaises(ValueError):
            output_limit(dict(max_tokens=512, max_completion_tokens=4096))

    def test_invalid_output_types_and_values(self):
        for value in (0, -1, 16385, True, False, 1.0, "512", [], {}, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                output_limit({"max_tokens": value})

    def test_all_boundary_pairs(self):
        for output, limit in ((512, 130552), (4096, 126968), (16384, 114680)):
            self.assertTrue(accepted_request(limit, {"max_tokens": output}))
            self.assertFalse(accepted_request(limit + 1, {"max_tokens": output}))

    def test_input_count_cannot_be_estimated_or_negative(self):
        for value in ("114680", 114680.0, True, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                accepted_request(value, {})

    def test_default_output_cap_is_used_before_boundary(self):
        self.assertTrue(accepted_request(114680, {}))
        self.assertFalse(accepted_request(114681, {}))


class Readiness(unittest.TestCase):
    def test_ready_requires_observed_overlapping_progress(self):
        self.assertEqual(validate_readiness(readiness()), [])

    def test_declared_slots_are_not_observed_slots(self):
        data = readiness()
        data["engine_batch_slots"] = 1
        self.assertIn("engine_batch_slots", validate_readiness(data))

    def test_props_downgrade_rejected(self):
        data = readiness()
        data["props_total_slots"] = 1
        self.assertIn("props_total_slots", validate_readiness(data))

    def test_sequential_requests_not_p2(self):
        data = readiness()
        data["progress"][1]["engine_progress_s"] = [6, 7]
        self.assertIn("overlapping_engine_progress", validate_readiness(data))

    def test_duplicate_slot_rejected(self):
        data = readiness()
        data["progress"][1]["slot"] = 0
        self.assertIn("distinct_engine_slots", validate_readiness(data))

    def test_unknown_progress_rejected(self):
        data = readiness()
        del data["progress"]
        self.assertTrue(validate_readiness(data))

    def test_fit_output_shortening_rejected(self):
        data = readiness()
        data["fit_max_tokens"] = True
        self.assertIn("fit_max_tokens", validate_readiness(data))


class PerformanceCells(unittest.TestCase):
    def test_valid_exact_lengths(self):
        for length in LENGTHS:
            self.assertEqual(validate_performance_cell(valid_cell(target_input_tokens=length, prompt_tokens=length)), [])

    def test_missing_final_tokenization_evidence(self):
        self.assertIn("final_sequence_counted", validate_performance_cell(valid_cell(final_sequence_counted=False)))

    def test_raw_prompt_length_not_accepted(self):
        self.assertIn("prompt_tokens", validate_performance_cell(valid_cell(prompt_tokens=4090)))

    def test_early_eos_is_failed_cell(self):
        self.assertIn("completion_tokens", validate_performance_cell(valid_cell(completion_tokens=511)))

    def test_cold_cache_reuse_is_failed_cell(self):
        self.assertIn("cold_prefix_reuse", validate_performance_cell(valid_cell(engine_cache_n=128)))

    def test_missing_cache_evidence_fails(self):
        row = valid_cell()
        del row["engine_cache_n"]
        self.assertIn("cold_prefix_reuse", validate_performance_cell(row))

    def test_partial_cancelled_disconnected_fail(self):
        for field in ("cancelled", "disconnected", "truncated", "output_limit_adjusted"):
            self.assertIn(field, validate_performance_cell(valid_cell(**{field: True})))

    def test_nonfinite_timestamps_fail(self):
        self.assertIn("request_time_interval", validate_performance_cell(valid_cell(complete_s=float("nan"))))

    def test_request_hash_required(self):
        self.assertIn("request_sha256", validate_performance_cell(valid_cell(request_sha256="unknown")))

    def test_p2_throughput_uses_both_completion_times(self):
        records = [valid_cell(), valid_cell(send_s=.1, complete_s=12, request_sha256="d" * 64)]
        result = p2_metrics(records)
        self.assertEqual(result["elapsed_s"], 12)
        self.assertEqual(result["output_tokens_per_s"], 1024 / 12)
        self.assertAlmostEqual(result["start_skew_s"], .1)

    def test_p2_rejects_same_content(self):
        with self.assertRaises(ValueError):
            p2_metrics([valid_cell(), valid_cell()])

    def test_p2_rejects_failed_member_not_drop_it(self):
        with self.assertRaises(ValueError):
            p2_metrics([valid_cell(), valid_cell(completion_tokens=42, request_sha256="d" * 64)])

    def test_p2_requires_barrier(self):
        with self.assertRaises(ValueError):
            p2_metrics([valid_cell(), valid_cell(dispatch_mode="sleep", request_sha256="d" * 64)])


class Timings(unittest.TestCase):
    def test_ttft_ignores_role_and_separates_reasoning(self):
        row = stream_metrics(10, [{"time_s": 11}, {"time_s": 12, "reasoning": "考える"},
                                  {"time_s": 14, "content": "回答"}])
        self.assertEqual(row["ttft_any_s"], 2)
        self.assertEqual(row["ttft_api_s"], 4)
        self.assertIsNone(row["engine_token_intervals"])
        self.assertEqual(row["sse_intervals_s"], [1, 2])

    def test_no_answer_ttft_is_null(self):
        self.assertIsNone(stream_metrics(0, [{"time_s": 1}])["ttft_api_s"])

    def test_sse_is_not_engine_token_timing(self):
        row = stream_metrics(0, [{"time_s": 1, "content": "multiple tokens"}], [1, 1.1, 1.3])
        self.assertEqual(row["sse_intervals_s"], [])
        self.assertEqual(row["engine_token_intervals"]["count"], 2)

    def test_nonmonotonic_sse_rejected(self):
        with self.assertRaises(ValueError):
            stream_metrics(0, [{"time_s": 2}, {"time_s": 1}])

    def test_five_samples_not_a_p95_claim(self):
        self.assertNotIn("p95", summary([1, 2, 3, 4, 5]))
        with self.assertRaises(ValueError):
            summary([1, 2, 3, 4, 5], include_p95=True)

    def test_hundred_samples_allow_p95(self):
        self.assertEqual(summary(list(range(100)), include_p95=True)["p95"], 94)

    def test_mixed_requires_observed_event_and_overlap(self):
        row = dict(scenario="short_wait", trigger_kind="prefill_started", trigger_source="engine",
                   dispatch_mode="observed_event", primary_send_s=0, trigger_s=1,
                   secondary_send_s=1.1, primary_complete_s=10)
        self.assertEqual(validate_mixed_trigger(row), [])
        row["dispatch_mode"] = "sleep"
        self.assertIn("event_driven_dispatch", validate_mixed_trigger(row))

    def test_mixed_prefill_not_inferred_from_api_send(self):
        row = dict(scenario="short_wait", trigger_kind="prefill_started", trigger_source="api",
                   dispatch_mode="observed_event", primary_send_s=0, trigger_s=1,
                   secondary_send_s=1.1, primary_complete_s=10)
        self.assertIn("observed_event_trigger", validate_mixed_trigger(row))


if __name__ == "__main__":
    unittest.main()
