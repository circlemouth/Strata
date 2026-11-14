import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bench.evo_x2.__main__ import evaluate
from bench.evo_x2.contract import LENGTHS
from bench.evo_x2.fixtures import build_fixtures, mismatch_paths, score_outputs, strict_json, structural_regressions
from bench.evo_x2.numerics import (CACHE_STATES, EXECUTIONS, LENGTH_GROUPS, THRESHOLDS,
                                  compare_logits, numerical_gate)
from bench.evo_x2.performance import memory_gate, speed_gate
from bench.evo_x2.workloads import build_plan, frozen_digest, synthetic_segments


def numeric_prefix(name, **changes):
    row = dict(id=name, length_group="short", execution="p1", cache_state="cold",
               kind="matrix", teacher_forced=True, full_vocabulary=True, new_clip_count=0,
               forced_token_ids=[0] * 32, vocabulary_size=3,
               reference_logits=[[2.0, 1.0, 0.0] for _ in range(32)],
               candidate_logits=[[2.0, 1.0, 0.0] for _ in range(32)],
               reference_history_sha256=[f"{i:064x}" for i in range(32)],
               candidate_history_sha256=[f"{i:064x}" for i in range(32)])
    return dict(row, **changes)


def numeric_matrix():
    rows = [numeric_prefix(f"{length}-{execution}-{cache}", length_group=length, execution=execution, cache_state=cache)
            for length in LENGTH_GROUPS for execution in EXECUTIONS for cache in CACHE_STATES]
    rows.extend(numeric_prefix(f"after-{kind}", kind="lifecycle", lifecycle=kind) for kind in ("cancel", "restore"))
    rows.extend(numeric_prefix(f"threshold-{position}", kind="threshold", sequence_position=position)
                for value in THRESHOLDS for position in (value - 1, value, value + 1))
    return rows


def speed_evidence():
    def row(suffix="_s", count=5, candidate=8):
        return {f"reference{suffix}": [10] * count, f"candidate{suffix}": [candidate] * count}
    return {"cold_p2": {str(length): dict(reference_pair_s=[10] * 5, candidate_pair_s=[8] * 5,
                                          request_latency=[row(), row()]) for length in LENGTHS},
            "short_latency": row(count=100),
            "mixed": {"decode_interval_p95_s": row(suffix=""), "short_ttft_s": row(suffix="")},
            "warm_ttft": {str(x): row() for x in (128, 512, 2048)}}


def memory_evidence():
    data = dict(accounting="deduplicated_owned_nonreclaimable_locked_gpu_backing",
                reference_owned_bytes=20 * 1024 ** 3, candidate_owned_bytes=16 * 1024 ** 3,
                major_latencies={key: {"reference_s": [1] * (100 if key == "short_p95" else 5),
                                       "candidate_s": [1] * (100 if key == "short_p95" else 5)}
                                 for key in ("cold_p2_completion", "short_p95", "mixed_decode_p95", "mixed_short_ttft",
                                             "warm_128", "warm_512", "warm_2048")})
    for side in ("reference", "candidate"):
        for key in ("reclaimable_file_cache_bytes", "cgroup_charge_bytes", "startup_peak_bytes", "warmup_peak_bytes", "coexist_headroom_bytes"):
            data[f"{side}_{key}"] = 100
    data["candidate_coexist_headroom_bytes"] = 101
    return data


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.fixtures = build_fixtures()
        self.answers = {case["id"]: json.dumps(case["expected"], ensure_ascii=False) for case in self.fixtures}

    def test_sixty_deterministic_unique_cases(self):
        self.assertEqual(len(self.fixtures), 60)
        self.assertEqual(len({case["id"] for case in self.fixtures}), 60)
        for category in ("extraction", "organization", "schema"):
            self.assertEqual(sum(case["category"] == category for case in self.fixtures), 20)
        self.assertEqual(frozen_digest(self.fixtures), frozen_digest(build_fixtures()))

    def test_frozen_fixture_snapshot_matches_fixed_digest(self):
        expected_hash = "37217bf28ec50a372daa01b704ab3402fe55fdd411b862037667ef201b60a8dc"
        path = Path(__file__).parent / "fixtures" / "ja_synthetic_v1.json"
        snapshot = strict_json(path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["cases"], self.fixtures)
        self.assertEqual(snapshot["fixtures_sha256"], expected_hash)
        self.assertEqual(frozen_digest(self.fixtures), expected_hash)

    def test_all_golden_answers_pass(self):
        self.assertEqual(score_outputs(self.fixtures, self.answers)["passed"], 60)

    def test_wrong_left_right_is_critical_regression(self):
        candidate = dict(self.answers)
        data = strict_json(candidate["extract-01"])
        data["side"] = "右"
        candidate["extract-01"] = json.dumps(data)
        result = structural_regressions(self.fixtures, self.answers, candidate)
        self.assertFalse(result["structural_gate_passed"])
        self.assertEqual(result["new_critical_errors"][0]["new_error_paths"], ["$/side"])

    def test_baseline_error_remains_visible(self):
        baseline = dict(self.answers)
        data = strict_json(baseline["extract-01"])
        data["count"] = 99
        baseline["extract-01"] = json.dumps(data)
        result = structural_regressions(self.fixtures, baseline, baseline)
        self.assertTrue(result["structural_gate_passed"])
        self.assertEqual(result["baseline"]["passed"], 59)

    def test_baseline_wrong_field_cannot_hide_new_wrong_field(self):
        baseline = dict(self.answers)
        data = strict_json(baseline["extract-01"])
        data["count"] = 99
        baseline["extract-01"] = json.dumps(data)
        candidate = dict(baseline)
        data["side"] = "右"
        candidate["extract-01"] = json.dumps(data)
        self.assertFalse(structural_regressions(self.fixtures, baseline, candidate)["structural_gate_passed"])

    def test_null_not_unspecified_empty_string(self):
        self.assertEqual(mismatch_paths({"A": None}, {"A": ""}), ["$/A"])

    def test_boolean_is_not_number(self):
        self.assertEqual(mismatch_paths({"x": True}, {"x": 1}), ["$/x"])

    def test_unrequested_diagnosis_is_failure(self):
        expected = {"S": "架空", "A": None}
        self.assertEqual(mismatch_paths(expected, dict(expected, diagnosis="invented")), ["$/diagnosis"])

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"side":"左","side":"右"}')

    def test_nonfinite_json_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e400"):
            with self.assertRaises(ValueError):
                strict_json('{"value":' + value + '}')

    def test_missing_candidate_is_not_success(self):
        self.assertEqual(score_outputs(self.fixtures, {})["passed"], 0)

    def test_missing_baseline_blocks_gate(self):
        result = structural_regressions(self.fixtures, {}, self.answers)
        self.assertFalse(result["structural_gate_passed"])
        self.assertEqual(len(result["unverified_baseline_cases"]), 60)

    def test_unknown_cases_not_ignored(self):
        with self.assertRaises(ValueError):
            score_outputs(self.fixtures, {"unknown": "{}"})


class Numerics(unittest.TestCase):
    def test_identical_full_distribution(self):
        self.assertEqual(compare_logits([1, 2, 3], [1, 2, 3]), (0, True))

    def test_shift_invariance(self):
        kl, top = compare_logits([1, 2, 3], [1001, 1002, 1003])
        self.assertAlmostEqual(kl, 0)
        self.assertTrue(top)

    def test_top1_is_not_sufficient(self):
        kl, top = compare_logits([0, 1, 2], [0, 1, 20])
        self.assertTrue(top)
        self.assertGreater(kl, .05)

    def test_nonfinite_logits_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                compare_logits([0, 1], [0, value])

    def test_complete_matrix_passes_and_keeps_strata(self):
        report = numerical_gate(numeric_matrix())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["prefixes"], 41)
        self.assertEqual(report["aggregate"]["positions"], 41 * 32)
        self.assertIn("execution:p2_slot1", report["strata"])

    def test_missing_matrix_cell_fails(self):
        self.assertFalse(numerical_gate(numeric_matrix()[1:])["passed"])

    def test_missing_threshold_boundary_fails(self):
        self.assertFalse(numerical_gate(numeric_matrix()[:-1])["passed"])

    def test_missing_lifecycle_fails(self):
        rows = [row for row in numeric_matrix() if row["kind"] != "lifecycle"]
        self.assertIn("missing_cancel_restore_prefixes", numerical_gate(rows)["errors"])

    def test_free_generation_divergent_history_not_teacher_forcing(self):
        rows = numeric_matrix()
        rows[0]["candidate_history_sha256"][5] = "f" * 64
        self.assertFalse(numerical_gate(rows)["passed"])

    def test_topk_rows_cannot_claim_full_vocabulary(self):
        rows = numeric_matrix()
        rows[0]["vocabulary_size"] = 100
        self.assertFalse(numerical_gate(rows)["passed"])

    def test_new_clip_rejected(self):
        rows = numeric_matrix()
        rows[0]["new_clip_count"] = 1
        self.assertFalse(numerical_gate(rows)["passed"])

    def test_local_failure_not_averaged_away(self):
        rows = numeric_matrix()
        rows[0]["candidate_logits"][0] = [0.0, 1.0, 2.0]
        result = numerical_gate(rows)
        self.assertFalse(result["passed"])
        self.assertTrue(any(error.startswith("stratum_numerical_gate:prefix:") for error in result["errors"]))

    def test_31_positions_insufficient(self):
        rows = numeric_matrix()
        rows[0]["forced_token_ids"].pop()
        self.assertFalse(numerical_gate(rows)["passed"])


class PerformanceMath(unittest.TestCase):
    def test_speed_gate_equal_weight_five_lengths(self):
        report = speed_gate(speed_evidence())
        self.assertTrue(report["passed"], report["errors"])
        self.assertAlmostEqual(report["cold_p2_geometric_speedup"], 1.25)

    def test_missing_length_not_silently_dropped(self):
        data = speed_evidence()
        del data["cold_p2"]["130552"]
        self.assertFalse(speed_gate(data)["passed"])

    def test_failed_run_not_removed_to_four(self):
        data = speed_evidence()
        data["cold_p2"]["4096"]["candidate_pair_s"].pop()
        self.assertFalse(speed_gate(data)["passed"])

    def test_per_request_regression_blocks_pair_speedup(self):
        data = speed_evidence()
        data["cold_p2"]["4096"]["request_latency"][0]["candidate_s"] = [10.6] * 5
        self.assertIn("request_latency_regression:4096:0", speed_gate(data)["errors"])

    def test_short_p95_requires_hundred(self):
        data = speed_evidence()
        data["short_latency"]["candidate_s"] = [8] * 99
        self.assertFalse(speed_gate(data)["passed"])

    def test_mixed_regression_blocks_speed_gate(self):
        data = speed_evidence()
        data["mixed"]["short_ttft_s"]["candidate"] = [11.1] * 5
        self.assertIn("mixed_regression:short_ttft_s", speed_gate(data)["errors"])

    def test_warm_regression_blocks_speed_gate(self):
        data = speed_evidence()
        data["warm_ttft"]["2048"]["candidate_s"] = [11.1] * 5
        self.assertIn("warm_regression:2048", speed_gate(data)["errors"])

    def test_speedup_below_threshold_fails(self):
        data = speed_evidence()
        for row in data["cold_p2"].values():
            row["candidate_pair_s"] = [9.9] * 5
        self.assertFalse(speed_gate(data)["passed"])

    def test_exact_ten_percent_elapsed_reduction_passes(self):
        data = speed_evidence()
        for row in data["cold_p2"].values():
            row["candidate_pair_s"] = [9.0] * 5
        report = speed_gate(data)
        self.assertTrue(report["passed"], report["errors"])
        self.assertAlmostEqual(report["cold_p2_geometric_elapsed_ratio"], .90)
        self.assertAlmostEqual(report["cold_p2_geometric_speedup"], 1 / .90)

    def test_just_under_ten_percent_elapsed_reduction_fails(self):
        data = speed_evidence()
        for row in data["cold_p2"].values():
            row["candidate_pair_s"] = [9.00001] * 5
        self.assertFalse(speed_gate(data)["passed"])

    def test_ten_percent_throughput_gain_is_insufficient(self):
        data = speed_evidence()
        for row in data["cold_p2"].values():
            row["candidate_pair_s"] = [10 / 1.10] * 5
        self.assertFalse(speed_gate(data)["passed"])

    def test_memory_saving_has_different_acceptance(self):
        report = memory_gate(memory_evidence())
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["owned_bytes_saved"], 4 * 1024 ** 3)

    def test_rss_saving_is_not_owned_physical_saving(self):
        data = memory_evidence()
        data["accounting"] = "rss"
        self.assertFalse(memory_gate(data)["passed"])

    def test_headroom_must_improve(self):
        data = memory_evidence()
        data["candidate_coexist_headroom_bytes"] = 100
        self.assertIn("coexist_headroom_not_improved", memory_gate(data)["errors"])

    def test_memory_gain_cannot_hide_latency_regression(self):
        data = memory_evidence()
        data["major_latencies"]["warm_128"]["candidate_s"] = [1.04] * 5
        self.assertIn("latency_regression:warm_128", memory_gate(data)["errors"])

    def test_missing_memory_peak_blocks_gate(self):
        data = memory_evidence()
        del data["candidate_startup_peak_bytes"]
        self.assertFalse(memory_gate(data)["passed"])


class PlanAndCli(unittest.TestCase):
    def test_plan_is_not_measurement(self):
        plan = build_plan()
        self.assertEqual(plan["kind"], "unexecuted_plan")
        self.assertEqual(len(plan["cold"]), 10)
        self.assertTrue(all(row["server_parallel"] == 2 for row in plan["cold"]))

    def test_warm_has_twelve_turns_per_conversation(self):
        rows = build_plan()["warm"]
        self.assertEqual(len(rows), 72)
        self.assertEqual([row["conversation"] for row in rows[:3]], ["A", "B", "A"])
        for increment in (128, 512, 2048):
            for conversation in ("A", "B"):
                self.assertEqual(sum(row["added_input_tokens"] == increment and row["conversation"] == conversation for row in rows), 12)

    def test_synthetic_content_differs_at_all_three_locations(self):
        first, second = synthetic_segments(0), synthetic_segments(1)
        self.assertTrue(all(first[key] != second[key] for key in first))

    def test_synthetic_checks_never_claim_production_qualification(self):
        report = evaluate(dict(schema_version=1, evidence_kind="synthetic_test", checks={"speed": speed_evidence()}))
        self.assertTrue(report["selected_checks_passed"])
        self.assertFalse(report["production_qualified"])

    def test_captured_checks_still_not_full_qualification(self):
        report = evaluate(dict(schema_version=1, evidence_kind="captured", checks={"speed": speed_evidence()}))
        self.assertFalse(report["production_qualified"])

    def test_unknown_provenance_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(dict(schema_version=1, checks={"speed": speed_evidence()}))

    def test_empty_checks_not_vacuous_success(self):
        with self.assertRaises(ValueError):
            evaluate(dict(schema_version=1, evidence_kind="captured", checks={}))

    def test_invalid_check_shape_rejected(self):
        for check, data in (("readiness", []), ("numerics", {}), ("numerics", [None])):
            with self.subTest(check=check), self.assertRaises(ValueError):
                evaluate(dict(schema_version=1, evidence_kind="captured", checks={check: data}))

    def test_boolean_schema_version_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(dict(schema_version=True, evidence_kind="captured", checks={"speed": speed_evidence()}))

    def test_cli_fixture_generation(self):
        result = subprocess.run([sys.executable, "-m", "bench.evo_x2", "fixtures"], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(strict_json(result.stdout)["cases"]), 60)

    def test_cli_reports_failure_without_contacting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(dict(schema_version=1, evidence_kind="synthetic_test", checks={"readiness": {}})))
            result = subprocess.run([sys.executable, "-m", "bench.evo_x2", "evaluate", str(path)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(strict_json(result.stdout)["selected_checks_passed"])

    def test_cli_rejects_invalid_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{ invalid }")
            result = subprocess.run([sys.executable, "-m", "bench.evo_x2", "evaluate", str(path)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid evidence", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
