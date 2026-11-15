"""Offline fixtures only: no host inventory, services, or engine probes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "evo_x2"))
from capture_manifest import capture, hash_file
from common import load_json, validate_document, write_private_json
from resolve_profile import resolve


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"synthetic model, not real weights")
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text('{"synthetic":true}')
        self.selection = {"schema_version": 1, "assets": [
            {"role": "model_shard", "path": str(self.model)},
            {"role": "tokenizer", "path": str(self.tokenizer)}]}
        self.profile = load_json(ROOT / "tools/evo_x2/examples/baseline-review.profile.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_hashes_selected_files_only(self):
        # Unselected sibling is never opened, even when it is not a regular file.
        os.mkfifo(self.root / "never-read")
        before = self.model.read_bytes()
        result = capture(self.selection)
        self.assertEqual(result["assets"][0]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(result["status"], "inventory_only")
        self.assertNotIn("host", result)
        self.assertNotIn("process", result)

    def test_allowlist_config_never_copies_credentials_or_free_text(self):
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps({"parallel": 2, "api_key": "DO-NOT-COPY",
                                   "prompt": "PRIVATE-PROMPT", "nested": {"password": "OTHER-SECRET"}}))
        self.selection["config"] = {"path": str(cfg), "fields": ["parallel"]}
        result = capture(self.selection)
        encoded = json.dumps(result)
        for secret in ("DO-NOT-COPY", "PRIVATE-PROMPT", "OTHER-SECRET"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(result["configuration"], {"parallel": 2})
        self.assertEqual(result["authentication"], [{"source": "selected_config", "present": True}])

    def test_credential_cannot_be_selected(self):
        self.selection["config"] = {"path": "unused", "fields": ["api_key"]}
        with self.assertRaises(ValueError):
            capture(self.selection)

    def test_unknown_keys_and_roles_fail_before_reads(self):
        for mutate in (lambda s: s.update({"scan_root": "/"}),
                       lambda s: s["assets"][0].update({"role": "logs"}),
                       lambda s: s["assets"][0].update({"recursive": True}),
                       lambda s: s.update({"schema_version": True})):
            selection = copy.deepcopy(self.selection)
            mutate(selection)
            with self.assertRaises(ValueError):
                capture(selection)

    def test_nonregular_and_duplicate_files_rejected(self):
        with self.assertRaises(ValueError):
            hash_file(self.root)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            hash_file(fifo)
        self.selection["assets"].append(self.selection["assets"][0].copy())
        with self.assertRaises(ValueError):
            capture(self.selection)

    def test_selected_process_is_fixture_only(self):
        proc = self.root / "proc" / "123"
        proc.mkdir(parents=True)
        (proc / "stat").write_text('123 (fixture with ) spaces) S ' + ' '.join(['0'] * 18 + ['456', '0']))
        (proc / "exe").symlink_to(self.model)
        (proc / "environ").write_bytes(b"PRIVATE_TOKEN=DO-NOT-COPY\0STRATA_PF_FUSED=0\0")
        self.selection.update({"pid": 123, "environment": ["STRATA_PF_FUSED"]})
        result = capture(self.selection, self.root / "proc")
        self.assertEqual(result["process"]["start_ticks"], "456")
        self.assertEqual(result["environment"], {"STRATA_PF_FUSED": "0"})
        self.assertNotIn("DO-NOT-COPY", json.dumps(result))

    def test_prefill_diagnostics_inventory_is_explicit_not_execution_proof(self):
        proc = self.root / "proc" / "123"
        proc.mkdir(parents=True)
        (proc / "stat").write_text('123 (fixture) S ' + ' '.join(['0'] * 18 + ['456', '0']))
        (proc / "exe").symlink_to(self.model)
        (proc / "environ").write_bytes(
            b"STRATA_PF_F16=0\0STRATA_PF_DM16=0\0STRATA_PF_FUSED_TRACE=1\0"
            b"STRATA_PREFILL_TIMING=0\0PRIVATE_TOKEN=DO-NOT-COPY\0")
        self.selection.update({"pid": 123, "environment": ["STRATA_PF_F16", "STRATA_PF_DM16",
            "STRATA_PF_FUSED_TRACE", "STRATA_PREFILL_TIMING", "STRATA_HIPBLASLT_VERBOSE"]})
        result = capture(self.selection, self.root / "proc")
        self.assertEqual(result["environment"], {"STRATA_PF_F16": "0", "STRATA_PF_DM16": "0",
                         "STRATA_PF_FUSED_TRACE": "1", "STRATA_PREFILL_TIMING": "0"})
        self.assertEqual(result["status"], "inventory_only")
        self.assertIn("quality_and_performance", result["unverified"])
        self.assertNotIn("DO-NOT-COPY", json.dumps(result))
        # Literal values are preserved. Presence-based flags are not normalized to booleans.
        for unknown in ("STRATA_EVO_PF_IQ_MODE", "LD_PRELOAD", "PRIVATE_TOKEN"):
            self.selection["environment"] = [unknown]
            with self.assertRaises(ValueError):
                capture(self.selection, self.root / "proc")

    def test_fused_experiment_changes_one_supported_selector(self):
        recipe = load_json(ROOT / "tools/evo_x2/examples/fused-int8.experiment.json")
        self.assertEqual(recipe["status"], "unqualified_experiment")
        self.assertFalse(recipe["deployable"])
        self.assertEqual(recipe["arms"], {"baseline": {"STRATA_PF_FUSED": "0"},
                                        "fused-int8": {"STRATA_PF_FUSED": "1"}})
        self.assertEqual(recipe["common_environment"]["STRATA_PF_F16"], "0")
        self.assertEqual(recipe["common_environment"]["STRATA_PF_DM16"], "0")
        self.assertNotIn("STRATA_EVO_PF_IQ_MODE", json.dumps(recipe))

    def test_environment_without_pid_refused(self):
        self.selection["environment"] = ["STRATA_PF_FUSED"]
        with self.assertRaises(ValueError):
            capture(self.selection)

    def test_missing_selected_config_and_invalid_scalars_refused(self):
        cfg = self.root / "config.json"
        self.selection["config"] = {"path": str(cfg), "fields": ["parallel"]}
        for value in ({}, {"parallel": True}, {"parallel": "secret"}):
            cfg.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                capture(self.selection)

    def test_unit_file_contents_are_never_serialized(self):
        unit = self.root / "example.service"
        unit.write_text('[Service]\nEnvironment=API_KEY=PRIVATE-KEY\n')
        self.selection["service"] = {"name": "example.service", "scope": "user", "unit_files": [str(unit)]}
        result = capture(self.selection)
        self.assertFalse(result["service"]["binding_verified"])
        self.assertNotIn("PRIVATE-KEY", json.dumps(result))

    def test_private_output_refuses_overwrite_and_symlink(self):
        out = self.root / "out.json"
        write_private_json(out, {"ok": True})
        self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            write_private_json(out, {})
        link = self.root / "link"
        link.symlink_to(out)
        with self.assertRaises(FileExistsError):
            write_private_json(link, {})

    def test_json_duplicate_and_nonfinite_refused(self):
        path = self.root / "bad.json"
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_json(path)

    def test_review_plan_never_claims_readiness(self):
        result = resolve(self.profile, capture(self.selection), review_only=True)
        self.assertFalse(result["deployable"])
        self.assertEqual(result["status"], "blocked_pending_qualification")
        self.assertEqual(result["server_config_proposal"]["evo_contract"], True)
        self.assertEqual(result["layout"], "baseline")
        self.assertNotIn("launch", result)

    def test_unqualified_production_rejected(self):
        with self.assertRaises(ValueError):
            resolve(self.profile, capture(self.selection))

    def test_unimplemented_modes_and_unknown_nested_keys_rejected(self):
        mutations = [lambda p: p["kernels"].update({"direct_fp16": "pair128"}),
                     lambda p: p["kernels"].update({"direct_fp16": "direct64"}),
                     lambda p: p["kernels"].update({"split_topk": "split"}),
                     lambda p: p["memory"].update({"layout": "gpu_full_complement"}),
                     lambda p: p["memory"].update({"explicit_allocation_guard": True}),
                     lambda p: p["qualification"].update({"passed": True}),
                     lambda p: p["contract"].update({"parallel_requests": 1}),
                     lambda p: p["contract"].update({"fit_max_tokens": True})]
        for mutate in mutations:
            profile = copy.deepcopy(self.profile)
            mutate(profile)
            with self.assertRaises(ValueError):
                resolve(profile, capture(self.selection), review_only=True)

    def test_cli_round_trip_and_sanitized_error(self):
        selection = self.root / "selection.json"
        selection.write_text(json.dumps(self.selection))
        output = self.root / "manifest.json"
        script = ROOT / "tools/evo_x2/capture_manifest.py"
        run = subprocess.run([sys.executable, str(script), '--selection', str(selection), '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        validate_document(load_json(output), "manifest")
        selection.write_text('{"DO-NOT-ECHO":invalid}')
        run = subprocess.run([sys.executable, str(script), '--selection', str(selection), '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertNotIn("DO-NOT-ECHO", run.stderr)

    def test_resolver_cli_review_and_production_refusal(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(capture(self.selection)))
        output = self.root / "review.json"
        command = [sys.executable, str(ROOT / "tools/evo_x2/resolve_profile.py"),
                   "--profile", str(ROOT / "tools/evo_x2/examples/baseline-review.profile.json"),
                   "--manifest", str(manifest), "--output", str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertFalse(output.exists())
        run = subprocess.run(command + ["--review-only"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(load_json(output)["deployable"])

    def test_shipped_schema_matches_validation(self):
        # Optional cross-check with a full standards validator in development.
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema unavailable; runtime tools do not require it")
        for name, document in (("selection", self.selection), ("manifest", capture(self.selection)), ("profile", self.profile)):
            schema = load_json(ROOT / 'tools/evo_x2/schemas' / (name + '.schema.json'))
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(document, schema)


if __name__ == '__main__':
    unittest.main()
