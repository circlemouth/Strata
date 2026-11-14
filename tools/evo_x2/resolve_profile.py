#!/usr/bin/env python3
"""Validate a baseline profile and emit a non-deployable review plan.

No configuration is installed, process launched, or qualification inferred.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
try:
    from .common import load_json, validate_document, write_private_json
except ImportError:
    from common import load_json, validate_document, write_private_json


def identity(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def resolve(profile, manifest, *, review_only=False):
    validate_document(profile, "profile")
    validate_document(manifest, "manifest")
    if not review_only:
        raise ValueError("deployment resolution unavailable: hardware qualification and allocation guards are not implemented")
    roles = {a["role"] for a in manifest["assets"]}
    if not roles.intersection({"model_shard", "pack"}) or "tokenizer" not in roles:
        raise ValueError("review requires explicitly selected model and tokenizer assets")
    return {
        "schema_version": 1,
        "status": "blocked_pending_qualification",
        "deployable": False,
        "profile": profile["profile"],
        "profile_sha256": identity(profile),
        "manifest_sha256": identity(manifest),
        "layout": "baseline",
        "server_config_proposal": {"evo_contract": True, "fit_max_tokens": False, "parallel": 2},
        "required_engine_observations": {"max_context": 131072, "batch_slots": 2},
        "inherited_settings": ["kv_dtype", "sampling", "thinking", "vision", "drafter"],
        "blocking_checks": ["complete_and_bind_runtime_manifest", "freeze_effective_inherited_settings",
                            "verify_device_and_build", "qualify_baseline_memory_and_cgroup_budget",
                            "verify_actual_p2_progress", "qualify_api_clients_and_quality",
                            "qualify_latency_throughput_and_co_resident_services"],
        "notes": ["No launch command is emitted.",
                  "Baseline does not satisfy or claim all-expert residency.",
                  "Hashes bind these input documents; they do not attest to hardware or qualification."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    try:
        result = resolve(load_json(args.profile), load_json(args.manifest), review_only=args.review_only)
        write_private_json(args.output, result)
    except (OSError, ValueError, KeyError) as exc:
        print(f"profile resolution refused ({type(exc).__name__}); no deployable configuration produced", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
