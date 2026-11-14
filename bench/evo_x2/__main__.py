"""Read-only CLI for plans, fixtures and explicitly scoped offline evidence checks."""

import argparse
import json
import sys
from pathlib import Path

from .contract import validate_mixed_trigger, validate_performance_cell, validate_readiness
from .fixtures import build_fixtures, strict_json, structural_regressions
from .numerics import numerical_gate
from .performance import memory_gate, speed_gate
from .workloads import build_plan, frozen_digest


def evaluate(document):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("evidence schema_version must be 1")
    kind = document.get("evidence_kind")
    if kind not in ("synthetic_test", "captured"):
        raise ValueError("evidence_kind must be synthetic_test or captured")
    checks = document.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("at least one named check is required")
    results = {}
    for name, data in checks.items():
        if name == "numerics":
            if not isinstance(data, list):
                raise ValueError("numerics check must be a list of prefix objects")
        elif not isinstance(data, dict):
            raise ValueError(f"{name} check must be an evidence object")
        if name in ("readiness", "cold_cell", "mixed_trigger"):
            validator = {"readiness": validate_readiness, "cold_cell": validate_performance_cell,
                         "mixed_trigger": validate_mixed_trigger}[name]
            errors = validator(data)
            results[name] = {"passed": not errors, "errors": errors}
        elif name == "numerics":
            results[name] = numerical_gate(data)
        elif name == "speed":
            results[name] = speed_gate(data)
        elif name == "memory":
            results[name] = memory_gate(data)
        elif name == "structure":
            result = structural_regressions(build_fixtures(), data["reference"], data["candidate"])
            result["passed"] = result["structural_gate_passed"]
            results[name] = result
        else:
            raise ValueError(f"unknown check: {name}")
    return {"schema_version": 1, "evidence_kind": kind, "checks": results,
            "selected_checks_passed": all(result["passed"] for result in results.values()),
            "production_qualified": False,
            "qualification_note": "Offline checks only. Hardware, provenance, full lifecycle, coexistence, blind prose review and soak evidence need separate verification."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fixtures", "evaluate"))
    parser.add_argument("input", nargs="?", type=Path, help="explicit local JSON evidence file for evaluate")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            if args.input is None:
                parser.error("evaluate requires an explicit local JSON path")
            if args.input.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("evidence JSON exceeds 64 MiB; split independent checks")
            result = evaluate(strict_json(args.input.read_text(encoding="utf-8")))
        elif args.command == "plan":
            result = build_plan()
        else:
            cases = build_fixtures()
            result = {"schema_version": 1, "synthetic_only": True,
                      "fixtures_sha256": frozen_digest(cases), "cases": cases}
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
        return int(args.command == "evaluate" and not result["selected_checks_passed"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"invalid evidence: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
