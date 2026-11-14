"""Sixty invented Japanese cases and strict structural regression scoring.

The fictional names and quantities below are test data, not medical advice.
"""

import json
import math


def build_fixtures():
    cases = []
    for i in range(20):
        side = "左" if i % 2 == 0 else "右"
        body = ("前腕", "下腿", "頬", "肩")[i % 4]
        date = f"2030-04-{i + 1:02d}"
        expected = {"body_site": body, "side": side, "date": date,
                    "count": i % 5 + 1, "concentration": f"{i % 3 + 1}%",
                    "dose": i % 4 + 1, "unit": "mL", "pain_present": i % 3 == 0,
                    "drug": f"架空薬{chr(65 + i % 4)}"}
        pain = "痛みあり" if expected["pain_present"] else "痛みなし"
        text = (f"完全に架空の記録です。{date}、{side}の{body}、{expected['count']}回。"
                f"薬剤名は{expected['drug']}、濃度{expected['concentration']}、量{expected['dose']} mL。{pain}。")
        cases.append({"id": f"extract-{i + 1:02d}", "category": "extraction",
                      "prompt": text + " 指定されたキーのJSONだけを返してください。日付はYYYY-MM-DD、数値は数値、痛みはboolean。"
                      + " キー: " + ", ".join(expected), "expected": expected})
    for i in range(20):
        side = "右" if i % 2 == 0 else "左"
        symptom = ("かゆみ", "赤み", "乾燥", "違和感")[i % 4]
        subjective = f"{side}の腕に{symptom}がある。痛みはない。"
        objective = f"記録上の計測値は{i + 1} mm。"
        expected = {"S": subjective, "O": objective, "A": None, "P": None}
        cases.append({"id": f"soap-{i + 1:02d}", "category": "organization",
                      "prompt": "架空の会話です。本人「" + subjective + "」 記録者「" + objective
                      + "」 S/O/A/PのJSONに整理してください。SとOには発言内の文をそのまま使い、"
                      "示されていない評価・計画はnullにしてください。所見・診断・処方を追加しないでください。",
                      "expected": expected})
    for i in range(20):
        code = f"SYN-{i + 1:03d}"
        label = f"項目\"{i + 1}\"\n次の行"
        # Half the cases exercise tool-shaped objects, half ordinary JSON.
        args = {"code": code, "state": ("pending", "complete")[i % 2],
                "note": label, "optional": None, "enabled": i % 2 == 0}
        expected = {"name": "record_synthetic_item", "arguments": args} if i % 2 else args
        cases.append({"id": f"schema-{i + 1:02d}", "category": "schema",
                      "prompt": "次の架空データを同じ構造のJSONだけで返してください。キー、null、boolean、"
                      "enum値、引用符、改行を保持してください。\n"
                      + json.dumps(expected, ensure_ascii=False), "expected": expected})
    return cases


def strict_json(text):
    """Reject duplicate keys and non-JSON numbers rather than silently rewriting."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON number overflows finite range")
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def mismatch_paths(expected, actual, path="$"):
    """Compare every critical field. No fuzzy match, coercion, or hidden extras."""
    if type(actual) is not type(expected):
        return [path]
    if isinstance(expected, dict):
        errors = [f"{path}/{key}" for key in sorted(set(actual) - set(expected))]
        for key in expected:
            errors.extend([f"{path}/{key}"] if key not in actual
                          else mismatch_paths(expected[key], actual[key], f"{path}/{key}"))
        return errors
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [path]
        return [error for i, (a, b) in enumerate(zip(expected, actual))
                for error in mismatch_paths(a, b, f"{path}/{i}")]
    return [] if expected == actual else [path]


def score_outputs(fixtures, outputs):
    """outputs is a case-ID to raw JSON answer mapping; absence is a failure."""
    ids = [case["id"] for case in fixtures]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fixture IDs")
    if not isinstance(outputs, dict):
        raise ValueError("outputs must map fixture IDs to raw JSON strings")
    if set(outputs) - set(ids):
        raise ValueError("unknown output case IDs")
    results = {}
    for case in fixtures:
        raw = outputs.get(case["id"])
        try:
            if not isinstance(raw, str):
                raise ValueError("missing raw JSON output")
            errors = mismatch_paths(case["expected"], strict_json(raw))
        except (ValueError, TypeError):
            errors = ["$invalid_json_or_missing"]
        results[case["id"]] = {"passed": not errors, "errors": errors, "category": case["category"]}
    return {"case_count": len(fixtures), "passed": sum(x["passed"] for x in results.values()), "cases": results}


def structural_regressions(fixtures, baseline, candidate):
    before, after = score_outputs(fixtures, baseline), score_outputs(fixtures, candidate)
    regressions = []
    for case in fixtures:
        key = case["id"]
        old, new = before["cases"][key]["errors"], after["cases"][key]["errors"]
        # An invalid baseline cannot establish correctness of any field. Keep the
        # case visible as unverified; do not count it as a numerical improvement.
        if "$invalid_json_or_missing" in old:
            continue
        added = sorted(set(new) - set(old))
        if added:
            regressions.append({"id": key, "new_error_paths": added})
    unresolved = [key for key, value in before["cases"].items() if "$invalid_json_or_missing" in value["errors"]]
    return {"scope": "structural_fields_only", "baseline": before, "candidate": after,
            "new_critical_errors": regressions, "unverified_baseline_cases": unresolved,
            "structural_gate_passed": not regressions and not unresolved,
            "blind_prose_review": "required_separately"}
