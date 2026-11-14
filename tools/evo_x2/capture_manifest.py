#!/usr/bin/env python3
"""Capture explicitly selected local files and one Linux process, without mutation.

This is a partial inventory, never a runtime qualification or readiness probe.
No shell command, network request, process enumeration, recursive walk or logs.
"""
import argparse
import hashlib
import os
import platform
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .common import load_json, validate, validate_document, write_private_json
except ImportError:
    from common import load_json, validate, validate_document, write_private_json

CONFIG_FIELDS = {
    "ctx": {"type": "integer", "minimum": 1},
    "parallel": {"type": "integer", "minimum": 1},
    "fit_max_tokens": {"type": "boolean"},
    "temperature": {"type": "number", "minimum": 0},
    "top_p": {"type": "number", "minimum": 0, "maximum": 1},
    "top_k": {"type": "integer", "minimum": 0},
    "presence_penalty": {"type": "number"},
}
AUTH_FIELDS = ("api_key", "api-key", "token", "password", "secret", "authorization")
ENV_FIELDS = ("STRATA_PF_FUSED", "STRATA_PF_FUSED_NATIVE", "STRATA_PREFILL_MMQ")


def hash_file(path):
    """Hash only a named regular file; reject in-flight replacements/changes."""
    resolved = Path(path).resolve(strict=True)
    fd = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        os.close(fd)
        raise ValueError("selected asset must be a regular file")
    with os.fdopen(fd, "rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    current = resolved.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError("selected file changed during capture")
    return {"path": str(resolved), "sha256": digest.hexdigest(), "bytes": before.st_size}


def selected_config(path, fields):
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    result = {}
    for name in fields:
        if name not in data:
            raise ValueError("selected configuration key is missing")
        result[name] = validate(data[name], CONFIG_FIELDS[name], "selected config")
    # Record presence only. Never serialize free-form config or credential values.
    return result, [{"source": "selected_config", "present": True}
                    for _ in [0] if any(bool(data.get(key)) for key in AUTH_FIELDS)]


def process_start(proc):
    # comm may contain spaces and parentheses; starttime is field 22.
    text = (proc / "stat").read_text()
    try:
        start = text.rsplit(")", 1)[1].split()[19]
    except IndexError as exc:
        raise ValueError("invalid selected process identity") from exc
    if not start.isascii() or not start.isdigit():
        raise ValueError("invalid selected process identity")
    return start


def capture(selection, proc_root=Path("/proc")):
    validate_document(selection, "selection")
    manifest = {
        "schema_version": 1,
        "status": "inventory_only",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "assets": [], "configuration": {}, "authentication": [],
        "environment": {},
        "unverified": ["loaded_asset_identity", "tensor_geometry", "device_and_toolchain",
                       "memory_budget_and_cgroups", "expert_residency", "runtime_p2",
                       "quality_and_performance", "co_resident_services"],
    }
    seen = set()
    for asset in selection["assets"]:
        record = hash_file(asset["path"])
        if record["path"] in seen:
            raise ValueError("duplicate selected asset")
        seen.add(record["path"])
        manifest["assets"].append({"role": asset["role"], **record})
    if "config" in selection:
        cfg = selection["config"]
        manifest["configuration"], manifest["authentication"] = selected_config(cfg["path"], cfg["fields"])
    if "pid" in selection:
        proc = proc_root / str(selection["pid"])
        before = process_start(proc)
        manifest["process"] = {"pid": selection["pid"], "start_ticks": before,
                               "executable": hash_file(proc / "exe")}
        if selection.get("environment"):
            env = (proc / "environ").read_bytes().split(b"\0")
            values = dict(entry.split(b"=", 1) for entry in env if b"=" in entry)
            for name in selection["environment"]:
                raw = values.get(name.encode())
                if raw is not None:
                    value = raw.decode("ascii")
                    if value not in ("0", "1"):
                        raise ValueError("selected environment setting has unsupported value")
                    manifest["environment"][name] = value
        if process_start(proc) != before:
            raise ValueError("selected process changed during capture")
    elif selection.get("environment"):
        raise ValueError("environment selection requires an explicit PID")
    if "service" in selection:
        svc = selection["service"]
        manifest["service"] = {"scope": svc["scope"], "name": svc["name"],
                               "unit_files": [hash_file(p) for p in svc["unit_files"]],
                               "binding_verified": False}
    if selection.get("host_metadata", False):
        manifest["host"] = {"hostname": platform.node(), "kernel": platform.release(),
                            "machine": platform.machine()}
    return validate_document(manifest, "manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        write_private_json(args.output, capture(load_json(args.selection)))
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        # Do not echo paths, raw input, credential-bearing values, or parser excerpts.
        print(f"capture failed ({type(exc).__name__}); no complete inventory produced", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
