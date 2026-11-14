"""Opt-in API limits for the EVO qualification candidate, not a performance claim.

No hardware probes, prompt data, credentials, or persisted qualification evidence
belong here. A runtime slot count is necessary, but not proof of concurrent progress.
"""
from __future__ import annotations

CONTEXT_TOKENS = 131072
OUTPUT_TOKENS = 16384
CONTEXT_SLACK = 8
PARALLEL_REQUESTS = 2
OUTPUT_ALIASES = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def normalize_output_limit(request: dict) -> int:
    """Validate all aliases before coercion or shared defaults can hide mistakes."""
    values = []
    for name in OUTPUT_ALIASES:
        value = request.get(name)
        if value is None:
            continue
        if type(value) is not int or not 1 <= value <= OUTPUT_TOKENS:
            raise ValueError(f"{name}: expected an integer from 1 to {OUTPUT_TOKENS}")
        values.append((name, value))
    if values and any(value != values[0][1] for _, value in values):
        raise ValueError("output token limits conflict: non-null aliases must agree")
    return values[0][1] if values else OUTPUT_TOKENS


def normalize_request(request: dict, output_key: str) -> dict:
    return {**request, output_key: normalize_output_limit(request)}


def validate_config(config: dict, *, fit_cli: bool = False, lazy_cli: bool = False) -> bool:
    enabled = config.get("evo_contract", False)
    if type(enabled) is not bool:
        raise ValueError("evo_contract must be true or false")
    if not enabled:
        return False
    if fit_cli or config.get("fit_max_tokens") is not False:
        raise ValueError("evo_contract requires explicit fit_max_tokens: false")
    if type(config.get("parallel")) is not int or config["parallel"] != PARALLEL_REQUESTS:
        raise ValueError("evo_contract requires parallel: 2")
    if lazy_cli or config.get("lazy_load") is True:
        raise ValueError("evo_contract cannot validate an unloaded engine; lazy loading is disabled")
    return True


def validate_runtime(engine) -> None:
    """Check the engine's observed values, never its requested configuration."""
    info = getattr(engine, "info", {})
    actual = (getattr(engine, "max_context", None), getattr(engine, "batch", None))
    reported = info.get("batch_slots")
    # INFO is a line protocol, so decimal strings are expected. Booleans are not counts.
    if type(reported) is str and reported.isdecimal():
        reported = int(reported)
    if (any(type(value) is not int for value in actual) or
            actual != (CONTEXT_TOKENS, PARALLEL_REQUESTS) or
            type(reported) is not int or reported != PARALLEL_REQUESTS):
        raise ValueError("EVO contract unavailable: observed context must be 131072 and both "
                         "engine batch and INFO batch_slots must be 2; no downgrade is permitted")


def effective_contract() -> dict:
    return {"context_tokens": CONTEXT_TOKENS, "parallel_requests": PARALLEL_REQUESTS,
            "default_output_tokens": OUTPUT_TOKENS, "max_output_tokens": OUTPUT_TOKENS,
            "context_slack_tokens": CONTEXT_SLACK, "fit_max_tokens": False,
            "qualification_required": True,
            "concurrent_progress_verified": False}
