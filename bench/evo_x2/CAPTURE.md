# Synthetic replay and bounded stream capture

This adapter adds a concrete collection path to the offline checks. Replay needs
no server, socket, model, credentials or GPU. It captures selected metadata,
timestamps, hashes and exact API usage. It does not keep prompt or answer text
in the report, and it never turns an SSE chunk into an engine token.

An optional HTTP adapter is disabled by default. When explicitly enabled, it
sends exactly two text-only synthetic requests to a separately prepared local
test server. It does not start, stop, restart, reconfigure or discover services.
It does not authenticate, read credentials, follow redirects, consult proxies,
enable tools/MCP, or call external hosts. No real server was contacted by the
included tests. Loopback HTTP behavior is tested with mocked connections only.

## Offline replay

From the repository root:

```sh
python -m bench.evo_x2.runner replay bench/evo_x2/fixtures/sse_replay_v1.json
python -m unittest bench.evo_x2.test_capture -v
```

The fixture includes invented frames and invented usage counts. Its text has
not been encoded into a 4096-token runtime prompt. It is only a parser/runner
test and must not be reported as a model benchmark. Every replay result is
tagged `evidence_kind: synthetic_test` and `production_qualified: false`.

A replay bundle has `schema_version: 1`, `synthetic_only: true`, exactly two
`requests`, and `streams` indexed by request ID. Each request contains:

- `id`: a distinct short ASCII identifier.
- `target_input_tokens`: the intended final encoded count, not a claim that the
  fixture has already been tokenized.
- `payload`: an explicit model alias, plain text messages, `stream: true`, an
  explicit 512 output limit, and the frozen sampling settings.

A stream is a list of `{ "at_s": number, "data": "SSE bytes as UTF-8 text" }`
objects. Offsets are synthetic and relative to that request's send timestamp.
The Python fake-stream API also accepts byte fragments, allowing tests to split
UTF-8 characters and CRLF across network boundaries. Replay does not sleep.

The exact serialized request bytes are hashed before capture. Sampling and
template fields are never added or silently changed. The supported payload
allowlist is deliberately small. It accepts plain role/content messages and
optional no-thinking template kwargs, but rejects tools, MCP, image URLs,
credential fields and arbitrary request metadata.

## Separately authorized local test capture

Only use this mode after independently preparing and authorizing an isolated
local test server and exact, frozen synthetic requests. The adapter cannot
determine whether a loopback port belongs to production. Do not point it at an
unapproved existing service.

```sh
python -m bench.evo_x2.runner loopback prepared-synthetic-requests.json \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --enable-loopback
```

Both endpoint and enable flag are required. The URL must use HTTP, a literal
loopback IP, an explicit port and exactly `/v1/chat/completions`. `localhost`,
non-loopback IPs, userinfo, alternate paths, queries and fragments are rejected
before connecting. HTTP 3xx/401/403 responses remain failures; no redirect or
login is attempted. There are no environment-based proxy or credential fallbacks.

The two workers meet at a barrier before sending. The report records the actual
client-dispatch start skew. This does not prove simultaneous engine progress;
that requires separate engine evidence. There are no automatic retries or
quietly discarded failed members. A failed or partial member stays in the pair.
Connections close after the SSE terminator or a failure. Socket inactivity and
total-duration bounds apply. A timeout does not prove normal engine completion.

## Stream evidence

The parser handles split UTF-8, LF/CRLF, comments, multiline `data` fields and
multiple events arriving in one read. It requires one choice, a stable response
ID, a supported finish reason, exact integer usage and `[DONE]`. A partial,
malformed, oversized, duplicate-usage or error-bearing stream is a failure.

Recorded fields include send time, first response byte, first nonempty answer
time, first nonempty reasoning-or-answer time, SSE event timestamps, final
usage, finish reason, selected API timing aggregates and text hashes/byte counts.
An empty role delta is not a first token. Multiple events in one read share the
observed arrival timestamp. Missing API cached-token usage stays null, never an
assumed zero. API timing aggregates do not become engine token timestamps.

The pinned upstream implementation creates an API response ID independently
from its internal engine request ID. It also maps a cancelled engine run to API
`finish_reason: stop`. These are static source findings, not measured behavior:
[pinned server implementation](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/serve/server.py).
For that reason, a fully received API stream still has
`normal_completion: false` until explicit engine correlation succeeds.

## Correlate separately supplied engine evidence

The runner does not scrape arbitrary logs or guess which internal request
belongs to which response. An instrumented local collector must record the
actual request binding, final tokenization and engine events. This package does
not add that server/engine hook. Existing slot snapshots or timestamp proximity
are insufficient. If the binding is unavailable, leave correlation incomplete.

```sh
python -m bench.evo_x2.runner correlate capture.json \
  --engine-evidence engine-sidecar.json
```

The sidecar envelope requires:

- `schema_version: 1`, exact matching `run_id`, `clock_id` and `evidence_kind`.
- `clock_binding: shared_monotonic_domain`, established by the collector rather
  than estimated from network timing. The helper does not fit clock offsets.
- A SHA256 for the frozen runtime manifest in `manifest_sha256`.
- Observed `configured_parallel: 2`, `props_total_slots: 2` and
  `fit_max_tokens: false`.
- Exactly two `records`, joined by explicit client request ID rather than order.

Each sidecar record has three objects:

1. `binding`: `method: instrumented_request_binding`, `client_request_id`, exact
   wire `request_sha256`, `api_response_id`, and `engine_request_id`.
2. `tokenization`: the same request and manifest hashes,
   `source: runtime_tokenizer` (`synthetic_tokenizer` only for replay tests),
   `includes_template_and_expansion: true`, final `encoded_input_tokens`, and
   tokenizer/template SHA256 values. Final count must equal both the intended
   length and exact API prompt usage.
3. `engine`: matching `request_id`, actual `slot`, `batch_slots`, `context_tokens`,
   `cache_n`, `completion_tokens`, `terminal_status: completed`, `resident: true`,
   and explicit false `cancelled`, `disconnected`, `truncated` and
   `output_limit_adjusted` flags. Include exactly 512 actual accepted-token
   timestamps in `token_timestamps_s`, all within the correlated request's
   interval. `prefill_started_s` is optional and must be an observed engine event.

Cold P2 correlation requires zero engine cache, matching API cached usage,
different internal requests, slots 0 and 1, and overlapping engine progress.
Neither a configuration value nor overlapping HTTP lifetimes proves P2. A
single missing or mismatched binding, clock, tokenization, slot, cache, terminal
state or token timestamp blocks P2 metrics. No SSE-based replacement is made.

Successful correlation exposes scoped cold-cell/P2 arithmetic. Synthetic values
remain labelled synthetic. Sidecars are supplied evidence, not cryptographically
authenticated proof. Both parser capture and correlation always keep
`production_qualified: false`; model identity, numerical/state correctness,
quality, memory, lifecycle, coexistence and soak requirements remain separate.

Exit code 0 means the selected operation completed (API streams for capture,
correlation checks for correlate), 1 means an observed check failed, and 2 means
invalid input or setup. An exit code is not a production acceptance decision.
