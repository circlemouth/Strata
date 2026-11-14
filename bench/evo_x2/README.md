# Offline qualification helpers

This directory contains a deterministic workload plan, sixty fictional Japanese
fixtures, independent contract checks, and offline comparison tools. The `plan`,
`fixtures`, `evaluate` and synthetic replay commands do not contact a server,
load a model, read service configuration, start or stop a process, or use a GPU.
They require only the Python standard library. The separate [stream capture
adapter](CAPTURE.md) has an explicit, disabled-by-default loopback-only mode for
an independently prepared local test server. Its tests use no networking.

These are qualification building blocks, not completed hardware qualification.
Unit-test timings and example values are synthetic. They are not model or device
performance results. The CLI always reports `production_qualified: false`.

## Run local checks

Run from the repository root:

```sh
python -m unittest discover -s bench/evo_x2 -p 'test_*.py' -v
python -m bench.evo_x2 plan
python -m bench.evo_x2 fixtures
python -m bench.evo_x2 evaluate /explicit/path/to/evidence.json
```

`plan` and `fixtures` print JSON. They do not execute requests. `evaluate` reads
only the explicit input file and writes its report to stdout. Exit status is 0
when the selected checks pass, 1 when a selected check fails, and 2 for invalid
input. Passing a selection is not passing the complete qualification protocol.
Input is limited to 64 MiB. The Python API supports larger in-memory datasets;
large full-vocabulary dumps should use a separate streaming collector.

The envelope is:

```json
{
  "schema_version": 1,
  "evidence_kind": "captured",
  "checks": {
    "readiness": {},
    "cold_cell": {},
    "mixed_trigger": {},
    "numerics": [],
    "speed": {},
    "memory": {},
    "structure": {"reference": {}, "candidate": {}}
  }
}
```

This empty example intentionally fails. Include only the checks being evaluated
and supply their evidence. Use `synthetic_test`, not `captured`, for invented
values. All numeric times are seconds on one monotonic clock for a run.

## Contract and workload

The oracle in `contract.py` uses 131072 context tokens, eight slack tokens,
parallelism two, and an output cap/default of 16384. Supported output aliases are
`max_tokens`, `max_completion_tokens`, and `max_output_tokens`. Non-null aliases
must agree. Booleans, floating point values, strings, zero, negative values and
values above the cap are invalid. Null or omission uses the default cap.

Boundary pairs cover 130552/130553 with 512 output tokens, 126968/126969 with
4096, and 114680/114681 with 16384. Counts refer to the final encoded sequence,
including system text, tools, the template and any image expansion. The package
does not have the runtime tokenizer and never labels character counts as tokens.
`synthetic_segments()` provides different start/middle/end signals for assembling
prompts. An authorized local collector must pad, encode, verify and freeze the
actual requests before comparing arms.

`build_plan()` describes five cold lengths at P1 and P2, an alternating two-
conversation warm workload (12 turns per conversation, for each of three input
increments), two event-driven mixed workloads, boundaries, lifecycle cases and
separate coexistence measurements. P1 still uses the P2-configured server. Cold
means no prefix reuse, not loading the model from disk for every request.

Readiness evidence requires the four integer fields `context_tokens`,
`configured_parallel`, `props_total_slots`, `engine_batch_slots`, plus
`fit_max_tokens: false`. `progress` contains two objects with distinct
`request_id`, engine `slot` (0 and 1), and increasing `engine_progress_s` lists.
Their progress intervals must overlap. Configuration alone does not prove P2.

Cold-cell evidence fields are shown in `test_contract.valid_cell`. Every field
checked there is required, including explicit false cancellation/disconnect/
truncation/adjustment flags, tokenizer/template/request SHA256 values, exact
prompt and completion counts, zero engine prefix cache, resident engine, and
normal completion. A failed cell remains failed. Do not discard early EOS or
shortened output and average the remaining runs. `p2_metrics()` additionally
requires distinct content, equal lengths, overlapping requests and barrier
dispatch. It reports the actual start skew and computes aggregate throughput
from first send to last normal completion.

`stream_metrics()` reports TTFT to nonempty answer content separately from TTFT
including reasoning. Empty role chunks do not count. SSE arrival intervals are
not token intervals. Engine token interval metrics remain null without explicit
engine token timestamps. Quantiles use the nearest-rank convention. Five-run
latency summaries provide median/min/max; short-request p95 qualification needs
at least 100 observations.

For mixed evidence, `scenario` is `decode_protection` or `short_wait`.
`trigger_kind`/`trigger_source` must respectively be `first_content`/`api` or
`prefill_started`/`engine`. `dispatch_mode` must be `observed_event`. Supply
`primary_send_s`, `trigger_s`, `secondary_send_s`, `primary_complete_s`; the
secondary must start after the observed trigger and before primary completion.
A fixed sleep or an API send is not evidence of engine prefill start.

## Structured Japanese answers

`build_fixtures()` creates 20 extraction, 20 SOAP-shaped organization and 20
JSON/tool-shaped cases. Every datum is invented. The fixed JSON snapshot is in
`fixtures/ja_synthetic_v1.json`; its hash is tested against the generator. Freeze
that artifact and its hash before collecting answers. This set is a structural
starter set, not a validated medical evaluation or proof of clinical safety.

The `structure` check takes two mappings from case ID to the **raw JSON answer
string**. Matching is strict and recursive. It preserves null, booleans, enum
values, escaped quotes and newlines, and rejects duplicate keys, extra fields,
non-finite JSON values and missing answers. No output cleanup or fuzzy matching
is applied. Baseline errors remain visible. A newly wrong field fails even if
another field was already wrong. Missing/invalid baseline JSON leaves the case
unverified and blocks the structural gate. Blind comparison of prose omissions
and additions is still required separately.

## Numerical comparisons

`numerical_gate()` requires at least 32 teacher-forced positions for each of:

- Five length groups × P1/P2-slot0/P2-slot1 × cold/reused (30 prefixes).
- Two additional cancel/restore prefixes.
- Before/at/after each enabled split-selection threshold (nine by default).

Each prefix records its ID, `kind`, `length_group`, `execution`, `cache_state`,
`teacher_forced: true`, `full_vocabulary: true`, `new_clip_count: 0`, frozen model
`vocabulary_size`, `forced_token_ids`, `reference_logits`, `candidate_logits`,
and matching `reference_history_sha256` / `candidate_history_sha256` lists.
Each logit row must contain the complete vocabulary at the corresponding forced
position. `kind` is `matrix`, `lifecycle` (with `lifecycle: cancel|restore`), or
`threshold` (with integer `sequence_position`). See `test_qualification` for a
small-vocabulary synthetic example of all required metadata.

The collector must verify vocabulary size and hashes against the actual frozen
assets. Self-reported metadata is not independently authenticated here. API top-k
logprobs and free-generation logits after divergence cannot establish this gate.
KL is computed from stable full-vocabulary log-softmax values. Limits are mean
0.001, p99 0.01, maximum 0.05, top-1 agreement at least 99.5%, and no new clipping
or non-finite values. Aggregate, length, execution, cache and per-prefix results
are retained. The local gate conservatively applies limits to every group so
that an isolated failure cannot be hidden by aggregate averages. Exact-copy
bytes, top-k ID/order parity and kernel dispatch still need independent tests.

## Performance and memory arithmetic

`speed_gate()` accepts all five cold P2 lengths, each with five or more aligned
reference/candidate pair completion times and both request latency series. It
takes the median paired candidate/reference elapsed-time ratio per length, then
the equally weighted geometric mean across lengths. The requirement is at most
0.90 (at least 10% less elapsed time, equivalent to at least 1/0.90× speedup). Each
request median may regress at most 5%; short-request p95 (100+ requests), mixed
decode interval p95, mixed short TTFT and each warm increment may regress at most
10%. `test_qualification.speed_evidence` documents the input shape using only
synthetic arithmetic values. No model benchmark is run by that test.

`memory_gate()` requires at least 4 GiB reduction in deduplicated owned physical
backing, improved coexistence headroom, and at most 3% major-latency regression.
Reclaimable file cache, cgroup charge, startup peak and warmup peak are separate
required fields for both arms. RSS alone is not acceptable accounting. Use the
memory input shape in `test_qualification.memory_evidence`; it is synthetic.

Arithmetic checks do not validate the collection process. Final adoption still
requires frozen assets/settings, all valid cells, independent numeric and state
correctness, the real tokenizer, fresh-process alternating comparisons, thermal
and clock evidence, real memory accounting, a 60-minute P2 run, lifecycle tests,
coexistence results, and the relevant API compatibility review. Report upstream
update comparisons separately from optimization comparisons. Do not call these
offline tools a production benchmark or claim a measured speedup from them.
