# EVO inventory and profile review

These opt-in tools help prepare an isolated Linux HIP/gfx1151 experiment. They do
not change Strata defaults, launch a process, install configuration, or qualify a
release. Python 3.10 or later is sufficient; no extra runtime package is needed.

## Capture only selected inputs

Copy `examples/selection.json` outside the repository and replace its placeholder
paths with the exact files selected for the experiment. List every shard and
relevant pack, tokenizer, template, projector and draft asset separately. Nothing
is discovered by directory walking, filenames, model names or revision labels.

```sh
python tools/evo_x2/capture_manifest.py --selection /private/selection.json \
  --output /private/runtime-manifest.json
```

The selection is validated against `schemas/selection.schema.json`. Unknown keys
and roles fail. Assets must be regular files; hashes cover full contents. A file
that changes during hashing fails capture. The collector does not modify assets.
It opens only explicitly selected files and, when requested, the selected PID's
`stat`, executable and allowlisted environment fields. A PID change during capture
fails. It never collects command lines, arbitrary logs, prompts, filesystem trees
or other processes. Hostname, kernel and machine metadata are opt-in via
`host_metadata: true`.

Optional selection fields:

- `pid`: a positive Linux PID chosen by the operator. There is no PID search.
- `environment`: selected names from `STRATA_PF_FUSED`,
  `STRATA_PF_FUSED_NATIVE`, `STRATA_PREFILL_MMQ`, `STRATA_PF_F16`,
  `STRATA_PF_DM16`, `STRATA_PF_FUSED_TRACE`, `STRATA_PREFILL_TIMING` and
  `STRATA_HIPBLASLT_VERBOSE`. Only literal `0`/`1` values are accepted. The last
  two flags are enabled by **presence**, including `0`; unset them to disable
  diagnostics. Inventory records values, not their effective meaning or execution.
- `config`: an object containing `path` to a JSON configuration and `fields` to
  retain. Supported fields are `ctx`, `parallel`, `fit_max_tokens`, `temperature`,
  `top_p`, `top_k`, `presence_penalty`. Each is type checked; all other content is
  omitted. Known credential fields contribute only an authentication-presence
  marker. Nested/free-form content and credential values are never copied.
- `service`: an object containing `scope` (`system` or `user`), `name` (a unit
  name ending in `.service`) and explicit `unit_files` paths, including selected
  drop-ins. Only paths, byte counts and hashes are recorded, not unit contents.
  This records the operator's selection; it does not verify systemd/PID binding.

The output follows `schemas/manifest.schema.json`, is created with mode 0600,
and cannot overwrite any existing file or symlink. JSON errors are sanitized.
The output still contains private paths, hashes and optional machine identity.
Keep outputs and selections outside this public repository; inspect before sharing.

### Explicit inventory limits

`inventory_only` is not a complete runtime freeze. Asset selection does not prove
which files the engine loaded. Config selection is not effective-setting discovery.
This version does not inspect tensor geometry, GPU properties, toolchain versions,
container mounts/digests, CPU topology, hierarchical cgroup limits, memory peaks,
residency coverage or co-resident services. These remain explicit unverified items.
It does not read arbitrary process environment or service contents to fill gaps.
Do not use a partial inventory as evidence of memory safety or qualification.

## Resolve a review plan

The shipped profile is a generic baseline review template, not a production
profile. It names one layout and one kernel mode instead of search candidates.

```sh
python tools/evo_x2/resolve_profile.py \
  --profile tools/evo_x2/examples/baseline-review.profile.json \
  --manifest /private/runtime-manifest.json --review-only \
  --output /private/review-plan.json
```

This checks closed schemas and requires selected model and tokenizer assets. It
emits an explicitly **non-deployable** baseline plan with input-document hashes,
the proposed `evo_contract: true`, `fit_max_tokens: false`, `parallel: 2` server
settings, required engine context/slot observations, and blocking checks.
The server API contract must be implemented in the same fork before those proposed
settings are usable. No launch command or engine configuration is generated.

Without `--review-only`, resolution fails: there is no qualified deployment path
in this tooling version. Arbitrary `passed` or result-ID fields cannot bypass this
gate. `direct64`, `pair128`, split top-k, optimized expert layouts and purported
allocation guards are rejected, not silently downgraded. Upstream modes do not
prove quality, performance, P2 progress, memory headroom or all-expert residency.
Inherited settings remain unresolved blocking work, never implied READY evidence.

The profile schema deliberately implements only this bounded baseline-review
subset. A broader tuning specification is not a deployable profile and must not
be silently accepted. Extend schemas, implementation and qualification together.
The dependency-free validator rejects schema keywords it does not implement.

## Offline checks

```sh
python -m unittest discover -s tests/evo_x2 -v
```

Tests use synthetic temporary assets and a synthetic proc tree. They never probe
a running service or GPU. When `jsonschema` is already installed, tests also
cross-check emitted documents against the standard schemas; otherwise that one
cross-check is skipped. These tests establish tooling behavior only, not hardware
correctness or performance.

## Isolated Linux gfx1151 fused-int8 experiment

`examples/fused-int8.experiment.json` corrects an experiment setup omission:
Linux gfx1151 does not receive the Windows fused prompt defaults. It explicitly
selects the existing native fused path. This is an **unqualified experiment**,
not a production profile or a claim of faster execution. The review-only resolver
still rejects unsupported direct-FP16 modes and does not install this recipe.

The two arms differ only in `STRATA_PF_FUSED=0` versus `1`. Both keep
`STRATA_PF_FUSED_NATIVE=1`, `STRATA_PF_F16=0` and `STRATA_PF_DM16=0`. Start a fresh
process for each arm because selectors are cached. Keep the binary, server,
model assets, hipBLASLt table, context, slots, request bytes and resource limits
identical. Establish the same-upstream B/C comparison before attributing a
change to fork code. Do not add scheduler changes to this experiment.

Static inspection of the 2026-11-15 Qwen3.8-Flash-Next native pack found these
pairs. All are covered by the existing gfx1151 fused-int8 implementation:

| Gate/up qtype | Down qtype | Layers |
| --- | --- | ---: |
| IQ2_S (22) | IQ4_NL (20) | 23 |
| IQ2_S (22) | Q2_0 (42) | 3 |
| IQ3_XXS (18) | IQ4_NL (20) | 21 |
| IQ4_XS (23) | IQ4_NL (20) | 1 |

Coverage alone does not prove dispatch. Stream-all eligibility, compiled GPU
kernels, allocation checks and layer conditions still apply. Keep all fallback
and undersized-buffer guards enabled. This path stages activations as int8;
it does not implement `direct64` or `pair128` and does not change expert residency.

For one diagnostic run, use the recipe's diagnostic environment. The HIP-only
`STRATA_PF_FUSED_TRACE=1` log records the native WMMA branch after launch error
checking: qtype IDs, gate/up and down routes, output dtype and tile rows. It prints
once per thread/device-slot/type/route, including the first call's routed-row
count. It does not synchronize the GPU, prove completion, cover non-native paths,
or log prompts. The device slot is the existing 32-entry dispatch-cache slot,
not an independent PCI identity. A log's absence is not proof of a particular
fallback. Disabled tracing avoids device queries and formatting; enabled tracing
has host bookkeeping overhead. Use the same instrumentation in comparison arms,
then confirm performance without diagnostics.

Keep `STRATA_HIPBLASLT_VERBOSE=1` output with the exact table hash, selected
solutions, fallback shapes and runtime library version. A selected tuning file
alone does not prove every GEMM used it. `STRATA_PREFILL_TIMING=1` reports GPU
**timeline** and wall intervals, not pure kernel duration under P2 scheduling.
Unset both variables for a non-diagnostic run; setting either to `0` is still on.

Before a performance claim, run the four matching `hip_prefill_fused_iq`
`--only='GATE / DOWN' --edge --no-timing` checks in an isolated GPU window. Keep
F16 and DM16 off. These kernel checks do not establish model quality. Then compare
identical cold 130,552-input-token P1/P2 requests on a two-slot, 131,072-context
server with a 512-output cap and an enforced 900-second wall deadline. Record
natural EOS/output lengths rather than calling them quality equivalence. Keep
admission wait, admitted-to-reading wait, prefill, decode and client TTFT separate.
Record cgroup CPU throttling counters before/after, memory events and available
pressure counters. Do not infer a source-code cause from those counters alone.
Retain individual runs and failures. Restore the prior service and verify it;
production adoption needs separate approval.
