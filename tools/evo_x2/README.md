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
  `STRATA_PF_FUSED_NATIVE`, `STRATA_PREFILL_MMQ`. Only `0`/`1` values are accepted.
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
