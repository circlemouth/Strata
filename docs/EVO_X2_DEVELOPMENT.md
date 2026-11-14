# EVO-X2 development branch

This branch is a qualification candidate, not a production release. It preserves
the upstream default serving path. No speed, memory reduction, numerical parity,
or hardware qualification is claimed by passing the offline tests.

## Source provenance

- Base: [Niko1221/Strata](https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09),
  commit `13eedf552417a6a791a7d13c6597e24300ee3a09`.
- Selective kernel reference: [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3/tree/f557b5440fe345757cae69ede2ed34826da5e612),
  commit `f557b5440fe345757cae69ede2ed34826da5e612`.
- Both source trees carry the MIT license. Keep the root `LICENSE` and source
  attribution. Model assets and bundled third-party components have their own
  licenses. No model assets or private runtime configuration are included here.

## Strict API candidate

Set `evo_contract: true`, `fit_max_tokens: false`, and `parallel: 2` in an isolated
server configuration. This changes API behavior and needs caller compatibility
testing before deployment. It does not change model, tokenizer, aliases, sampling,
thinking, vision or KV settings. Do not turn it on in an existing service merely
to try the tests.

Chat Completions, Responses and Anthropic Messages use the same output rule:
omitted or null means 16384; integers 1 through 16384 are accepted. Booleans,
strings, floats, zero, negative numbers and larger values are rejected with 400.
Non-null `max_tokens`, `max_completion_tokens` and `max_output_tokens` aliases
must agree. Shared web settings cannot silently replace this contract's default.

After final template and image-token encoding, input + output + 8 must fit 131072.
Neither the input nor output allowance is shortened. Startup and request paths
require observed engine context 131072, engine batch 2, and INFO batch_slots 2.
A later observed downgrade makes health and generation unavailable. `/props`
reports the contract and says explicitly that concurrent progress is unverified.
These slot checks are not proof of real concurrent execution.

The default server behavior without the opt-in remains upstream's behavior.
The existing API keys and access controls still apply; this option grants no
additional access.

## Offline memory planner

`EvoMemoryPlan` is a thread-safe byte-accounting primitive with independent host,
GPU-addressable and ancestor-cgroup limits. Every backing has an identity, owner,
purpose, lifetime and explicit charges. An explicit alias counts the backing
once. Pending work reserves memory before commit; cancellation releases only the
last reference. Unknown demand, negative budgets, arithmetic overflow and missing
ancestor charges fail closed. Zero-byte CPU complements are valid.

Host reserve is the next 256 MiB multiple of max(12 GiB, 10% of total RAM).
Each finite cgroup independently keeps max(1 GiB, 5% of its limit), also rounded
up. Cgroup reserves are not subtracted again from the host budget. An optional
headroom sample denies additional allocations before they consume reserve.
Five seconds classifies sustained pressure; it is not an admission grace period.

This library is **not yet wired into all engine allocations or request admission**.
It cannot enforce a runtime memory cap by itself. Do not advertise it as an active
allocation guard. Full integration requires measured allocation ownership and
startup/steady-state peaks for every target session, draft, KV, prefill, graph,
snapshot, cache, runtime and co-resident workload.

## Build and verification

Use a separate checkout, build directory, configuration home and Python environment.
Do not run generic source setup against an existing user's Strata configuration.

```sh
python -m pytest serve/test_evo_contract.py serve/test_context_overflow.py \
  serve/test_runconfig.py serve/test_responses.py tests/evo_x2 -q
cmake --preset cpu-check
cmake --build --preset cpu-check -j 4
ctest --preset cpu-check
```

A future HIP build must use the existing approved toolchain and explicitly set
HIP on, CUDA off, and `CMAKE_HIP_ARCHITECTURES=gfx1151`. Never spoof gfx1100 or
change a host driver, BIOS, power limit or toolchain as part of a code comparison.
CPU tests do not compile or execute HIP kernels. Follow the individual candidate
documentation for unperformed checks.

## Deployment remains blocked

The [inventory and profile tools](../tools/evo_x2/README.md) emit only a partial
inventory and a non-deployable review plan. No benchmark report, release image,
qualified profile, service unit or cutover automation is shipped by this stage.
Keep private manifests, paths, hostnames, credentials and run logs outside the
repository.

Before release, freeze the actual current runtime and model assets, validate
128K/P2 with simultaneous progress, verify all allocations and expert coverage,
run numerical and Japanese functional gates, then compare current/base/candidate
separately on the same hardware. Record cold/warm and P1/P2 independently, including
failed cells, mixed traffic, cancellation, cache reuse, co-resident workloads and
the sustained run. An early EOS, truncated request or degraded slot count cannot
be counted as a performance success.

Keep production unchanged until a qualified immutable release and a verified
rollback path exist. A comparison that would unload production or compete for
its model RAM must be separately authorized and scheduled. Never double-load
baseline and candidate, terminate other services or drop system caches to make
a candidate fit.
