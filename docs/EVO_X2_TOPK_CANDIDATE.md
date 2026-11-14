# Experimental EVO split top-k candidate

Status: **development candidate, disabled by default, not production qualified**.
The HIP implementation and Verifier adapter have not been compiled or run with HIP.
There is no speed claim. Keep the production profile on `register` until the missing
checks below pass. This change does not certify P2 split execution.

## Source and scope

Base: [Niko1221/Strata at 13eedf55](https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09).
The six-pass selection algorithm is adapted from the split section of
[nicejunjie/Strata-rdna3 qsa_select.cu at f557b544](https://github.com/nicejunjie/Strata-rdna3/blob/f557b5440fe345757cae69ede2ed34826da5e612/src/kernels/cuda/qsa_select.cu).
Both pinned repositories carry the MIT license, copyright 2026 Niko1221 and the
Strata contributors. The root LICENSE is retained. No full source file is replaced.

The adaptation keeps the existing `order_key` and weighted-cell rule: NaNs are
lowest, signed zeros compare equal, ties choose the lowest token IDs, and the
selected IDs are emitted in ascending ID order. It does not change the scorer,
QSA width, model, quantization, or sampling. CUDA paths are unchanged by design,
but no CUDA build was performed here.

The new HIP kernels have no mutable device-global scratch. Every call receives
histogram, state, and count pointers plus the **allocated** query/chunk strides.
The fixed kernel launch sequence is a memset followed by four histogram passes,
count, and emit. There is no cross-CTA barrier or spin loop. Launch failure is fatal
rather than falling back after part of the algorithm has run.

## Configuration and guards

- `STRATA_EVO_TOPK_MODE=register|split`, default `register`.
- `STRATA_EVO_TOPK_MIN_TOKENS=65536|98304|114688`, default `65536`.
- Unknown modes and thresholds fail Verifier initialization before allocation or capture.
  They do not silently fall back. Unsupported hardware/geometry may still fall back.
- Linux HIP, exact gfx1151 (HIP feature suffixes permitted), integrated, wave32,
  and one visible HIP device are required. An active `HSA_OVERRIDE_GFX_VERSION`
  disables it. No architecture spoofing is used.
- QSA index geometry is 128 dimensions, 4 heads, block size 4. Calls use FP32
  scores and int32 steps/IDs; pointer alignment, capacity and scratch are checked.
- `STRATA_TOPK_OLD` retains the original kernel and disables the candidate.
- Scratch supports 1–16 queries and at most 512 chunks of 2048 score blocks.
  Invalid or unsupported calls return before launch so the caller uses upstream.

For capacity Q and B score blocks, C=ceil(B/2048), logical scratch bytes are
`4 * Q * (1034 + 2 * C)`. Each histogram/state/count segment is aligned to 256
bytes. At Q=16, B=32770, logical bytes are 68352 and allocated bytes are 68608.
The Verifier reserves it in its own arena before graph capture. The existing
Verifier destructor synchronizes its streams, destroys graphs, then frees the
arena. This view is not part of session snapshots. Rebinding a session does not
restore a saved scratch pointer.

## Narrow Verifier integration

Each eligible Verifier owns a separate scratch allocation, including a Verifier
constructed for batching. This initial adapter uses it **only in non-batch
window graphs**. Batched-slot graphs still use upstream register selection.

`capture_all()` builds register and split window variants in warmup. At execution,
the candidate is selected only when **pos0 >= threshold**, T <= scratch Q, and
the matching split graph already exists for the current residency variant.
Missing or evicted split graphs fall back to register. Crossing the threshold
never allocates scratch or captures a new split graph. Normal upstream register
graph creation remains unchanged. No automatic call to warmup is added to the
server: a caller that never invokes `capture_all()` stays on register.

Diagnostics distinguish requested split configuration, scratch size, split-window
selection decisions, below-threshold decisions and missing warm graphs. These
are host window decisions, not timing evidence or actual GPU kernel counters.

## Verification performed offline

Host-only command:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
  tests/evo_x2/topk_contract_test.cpp -o /tmp/evo_topk_host
/tmp/evo_topk_host
python tests/evo_x2/test_topk_contract.py
```

The host test checks layout limits/overflow, real allocation strides, configuration,
capability rejection and threshold boundaries. It compares a weighted radix/chunk
model with an independent stable numeric-sort oracle in 425 cases, including
ties, signed zero, NaN/Inf, empty tails and partial chunks. It does not execute or
simulate HIP synchronization. The host test also passed AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection disabled. LeakSanitizer itself is
blocked by the executor's tracing environment, so no leak-sanitizer pass is claimed.

The new `qsa_evo_topk_parity` CMake/CTest target is registered only for Linux HIP.
It is synthetic and loads no model. It has not been compiled or run. On matching
hardware it compares exact output buffers against `qsa_block_topk_ref`, with two
separately allocated contexts/streams, captured graph replay, poisoned scratch,
guard bytes, non-finite scores, partial blocks and Q=1/2/16 under Q=16/C=17 storage.
A completion event fences input/scratch reuse. Wrong hardware returns CTest skip 77.

## Remaining work before adoption

1. Build the pinned baseline and candidate with the actual Linux HIP toolchain,
   CUDA disabled and `CMAKE_HIP_ARCHITECTURES=gfx1151`. Run the HIP parity target,
   existing top-k parity tests and device memory/race diagnostics. Build other
   touched backends as required by the repository before requesting upstream review.
2. Integrate both variants into the **batch-slot graph cache** with a variant-aware
   cache key and per-request actual-position decision. Warm every needed batch
   layout and window ahead of requests. Until then, P2 batch-mode speed is unchanged.
3. Add real engine tests for snapshot restore, cancellation and slot reuse after
   completion, graph eviction/residency transitions, both P2 slots and batching.
   Synthetic event-fenced replay does not prove server cancellation correctness.
4. Add runtime execution/fallback counters for the real batched path and verify
   the selected variant actually launches. Connect readiness to frozen
   qualification evidence; an environment setting is not qualification.
5. Measure thresholds 65536, 98304, 114688 and later positions on the target device.
   Require exact ID/order parity, at least 5% faster top-k and no end-to-end
   regression. Keep register if none qualifies. Do not infer results from another GPU.

No EVO access, model loads, production changes, service restarts, GitHub writes,
credential reads or private runtime data were used for this candidate.
