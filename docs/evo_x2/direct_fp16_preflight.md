# Direct-FP16 expert preflight preparation

Status: **host-tested preparation only**. No GPU kernel is ported or enabled by this
change. It does not implement the complete direct64/pair128 feature, prove FP16
numerical quality, or qualify a production profile. Runtime prefill and its defaults
are unchanged. In particular, `STRATA_PF_F16` and `STRATA_PF_DM16` keep their existing
meanings, and `STRATA_EVO_PF_IQ_MODE` is not registered or read by the engine yet.
A profile resolver must reject direct64/pair128 as unavailable until integration and
qualification are complete. Do not pass an unimplemented environment variable to the
engine and interpret its unchanged upstream execution as a pair benchmark.

## What is implemented

`include/strata/prefill/evo_fp16_plan.hpp` is a pure host-side contract for the future
port. Its inputs and outputs have no mutable process-global state or GPU pointers.

- Strict `upstream`, `direct64`, `pair128` parsing; unset means upstream. Unknown values
  and direct-mode conflicts with explicit fused/native/MMQ disables are errors.
- Linux, HIP, integrated gfx1151, wave32, compiled geometry, actual layer qtypes,
  contiguous native GGUF row bytes, offsets, minimum blob extent and alignment checks.
- An initial explicit gate/up allowlist of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_XS; down Q2_0 or IQ4_NL. All other formats stay upstream. Q4_K/Q5_K and newer
  upstream qtypes are deliberately not claimed supported by this preparatory port.
- Actual chunk length of at least 1024, the existing stream threshold if higher,
  `ring > stage`, source, native layout, no peer, no Hadamard, no grouped scratch,
  and fused override checks. OR-ing capabilities does not bypass these outer guards.
- Checked direct X/H, FP32 Dm and grouping-table bytes, with the independently priced
  **full actual-chunk** upstream fallback retained per region. Each region is rounded
  to 256 bytes. An undersized fallback is rejected, not launched after buffer shrink.
- A per-expert tile partition: full 128-row pairs only; every remaining complete
  64-row tile and bounded partial tail uses the same direct-FP16 representation.
  There is no quad mode and no rounded-up pair that reads padding.
- Direct plans always set `dm16_used=false`. Kernel availability defaults to false.
  A synthetic test setting it true is not evidence that a kernel exists or is qualified.

The planner returns `direct_candidate`, `upstream`, or `reject`, a selected mode and
reason. A candidate is explicitly **unqualified**. It is not an execution permit or
lifetime proof. The future caller must derive facts from its actual layer/allocator,
not from the model's display name or user claims.

`Workspace.total` counts only the shared GU/H/Xq/Hq/Dm regions. Routing arrays,
expert ring, fallback weight dequantization scratch, graphs, other Prefill buffers,
KV, and the rest of the model are not included. The caller must add their actual
owned allocations to the memory plan without counting aliases twice. The caller
must price the chosen upstream fallback independently for the same actual token
count; the planner intentionally does not approximate all upstream plans.

## Remaining integration and validation

These are required together before the runtime mode can be exposed:

1. Port direct64, pair128 and conversion kernels into the pinned upstream geometry
   abstraction. Replace the reference's hardcoded 1280/2560 widths. Use explicit
   supported qtype switches, never a default branch that reinterprets an unknown type.
   Add all-lane WMMA/barrier rules and a proven bounded direct64 tail.
2. Replace the reference's device-global clip/maximum counters and callback globals
   with run/session-owned telemetry and a call context. Record saturation, nonfinite
   values and maximum magnitude, including H. Never let one session reset another's
   counters. Keep direct Dm and combine in FP32 regardless of upstream DM16 flags.
3. Wire the mode into configuration registration, help, effective settings, strict
   unknown/conflict errors, `native_supported`, `fused_ring`, `mmq_plan`, the actual
   `fused_nat` dispatch and per-layer logs. Preserve outer safety checks. Select the
   whole layer before converting X. Do not switch an FP16 pointer into the int8 path.
4. Wire exact requirements through `moe_bufs`, `moe_set_bytes`, `bytes_needed_impl`,
   allocation/carve/relayout, borrow/shrink and the final fallback capacity backstop.
   Ensure the same requirements apply at initialization and for shorter actual chunks.
   Prove scratch ownership, allocation before graph capture, no address changes during
   graph use, event completion before cancel reuse, and safe restore rebinding. The
   planner's booleans do not implement these lifetimes or prove P2 safety.
5. Compile the affected HIP translation units for gfx1151. Query actual compiled
   register/LDS/occupancy resources; do not use gfx1100 WGP constants or double the
   reported multiprocessor count. The reference pair's 36 KiB base LDS is not its full
   resource cost. No GPU toolchain or device was available for this change.
6. Compare direct64 with pair128 for every supported format/shape and sparse/tail
   case, then compare upstream with the new FP16 numerical mode. Collect teacher-forced
   full-vocabulary KL/top-1, clips/nonfinites, fixed Japanese quality cases, P2,
   cancellation/restore and sustained mixed-load measurements. Record selected modes
   and execution counts so all-upstream fallback cannot pass as pair performance.

No production, model, GPU, credentials or private runtime manifest is needed for the
host tests below. No measured speedup, memory savings beyond the exact logical X/H
formula, or qualification is claimed.

## Host tests

From the checkout root:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -Iinclude -Ithird_party/strata-ggml/ggml/include \
  -Ithird_party/strata-ggml/ggml/src \
  src/prefill/evo_fp16_plan_test.cpp -o /tmp/evo_fp16_plan_test
/tmp/evo_fp16_plan_test
```

The test derives qtype row extents from the vendored ggml structs, checks the known
8192-token X/H quantities and 52.5 MiB delta only for the matching geometry, tests
invalid inputs and capacity/overflow/threshold boundaries, and exhaustively verifies
row coverage for 0 through 9999 rows in both direct modes. Reusing a second input
cannot change the first plan. This is host policy isolation, **not a GPU P2 test**.

The target `evo_fp16_plan_test` is also registered in `cmake/backends/cpu.cmake`.
CMake/CTest configuration was not run here because the executable is unavailable.
Direct C++ compilation passed with warnings as errors for the default geometry and
the four existing alternate geometry headers (q4_big, q4_deep, q4_lite, q4_mid). UBSan and ASan+UBSan passed
with leak detection disabled; this sandbox's ptrace environment prevents LeakSanitizer
from starting, so no leak-sanitizer success is claimed. No HIP/CUDA/SYCL compilation,
kernel parity, model inference, performance benchmark or production qualification ran.

## Provenance and license

Base: [Niko1221/Strata at 13eedf552417a6a791a7d13c6597e24300ee3a09](https://github.com/Niko1221/Strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09).
The geometry, native format contract and grouping-table sizing come from this base.
Reference reviewed: [nicejunjie/Strata-rdna3 at f557b5440fe345757cae69ede2ed34826da5e612](https://github.com/nicejunjie/Strata-rdna3/tree/f557b5440fe345757cae69ede2ed34826da5e612),
especially `act_f16`, `native_h11_kernel`, `native_h11p_kernel`, `put_fast`,
`experts_native_f16` in `src/prefill/moe_fused_iq.cu`. The reference informs the planned
port but its kernel body, occupancy heuristic, global counters and callbacks are not
copied into this change.

Both pinned trees carry the MIT license, copyright (c) 2026 Niko1221 and the Strata
contributors. The root LICENSE is preserved unchanged. Keep that notice and the
reference attribution with any later substantial kernel port. Model weights and
third-party dependencies keep their own licenses.
