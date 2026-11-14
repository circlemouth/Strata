# Direct-FP16 typed workspace preparation

Status: **host-tested interface scaffold, no runtime caller**. This builds on
`evo_fp16_plan.hpp`; it does not enable direct64 or pair128, implement their kernels,
allocate GPU memory, prove GPU ordering, or qualify a production profile.

## Concrete upstream compatibility

At pinned upstream `13eedf552417a6a791a7d13c6597e24300ee3a09`,
`src/prefill/prefill.cpp` has these contracts:

- `Alloc::take<T>(n)` prices each allocation as
  `((uint64_t(n) * sizeof(T) + 256 + 255) & ~255ull)`. The additional 256-byte guard
  exists even for a zero-byte take. Count-only allocation can add a further page
  granule; that is a separate owned-allocation accounting step.
- `carve` binds `m.GU`, `m.H`, `m.Xq`, `m.Hq` and `m.Dm` within the shared prefill
  region. Other tensors occur between those takes. GDN/QSA/HC scratch can alias the
  region at different phases. A new contiguous five-buffer layout must not be
  silently substituted for that real ordering.
- The fused-native calls take `m.GU` as grouping tables, `m.Xq` as int8 activations,
  `m.H` as int8 intermediate storage and `m.Dm` as down outputs. Their `void*`/float
  pointer types do not identify the representation. Direct FP16 needs a deliberate
  typed adapter, not a cast of an existing `experts_native` call.
- `m.cs` orders compute, while copy streams and ring events manage expert blob
  availability. A workspace lifetime event must cover the final combine and any
  telemetry copy on the compute stream, not only an expert batch or ring copy.

`allocation_take_bytes` mirrors the guard-and-alignment formula with overflow
checks. `allocation_bytes` sums only these five takes. It does not price other
prefill tensors, the containing region's own guarded take, allocation page granules,
ring, graphs or telemetry. The earlier planner's `Workspace.total` is the aligned
payload envelope; it is not the upstream allocator's final physical allocation size.

`Binding` receives the **actual five spans from the allocator**, in any address
order. It verifies required payload capacities, 256-byte address alignment, null
nonempty spans, address wrap and every pairwise overlap. Its capacity excludes the
extra allocator guard. It does not dereference or free a device address. Aliasing
with tensors outside these five regions still requires integration-side ordering.

## Typed views and lifetime hooks

`WorkspaceGuard` is a noncopyable, nonmovable, host-thread-confined member for the
future execution context. It has no mutable global state. One guard is needed per
independently executing prefill context. Passing a context-owned flag is a required
contract assertion, not proof of an allocator's ownership.

1. Before capture, allocate using the real allocator and create a backend completion
   event. `bind(input, binding, stream)` reruns preflight using the binding's actual
   capacities. It refuses upstream/fallback/error plans and unavailable kernels.
2. Retain one graph reference before capturing any graph that embeds these addresses.
   Retained references block rebinding, dropping and owner release.
3. `acquire` locks the layer representation before returning distinct FP16 X/H,
   FP32 Dm and grouping-table view types. Views expose only direct payload extents,
   not larger fallback capacities. There is no direct-to-upstream fallback operation
   after acquisition. Do not cache a view beyond its acquired layer.
4. Queue conversion, direct kernels, FP32 combine and telemetry using the bound
   compute stream. The kernel wrappers and telemetry ABI are not implemented here.
5. `retire(fence)` invokes the supplied backend record callback on the bound stream.
   A failed record leaves the guard acquired and blocks reuse. The event must already
   exist; the callback must not allocate. The intended HIP adapter maps record to
   `hipEventRecord` and maps query success/not-ready/other errors to
   complete/pending/error. That adapter must be compiled and tested separately.
6. `poll` permits reuse only after the query reports complete. Query errors poison
   the guard, with no reset method that could assume the GPU has stopped. Cancellation
   uses this same event retirement; cancel requested does not mean memory is reusable.
7. After replay completes and every graph reference is released, `drop` releases only
   host bookkeeping. `can_release_owner` tells the integration when ordinary freeing
   or resizing is permitted. The class does not own allocations, synchronize in its
   destructor, or enforce an external owner's behavior. Failed contexts require
   backend error handling and proven quiescence before destruction/free.

Do not serialize this guard, bindings, views or fences in a conversation snapshot.
Restore must bind the destination context's own allocated addresses. Keep event and
callback state alive until completion. The host tests exercise these rules with
synthetic addresses and events; they do not demonstrate GPU P2/cancel/restore safety.

The conservative guard waits for completion before another acquisition. A future
integration that permits same-stream reuse without a host poll needs a separate
stream-ordering proof and corresponding tests. No throughput claim is made here.

## Exact remaining interfaces

- No direct64/pair128 entry point exists in this branch. The existing native entry
  points require int8 X/H. Do not connect typed views to them.
- The source and lifecycle of the complete MoE-region owner must be wired through
  init/carve, relayout, borrowing, shrink, destruction and snapshot restoration. A
  guard object alone does not implement or audit those owners.
- The concrete HIP event adapter and graph capture/replay lifecycle are absent.
  The adapter must return errors honestly, record after the last queued use, and
  avoid allocation or unexpected synchronization during capture.
- Context-owned GPU saturation/nonfinite/maximum counters, their reset/collection
  kernels, and a typed telemetry hook are still undefined. No ABI is guessed here.
- Runtime mode registration, dispatch, actual execution counters, build/resource
  checks and all numerical/performance qualification gates remain outstanding.

The parent planner documentation lists the rest of the complete port requirements.
All direct modes must remain unavailable in the profile resolver until those gates
are satisfied. No production source or existing environment-variable behavior changes.

## Verification

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -Iinclude -Ithird_party/strata-ggml/ggml/include \
  src/prefill/evo_fp16_workspace_test.cpp -o /tmp/evo_fp16_workspace_test
/tmp/evo_fp16_workspace_test
```

The host test covers the literal upstream allocation formula, zero-byte guards,
size/address overflow, every region-pair alias, typed representations, actual
capacity overriding an optimistic plan, graph retention, failed event record,
bound-stream forwarding, cancel/pending reuse refusal, separate-context completion,
destination rebinding and sticky event errors. Default plus all four alternate
compiled geometries pass with warnings as errors. ASan+UBSan (leak detection disabled
because of sandbox ptrace) and standalone UBSan pass. The original planner test also
still passes. CMake/CTest, HIP compilation, GPU kernel execution, live models, numerical
parity and performance tests did not run; this environment has no CMake or HIP compiler.

## Attribution

The memory and event interfaces were inspected in the pinned
[Niko1221/Strata prefill source](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/prefill/prefill.cpp).
The intended direct-FP16 port remains based on the separately reviewed
[nicejunjie/Strata-rdna3 reference](https://github.com/nicejunjie/Strata-rdna3/blob/f557b5440fe345757cae69ede2ed34826da5e612/src/prefill/moe_fused_iq.cu).
Both pinned trees carry MIT, copyright (c) 2026 Niko1221 and the Strata contributors.
The root license and production source remain unchanged. No reference kernel,
global counter, callback global or occupancy constant is copied in this scaffold.
