# HIP split top-k: batch lifecycle wiring plan

Status: **design and CPU reference only**. No batched-slot native dispatch is added
by this follow-up. The batch path remains register, and the whole split candidate
remains off by default. No HIP compiler/device was available. The observations
below come from source inspection, not an observed GPU or production failure.

## Why a batch toggle is not a safe small change

At the pinned baseline and the first candidate:

- `include/strata/core/verify.hpp`: `batch_key` is `[hbase, rows...]`; `bkey` appends
  the doorbell/residency bit. The same base identity is used by `exec_bm_`,
  `commit_bm_`, `save_bm_`, `dcommit_bm_` and the LRU map.
- `Verifier::capture_batch` sets `sel_long_=0`. The three batch selection arms in
  `record_window` call `qsa_block_topk` directly: all-row attention, slot groups,
  and per-row/streamed attention. Changing only `idx_select` does not reach them.
- `Verifier::stage_batch` can instantiate and LRU-evict graph layouts while staging
  a request. Merely adding split capture here violates the planned warmup-only
  rule and can increase request-time peak memory.
- `batch_launch_rows` launches forward plus deferred-save graphs. `batch_commit_rows`
  finds the deferred-commit graph later. `run_slot_rows`, `batch_launch` and
  `commit_slot_prefixes` have additional forward/commit lookup sites.
- `evict_batch_graph` synchronizes the context stream and removes ordinary forward
  and commit entries; deferred save/commit maps have separate lifetimes. A new
  variant must not accidentally destroy a graph needed by an outstanding verdict.
- `src/program/generate.cpp` passes P2 session pointers to the stage's existing
  `Verifier::init_slots`. This is not evidence that P2 has two independent
  Verifier objects. Optional pipelined windows have a second Verifier. Scratch
  accounting must follow actual context objects, not multiply blindly by P2.
- The current `capture_all` covers non-batch window sizes, not row-layout variants.
  In `generate.cpp`, its observed caller is the pipelined-window preparation path.
  It does not prove a normal P2 server warmed any split graph.

A naive extension that puts the mode into global `bkey` changes commit/deferred
lookup identity as well as forward identity. Recomputing that identity after a
mode/residency change can select an absent or different bundle. This is a hazard
of the proposed extension, not a claim that the unchanged register path fails.

Pinned source:
[Verifier](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/core/verify.cpp),
[Verifier ownership and keys](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/verify.hpp),
[program construction](https://github.com/Niko1221/Strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/program/generate.cpp).

## Smallest coherent native delta to implement next

1. **Keep the existing base key and register maps unchanged.** Add a forward-only
   split map keyed by that same base key, or a separate forward-key type
   `(base_key, topk_variant)`. Do not add mode to `bkey`, commit, save or deferred-
   commit identities. Keep one ordinary commit graph per base layout unless an
   actual kernel dependency requires another one.
2. **Add an explicit idle warmup entry point.** After `init_slots` and before
   requests are admitted, enumerate the permitted P2 row layouts, MTP group widths,
   hbase values and residency variants. Capture the register and split forward
   variants plus their required commit/save graphs while no context is in flight.
   Count both forward variants in graph memory. Publish a split-ready entry only
   after instantiation succeeds. A failed optional split warmup may leave the
   known-good register variant available, with a recorded reason.
3. **Choose once and store a launch ticket.** After the existing row validation and
   `refresh_ar`, form the base key from the actual rows/hbase/residency. For this
   initial all-or-nothing batch design, choose split only when every slot group's
   first actual position is at or beyond the configured threshold, all calls fit
   Q/C bounds, and that exact split graph is already warm. A batch mixing short and
   long slots stays register. A group whose final draft crosses the threshold but
   whose first row is below it stays register. Never choose from max_context or
   the largest position in the batch.
4. **Route all three batch top-k arms through a local selector.** It receives the
   ticket's captured variant, actual query count, score/step/ID row offsets and this
   Verifier's already-owned scratch. Keep scoring, QSA widths and attention exactly
   as before. The all-row and grouped arms can reuse the same scratch sequentially
   on `cs_`; per-row calls must each reset it. Do not share it with another Verifier.
5. **Use the stored forward choice at every launch site.** Update `run_slot_rows`,
   `batch_launch` and `batch_launch_rows`. Commit/save calls retain the ticket's
   immutable base identity. Do not recompute a commit key from a later request's
   mutable mode or residency. Deferred launch groups need their own tickets rather
   than one overwritten `last_variant` field.
6. **Protect ownership across every API.** Check the same context's single-window,
   pipelined and batch in-flight state before new capture, storage replacement or
   reuse; `b_running_` alone is not a universal fence. Cancellation marks output
   discard but does not free scratch or a slot. Reuse follows successful completion
   of the correct context's stream/event generation. An old event or another
   context's event cannot prove completion. Failed/uncertain completion quarantines
   the context instead of treating cancellation as success.
7. **Evict variants as one layout budget entry.** Synchronize the owning stream,
   protect current launch and outstanding verdict tickets, then erase both forward
   variants and the correct ordinary ancillary entries. Preserve deferred save/
   commit entries still required by their own tickets. A missing split graph after
   eviction returns to register; it is not captured at threshold crossing. The
   default register-only LRU behavior must remain unchanged.
8. **Keep scratch out of snapshots.** Restoring a session retains the destination
   context's live allocation. If context storage actually changes, complete work,
   destroy every graph referring to the old address, advance its generation and
   warm again. A session snapshot never imports an old pointer or native graph.

That delta crosses graph construction, cache lookup, three selector arms, three
launch APIs, verdict lifetime, eviction and startup. It is deliberately not
landed as an uncompiled toggle in this offline pass.

## Executable CPU lifecycle reference

`tests/evo_x2/topk_batch_lifecycle_test.cpp` is explicitly a test-only model. It uses
real `batch_rows_error`, the candidate's real config/threshold parser, and real
scratch-size calculation. It does not replace native graph code or provide a
runtime safety guarantee. Run with:

```sh
python tests/evo_x2/test_topk_contract.py
```

The test demonstrates the naive forward-key collision and verifies a separate
forward namespace, shared commit identity, all three threshold boundaries,
contiguous MTP groups, mixed-length fallback, failed/missing warm graphs, no
request-time graph creation, two isolated context allocations, cancellation before
completion, wrong-context and stale completion events, stale verdict tickets,
pending-verdict slot overlap, eviction protection, destination-owned restore,
generation invalidation and failed-completion quarantine. It passed the host
compiler with warnings as errors and ASan/UBSan (leak detection disabled due to
the executor's ptrace limitation).

Before native acceptance, run the existing and new HIP parity targets and add
real P2 mixed-length, changing-layout, cancellation, restore, deferred verdict,
residency transition and eviction tests. Include trace evidence that both P2 slots
execute the chosen variant. These tests and the 5% top-k/no-end-to-end-regression
performance gate are still **not run**. Host reference success does not qualify
or enable the native candidate.
