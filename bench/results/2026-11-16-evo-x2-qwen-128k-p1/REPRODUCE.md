# Reproduction and publication scope

This is a custom six-run P1 comparison, not the repository standard benchmark suite. It has two repetitions per condition, rather than the community guideline of at least three. No P2, quality-suite, or shipped-production performance result is claimed.

## Request and settings

`python make_prompt.py OUTPUT.json` recreates the exact synthetic request bytes without sending a request. Its SHA256 must match `f3e54c5a3e33068ee014d931c7ad59f7eb63d5af0427bc7b923f910d5a431cdc`. The prompt is 130,540 copies of `green`, separated by one space with no trailing space. With the frozen tokenizer/template and request flags, the server reported 130,552 input tokens. Do not substitute a different tokenizer/template and assume the same count.

The inert JSON files in `tooling/` record the matching argument/environment contract using generic container paths. They contain no host paths or credentials and cannot start a service by themselves. C0/C1 differ only in `STRATA_PF_FUSED`; B1 matches C1 while using the official sibling native binary. `environment.json`, `source-build-proof.json`, and `runs.json` preserve options, runtime/library fingerprints, source and binary hashes, and all measurements. Start a fresh process for each run, use order C0/C1/B1/B1/C1/C0, and verify zero cached tokens and one active request with two configured server slots. Loading/warm-up are outside the recorded P1 wall time.

The native source pins are C=`4d754ceba2d8bf54cbd1094e00352c27478f6198` and B=`3af562d541ec76c8da63d568248a919c8a54906b`. The C commit is preserved in the owner's private recovery bundle; this evidence commit does not publish that entire source history. Exact C replay requires that bundle and its documented public prerequisites. The measured backend-rule files are byte-identical between B/C; their hashes are recorded, not inferred from version labels. Do not claim full public reconstruction from the report alone.

## Read-only tooling checks

Run `python -B -m unittest discover -s tooling -p 'test_*.py'`. Importing the modules and running these tests does not access a GPU or change services. Three archive-dependent checks are explicitly skipped when the archived source trees/private launcher are absent. They passed in the sealed campaign; the public test run is reported separately. The three skip guards are publication packaging changes, not changes used to admit the benchmark.

`dispatch-evidence-minimal.patch` contains only the four collector/test diffs from the pre-correction campaign snapshot. `PATCH-BASES.json` records the original and reviewed hashes. It is a zero-context diff (`git apply --unidiff-zero`), not a patch against the Strata engine or against this repository root. The reviewed collector implementations are included for inspection. No maintenance, stop/start, watchdog, or supervisor code is published here.

## Retention and limits

Raw response bodies and private lifecycle data are excluded. Content hashes and token counts preserve output variance: both C0 outputs match, and all four fused outputs match, but C0 differs from fused. The first interrupted afternoon observation is excluded from the six-run estimand. Earlier reports remain in their original dated folders and must not be pooled with this source/configuration.

Restoration fields describe the completed experiment, not instructions to change the current service. Preserving this report does not authorize deployment, another benchmark, or cleanup. The native correction in this iteration was none: the correction was to the measurement classifier.
