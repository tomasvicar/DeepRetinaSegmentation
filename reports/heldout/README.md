# Reviewed stage2 CPU verification

The exact isolated Git checkout used for the successful full run was detached at
`8c56594f36a92e569e901fcce0ed99ab2179629f`, tree
`a05f0ad2adc1de5d63393eebee7d6578b6cfcb52`. This is a real shallow checkout fetched
from the published branch, not a manually assigned Git identity.

- **687 passed**, 58 warnings, 26.16seconds; pytest exit0.
- Actual launch: Ada, nice10, CUDA hidden, OMP/MKL/OpenBLAS1 and initial Torch CPU
  threads1. Existing tests intentionally change Torch threads/backend settings;
  final threads2 and deterministic/warn-only flags true are explicitly retained
  in the receipt. This does not claim a single immutable in-memory runtime or a
  maximum one-thread guarantee for every test case.
- No CUDA initialization; no actual corpus VAL/TEST selection or evaluation.
  The full suite includes synthetic inputs and available original/bundled
  checkpoint-reference tests. Model weights are never committed by this proof.
- Tracked Git files remained clean. Actual package-source bytes and project
  metadata digest before/after remained
  `3779e1b5e9577d6870cf5e9a3ee48fb2271941dbbca12891f7a7f002075c4c9c`.
- [Exact receipt](stage2-fullsuite-proof.json) records installed dependency
  versions, runtime observations, source file list, UTC times, environment,
  commit/tree and SHA256 of the complete [test log](stage2-fullsuite-pytest.log).
  Log SHA256:
  `081d74250305dcd8ac75158e8c93bf03228c5e1116891f35fa69a5293c7bda23`.

The unchanged external runner is preserved as a reference at
[`run_8c56594_fullsuite_proof.py`](../../tests/reference/heldout/run_8c56594_fullsuite_proof.py).
Its hash matches the receipt. It is a record of this exact invocation with
historical scratch paths, not a new public command. The source checkout was
created with `git init`, `git fetch --depth=1 URL
refs/heads/experiments/reviewed-unified-heldout` and detached checkout of the
fetched actual commit. The runner invokes `pytest.main(['-q','-p',
'no:cacheprovider'])` using the frozen existing Python3.12.13/Torch2.6cu124 uv
environment; logs/proofs open exclusively. For another verification, choose fresh
scratch/log paths and record the new invocation hash instead of reusing this
receipt.

The first archived-tree attempt is not hidden: its log SHA and four failures
are retained in the receipt. Three HPO tests required a real Git HEAD absent in
a plain `git archive`. The new CLI test also correctly rejected a runtime mismatch:
earlier training tests had changed deterministic flags in the parent interpreter,
whereas the CLI started fresh defaults. The synthetic stage2 fixture now selects
and restores every relevant backend/dtype/thread flag, and an explicit test
continues to reject actual backend mismatches. Those fixture fixes do not relax
the production gate.

The output ownership review found and fixed `stat` following a foreign symlink
to the writer's still-open inode. Publication and rollback now use `lstat`.
A regression requires preserving a concurrently supplied `/proc/self/fd/...`
symlink rather than unlinking it as if it were our own output entry.

This proof is software verification, not model-quality evidence, held-out
blindness proof or hardware attestation. Freeze the final integrated source
before producing fresh actual VAL evidence and any selection record.
