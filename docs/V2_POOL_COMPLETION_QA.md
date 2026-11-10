# Typed completion QA from trusted V2 pool evidence

`scripts/adapt_pool_completion_qa.py` bridges the V2 pool receipt schema to the
existing `compare_completed_val.py` completion contract. The comparison gate is
unchanged. The adapter creates a new typed metadata document only after validating
an actual E60 receipt, its exact final task, the bound completed coordinator decision,
and the independently copied read-only milestone metadata. It does not produce
training evidence or turn a partial receipt into a completed run.

No actual E60 completion is claimed by this change. The positive tests use synthetic
metadata with no checkpoint files. A stripped actual E3 receipt and its recorded
4573/5700 contributing-annotation count are retained as a rejection fixture.

## Inputs and trust

Supply SHA256 values from independently reviewed original receipt/task/decision
artifacts. These are hashes of their original file bytes, including whitespace.
Computing a new hash for an unreviewed input does not authenticate that input.
Separately, the coordinator decision must bind the canonical JSON receipt digest;
this digest can differ from the original pretty-printed receipt file hash.

```bash
PYTHONPATH=src python scripts/adapt_pool_completion_qa.py \
  --receipt /reviewed/e060-receipt.json \
  --trusted-receipt-sha256 ORIGINAL_RECEIPT_BYTES_SHA256 \
  --task /reviewed/e060-task.json \
  --trusted-task-sha256 ORIGINAL_TASK_BYTES_SHA256 \
  --decision /reviewed/e060-decision.json \
  --trusted-decision-sha256 ORIGINAL_DECISION_BYTES_SHA256 \
  --manifest /verified/corpus-v2/manifest.json \
  --milestone /original/readonly/pool-milestone \
  --output /new/completion-qa.json
```

The supplied manifest must have the fixed reviewed manifest SHA
`17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`.
There is no production override for this pin. Only JSON metadata is read; image,
mask, FOV, checkpoint and cache-payload files are never opened. The milestone path
must match the original receipt; metadata files and directory must be read-only,
real paths, and sidecar bytes must match the original independent-copy hash receipt.
Outputs use exclusive creation and cannot overwrite a prior report.

## Required evidence

- Receipt and checkpoint QA both record actual E60; budget is completed, finite
  weights/optimizer checks passed, and the final run returned0 without TEST evaluation.
- The coordinator decision is `finished`/`COMPLETE` and binds this exact receipt,
  trial, final task and lease. The selected best epoch/score matches all60 contiguous
  CSV rows, the terminal summary and best VAL sidecar.
- Plan, config, source6b3dfad, execution/training hashes, corpus00d9 fingerprint,
  manifest, embedded geometry policy and worker/GPU identity match. Cached runs use
  the verified18c3 cache with canonical fallback disabled; canonical workers use no cache.
- The exact trusted-manifest TRAIN photo IDs match all2529 actual photo-visit keys,
  each with at least60 visits, delivering60×8192 samples. Source coverage matches
  the21 declared source inventories. DRIVE/RITE share16 photos, so source-local
  photo totals are2545 while unique TRAIN photographs remain2529.
- **Every one of the5700 eligible TRAIN annotation record IDs** must appear in the
  actual annotation-visit map with a positive count. Eligible and contributing
  counts must both equal5700. Same-count foreign IDs, missing IDs and zero counts
  fail. Annotation-visit totals are recorded separately; no equality or bound to
  delivered samples is imposed because one sample can merge annotations.

The typed result retains the complete original receipt text and its original-byte
SHA, original task/decision provenance, immutable milestone proof, sidecar hashes,
and exact eligible annotation/photo ID digests. It maps the trusted pool finite
attestations into `finite_best_last_weights_and_last_optimizer`; it does not rerun
checkpoint tensor or optimizer inspection. Keep the original reviewed pool QA and
milestone alongside this metadata bridge.

Fresh VAL production still loads the checkpoint with its original frozen training
source6b3dfad. This adapter runs separately and changes neither that source nor any
checkpoint, data, study, worker, default, comparison policy or TEST gate.

Verification:49 CPU tests pass, including unchanged comparison tests, synthetic
consistent E60 acceptance, cached/canonical identity, original-byte pins, actual E3
rejection, source/policy/budget mismatches, changed/writable milestone metadata,
partial annotation coverage and same-count wrong IDs. Ruff passes. Evidence is in
`reports/v2-completion-qa-adapter/`.
