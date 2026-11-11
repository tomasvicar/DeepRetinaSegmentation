# Fixed terminal60 VAL baseline, distinct from best checkpoint selection

The actual completed V2 control used all2529 TRAIN photos and all5700 eligible
annotation records by terminal60. Its VAL-selected best checkpoint was epoch52
and had5699 contributed annotations; one unique RIGA observer annotation had not
yet contributed. These two checkpoints must not be conflated. Neither a full-run
completion receipt nor a duplicated/photo-alias argument changes the earlier
checkpoint's actual coverage. No ground truth or model was opened by this change.

`scripts/compare_terminal60_val.py` is an explicit opt-in metadata-only wrapper
for the **fixed terminal60 all-annotation baseline**. The original best-checkpoint
comparison remains the default. `compare_completed_val.py`, the strict6b pool
completion adapter, selection gate, TEST evaluator and default prediction paths
are unchanged. The new route never renames a `last.pt` hash to `best.pt` or edits
original QA/report bytes. It cannot authorize the existing final-selection route.

Required terminal evidence:

- fresh enriched VAL evidence for actual epoch60, full steps and
  `checkpoint.sha256 == completion_qa.checkpoint_sha256['last.pt']`;
- original typed `trusted_pool_completion_qa` with terminal60, finite best/last
  weights and last optimizer, source6b/executioncb5/training9075, matching
  canonical manifest/content and original-reference geometry policy;
- all2529 TRAIN photos, minimum60 visits,60×8192 delivered samples; all5700
  eligible manifest annotation IDs contributed at least once, with exact sorted
  annotation/photo ID hashes from the immutable manifest (original adapter uses
  compact sorted JSON plus newline). Actual minimum annotation visits is2;
  annotator variants are not incorrectly required to have60 visits each;
- unchanged full-data60 config/hash, actual runtime/source identities and their
  OWN externally pinned contexts; exactly770 paired VAL photos, annotation/source
  attribution and GT/count support checked by the existing common comparator.

The candidate context uses the existing `completed_val_unified_context` schema
and MUST additionally declare the top-level field:

```json
"checkpoint_role": "fixed_terminal60"
```

The external SHA binds this role together with its original fresh report SHA,
completion-QA SHA and all existing `bindings` objects. Do not relabel historical
best52 evidence as fresh terminal60 evidence. Root must produce fresh last60 VAL
and review/pin its own context. The original reference context remains its own
unchanged externally pinned artifact; source identities across pipelines may
differ. This comparison does not mint new VAL evidence or selection receipts.

```bash
uv run python scripts/compare_terminal60_val.py \
  --reference original-validation.json \
  --reference-context reviewed-original-context.json \
  --trusted-reference-context-sha256 "$ORIGINAL_CONTEXT_SHA" \
  --candidate fresh-terminal60-validation.json \
  --candidate-context reviewed-terminal60-context.json \
  --trusted-candidate-context-sha256 "$TERMINAL60_CONTEXT_SHA" \
  --completion-qa unchanged-completed60-qa.json \
  --manifest /path/to/canonical/manifest.json \
  --output fixed-terminal60-val-comparison.json
```

The result has a distinct
`kind=completed_val_fixed_terminal60_frozen_original_comparison`,
`checkpoint_role=fixed_terminal60`, `baseline_only=true`,
`selection_authorized=false` and `test_evaluation_enabled=false`. Original
`best.pt`/`last.pt` hashes and original best epoch52 are preserved separately in
`original_completion_checkpoint_sha256` and `original_completion_best_epoch`.
The candidate checkpoint remains the actual last60 identity. Fixed terminal60 is
a release-baseline policy, not a claim that its VAL score exceeds best52.

[The existing frozen VAL comparison contract](COMPLETED_FROZEN_VAL_COMPARISON.md)
still controls all other gates, byte snapshots and exclusive output behavior.
Synthetic2529-photo/5700-annotation/770-VAL fixtures test acceptance, best52
rejection, wrong last hash, missing/wrong role, partial annotations, source,
geometry and steps, plus unchanged default rejection. No actual trained model,
GT, training or TEST evaluation was performed during implementation.

Validation:80 targeted metadata tests pass in8.93s (CPU/thread1, CUDA hidden),
including the unchanged default/older comparison gates. Ruff check/format pass.
