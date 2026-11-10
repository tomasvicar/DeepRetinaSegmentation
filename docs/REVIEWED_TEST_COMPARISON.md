# Compare reviewed original and unified TEST reports

`scripts/compare_reviewed_test.py` compares saved TEST count reports after both
pipelines have completed their separately reviewed execution routes. It reads JSON
metadata and counts only. It does not execute a model, load weights, open GT arrays,
authorize TEST evaluation, change a threshold or select a model.

The original producer is `scripts/evaluate_original_frozen.py` from commit
`be72a953bc502e5a032c99a426009b0ac4aceb0e`. Its original policy is frozen from VAL
and explicitly names the selected unified selection SHA. The unified input is the
whole `unified_reviewed_heldout_evaluation` wrapper emitted by `heldout.py` after
its existing stage1 selection gate and explicit stage2 execution. Neither producer
nor existing VAL/TEST gate is changed by this helper.

```bash
uv run python scripts/compare_reviewed_test.py \
  --original-report /reviewed/original-test.json \
  --original-policy /reviewed/original-policy.json \
  --trusted-original-policy-sha256 EXTERNALLY_REVIEWED_POLICY_FILE_SHA256 \
  --candidate-report /reviewed/unified-stage2-test.json \
  --selection /reviewed/actual-unified-selection.json \
  --trusted-selection-sha256 EXTERNALLY_REVIEWED_SELECTION_FILE_SHA256 \
  --manifest /verified/canonical/manifest.json \
  --output /new/reviewed-test-comparison.json
```

Use the original externally reviewed SHA256 values, including original whitespace
and final newline. The supplied manifest must match the exact manifest SHA bound
by both reviewed artifacts. There are no hardcoded current corpus/photo counts:
eligible TRAIN/VAL/TEST metadata inventories are reconstructed from this manifest.
The existing output is preserved; publication uses exclusive file creation.

The helper checks:

- Original TEST family, complete photo/record inventory, fixed recipe, four
  checkpoint hashes, source identity and canonical data match its own reviewed policy.
- The original policy's selected unified selection SHA and the stage2 wrapper's
  selection SHA both identify the externally pinned actual selection file bytes.
- Unified decision, training completion, validation provenance, source, canonical
  data and inference protocol equal its own selection artifact. Completed60 epochs,
  terminal steps, selected best epoch/steps, full-data configuration, configuration
  hash, threshold and selected checkpoint identity remain consistent.
- The source identities of the two pipelines may differ. Each is checked against
  its own frozen artifact; they share the manifest/content and exact TEST domain.
- Both reports contain every eligible TEST photo and annotation record exactly
  once, with deterministic first-sorted annotation-source attribution. The unified
  top-level photo and record counts match the dynamic manifest inventory. Both
  different inventory-hash schemas are reconstructed from metadata.
- Generic paired comparison verifies identical per-photo valid/positive GT count
  support and conditional A/V domains without opening GT. VAL reports, subsets,
  foreign IDs and conflicting pre-existing metric identity fields fail.

A new temporary candidate view is derived from the wrapper's metrics plus its
checked canonical data and checkpoint identity. The whole original report bytes,
derived view and original manifest bytes are passed through a temporary directory
to the unchanged generic `compare_reports`, using **2000 draws, seed42**, source
intervals and conditional/end-to-end A/V comparisons. Inputs are never retagged or
rewritten, and their bytes are rechecked after comparison.

`whole_input_sha256` records all five whole input file hashes. In particular,
`candidate_wrapped_report_sha256` identifies the complete original stage2 wrapper;
`candidate_report_sha256` and `derived_candidate_view_sha256` identify the temporary
flat metrics view consumed by generic comparison. These are deliberately separate.
The output also records the externally reviewed artifact pins and all four original
checkpoint hashes.

This is a conditional comparison of saved reports. It does not prove historical
blindness, authenticate reviewer identity, rerun model/GT validation, or measure
training-seed uncertainty. The original producer/selection artifacts retain those
boundaries. No actual TEST report or quality result is produced by this code change.

Verification:70 CPU tests pass (new helper plus existing generic and completed-VAL
comparison tests), including synthetic consistent reports with different pipeline
sources, external SHA and source/policy/checkpoint mismatches, rebound partial-run
metadata, VAL-as-TEST, protocol/record/GT-support discrepancies and an import probe
that forbids Torch, selection and heldout execution modules. Ruff passes. Evidence
is in `reports/reviewed-test-comparison/`.
