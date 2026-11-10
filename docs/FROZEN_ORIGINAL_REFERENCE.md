# Fixed strongest-original reference, full VAL and reviewed TEST

`scripts/evaluate_original_frozen.py` supplies the missing executable route for
the **declared selected** four-network reference: raw dedicated vessels,
calibrated provided-FOV AV, raw dedicated disc0 and raw universal cup intersected
with the **same-forward** predicted disc0. All segmentation thresholds are strict
`logits > 0`; clipping occurs in canonical coordinates. This does not change
original prediction defaults, relax old VAL-only experiments, or relabel saved
VAL counts as TEST.

The weights are fixed to the four SHA256 identities in `MODEL_SHA`. Safe model
loading uses those checked byte snapshots via `BytesIO`, `weights_only=True`
inside the existing loaders; checkpoint paths are not reopened for execution.
Raw RGB single-net inputs are `RGB/255 - 0.5`, with each checkpoint's native
tiling. AV reuses `calibrate_provided_fov`, `prepare_vesclass_matlab_calibrated`,
`predict_av_input`, `restore_classes` and canonical `ImageGeometry.prepare_mask`.
The predicted vessel and canonical provided FOV are NN-restored to the native
frame exactly as in `refresh_original_av.py`; AV ties remain class0. The original
raw proxy and automatic-FOV alternative are not used.

Input JSON metadata determines all eligible photo/annotation inventories. There
are no hardcoded770/856 or612 cup counts in the evaluator. Corpusv2 metadata has
770 VAL and856 TEST photographs, including27 newly prepared IDRiD disc TEST
photographs. IDRiD disc rows and photo metadata must retain the explicitly
recorded nominal50-degree acquisition angle. `capture_geometry_policy` validates
the manifest-bound policy for all newly introduced sources. Missing cup support
stays unknown; it never becomes negative cup GT. No fresh source-archive decoding
is needed for these disc-only IDRiD records: raw archive reading occurs only when
an AV reference has known classification pixels.

## Commands

Use a **separate frozen evaluation checkout** containing this script and matching
package sources. Keep the training checkout and its execution digest unchanged.
`MODEL` below denotes `src/deepretina/pretrained`; `DATA` is the read-only prepared
corpusv2 directory; `RAW` is the existing OwnCloud fundus archive root. Use the
same source/runtime for all original VAL/freeze/TEST steps.

```bash
uv run python scripts/evaluate_original_frozen.py val \
  --manifest DATA/manifest.json --data-root RAW \
  --universal MODEL/legacy-single-net-universal.pt \
  --vessel MODEL/legacy-single-net-vessels.pt \
  --disc MODEL/legacy-single-net-disc-0.pt --av MODEL/legacy-av-v9.pt \
  --device cpu --cpu-threads 1 --output reports/original-fixed-val.json
```

Compare fresh VAL per-photo/head integer counts and conditional AV support to
`reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json`.
The old report has the parent corpusv1 manifest/content identity, so use the
separately verified VAL-domain bridge to establish byte-identical VAL support;
do not rewrite its provenance to look like a fresh corpusv2 report. Fresh CPU
backend forwards may differ from historically composed counts: the existing
paired-cup study already records a two-FP native cup drift. Any new mismatch
needs explicit measurement and explanation before claiming reproduction. Tests
and functional execution alone do not establish equivalent quality.

Only after full60 training completion, measured VAL selection and the unified
selection artifact are frozen, freeze this original reference policy. Supply the
actual externally retained SHA of that unified selection artifact:

```bash
uv run python scripts/evaluate_original_frozen.py freeze \
  --validation-report reports/original-fixed-val.json --manifest DATA/manifest.json \
  --selected-unified-selection-sha256 UNIFIED_SELECTION_SHA \
  --rationale 'Declared original policy fixed from VAL before TEST comparison' \
  --output review/original-policy.json
```

Freeze checks complete current-source VAL inventory, source attribution, count
consistency, aggregate consistency, policy/data identity and decision timing.
The emitted SHA is the external review digest for the next command. The unified
selection SHA is a binding to the separate decision, **not** a claim that this
original-policy freezer verified unified training completion. The unified
stage1/stage2 gate performs that verification independently.

```bash
uv run python scripts/evaluate_original_frozen.py test \
  --execute-reviewed-heldout \
  --policy review/original-policy.json --expected-policy-sha256 ORIGINAL_POLICY_SHA \
  --manifest DATA/manifest.json --data-root RAW \
  --universal MODEL/legacy-single-net-universal.pt \
  --vessel MODEL/legacy-single-net-vessels.pt \
  --disc MODEL/legacy-single-net-disc-0.pt --av MODEL/legacy-av-v9.pt \
  --device cpu --cpu-threads 1 --output reports/original-fixed-test.json
```

The external original-policy digest, exact recipe/source/runtime, metadata
identity and four checkpoint byte digests are checked **before** device
resolution, model loading or any image/GT decoder. Full canonical byte integrity
then runs before CUDA/forward; source/canonical checks repeat after inference.
Bad review inputs fail without TEST decoding or CUDA initialization. Both inputs
and source must remain immutable throughout the operation; output creation is
exclusive. These unsigned records cannot authenticate a reviewer or prove
historical blindness.

The selected unified model uses the existing `scripts/unified_validation.py`,
`scripts/unified_selection.py create/check` and `scripts/unified_test.py
--execute-reviewed-heldout` route, from its final pinned evaluation source and
runtime. The canonical gate derives TRAIN/VAL/TEST counts from metadata and is
not limited to corpusv1's2221 training photographs. No general CLI test bypass is
needed. Do not change evaluation package sources between fresh VAL evidence,
selection and TEST; the source/runtime identity is part of that selection.

For final paired uncertainty, `deepretina.comparison.compare_reports` already
accepts split`test`, `include_sources=True`, `include_av=True`,2000 draws/seed42.
The original report is flat. The unified reviewed-stage2 report is wrapped:
make a **derived comparison view** from its `metrics` and attach
`manifest_sha256`/`prepared_content_sha256` from `canonical_data` and checkpoint
SHA from `checkpoint`; retain the original whole stage2 report and its SHA.
Verify that its `selection_sha256` is the exact unified decision digest bound
into the original policy. This is a metadata-only operation, never a rerun or
TEST-driven model/threshold selection. `compare_completed_val.py` remains VAL-only.

## Verification boundary

Synthetic tests cover dynamic inventories, IDRiD angle rejection, unknown cup
support, fixed source/weight/recipe review, failure before model/GT/CUDA,
strict-zero segmentation decoding, same-forward cup clipping, calibrated AV
calls/NN/tie0 and a complete synthetic VAL→freeze→TEST path. A separate test
really loads all four bundled checkpoints via CPU BytesIO without data or
inference. No actual TEST assets were read, no real TEST score was produced,
and no actual full VAL reproduction was run while adding this route.

Historical original training exposure, raw/normalized provenance and documented
photometric residuals remain qualifications in every report. Final comparisons
must show each head/source and conditional AV separately; a macro increase does
not establish improvement on every task.
