# Final TEST readiness audit — no TEST execution

Audited HPO705b3f5 and product2350bbf read-only. Existing software already has the
required split-specific, typed execution routes. **No runtime/ML/CLI change or
family broadening is needed.** This branch corrects stale TTA documentation and
adds metadata-only dispatch tests; it does not freeze/select a candidate or grant
execution permission. No real model, image, GT/TEST array, NN or GPU was accessed.

## Current corpus and exact operators

The pinned17a manifest declares TRAIN2529photos/5700records/2042known groups,
VAL770/993/612 and **TEST856/1118/696**. Read-only JSON audit confirms photo IDs,
source IDs and known split-group IDs never span partitions. Counts and hashes are
in `audit.json`. This does not establish unknown patient independence or original
model training blindness. Stage1 `data_identity` explicitly rejects source/photo
cross-splits; the paired comparator also rejects group cross-splits. The fixed
manifest is already group-clean, independently checked here; do not generalize
that observation to another manifest without its own audit.

`unified_validation.py` is intentionally VAL-only; adding `--split test` would
bypass the reviewed route and is not appropriate. `scripts/unified_test.py` and
`heldout.evaluate_reviewed_heldout` already evaluate only TEST after full schema2
selection/role/family/source/data/protocol verification. Stage2 accepts **no split,
threshold, tiling or TTA override**. It reconstructs the frozen descriptor and
loads checked checkpoint bytes, never a replaced path. Shared evaluator/mapping/
FOV/known-class AV/joint-fusion math stays unchanged. Metrics are checked against
actual TEST photo/record/support inventory, not a renamed VAL header.

### Original4nets:182 is not a weaker cup reference

Shipped `predict_original`, fixed evaluator44f1faf4…a61 and strong benchmark policy
all use the SAME four weight SHAs: dedicated vesseldae07480…, disc0raw7d6ddc3b…,
universal9fa4f9ff… for cup, and AV82428903…. Raw25, native tiles, strict logits>0,
universal cup∩SAME-forward disc0 **before** nearest restoration, and calibrated
provided-FOV AV from predicted vessel are the declared fixed operators.

Complete strong reference528233fe…bfa2c exists, with770photo counts; separate fresh
full770 reproduction53b5c0a6…04c6 matches all3850 confusion tuples exactly.
Nearest-backend original182ad346…6836 also declares that same composition. File-only
comparison proves its vessel/disc/**cup** TP/FP/FN/valid tuples are identical on
**every** VAL photo to528: cupDice is.4353989841007063 in both. This large gap to
new-model cup scores is not explained by swapping cup clipping or a weaker model
in182. Artery/vein differ on only2photos: pooledA TP−7/FP+10/FN+7; veinTP−18/FP+6/
FN+18, unchanged GTsupport; conditional AV also changes on2photos. Retain that
separate recorded nearest-backend/source measurement, not a retroactive528 tag.

528 has oldv1 global data pins;182 has current17a/00d9. Existing exact unchanged-VAL
bridge qualifies shared VAL support, **never TEST**. Their envelopes/source/runtime
remain different even where scores coincide. For final quality/speed coherence,
choose the actual fixed4net operator/source/reference together; resident five-mask
benchmark latency is not full productAPI load/probability/output latency. Existing
single-TRAIN product parity is operator evidence, not a new770photo quality run.

## Family and leader eligibility

| Candidate identity | Actual typed family/route | Requirement before freeze |
|---|---|---|
| Source72 standalone TPE/loss2 | `standalone_72`, original generic4b QA+reviewed protocol+observed supervisor terminal | Full60 and selected weight's own5700; never borrow LAST coverage for partial BEST |
| Source72 native V3 | `pool_72_v3`, original native task/COMPLETE receipt/index | Preserve native provenance, no standalone retag |
| raw512 BEST50 rot4/.35 | **`curated_fd44`, `training_best`**, SHA521b8881…32cea, original indexf5fad624…546a3 | Exact original selected/BEST/LAST/CSV/summary/QA paths+bytes/indexSHA, own5700, full original curated chain and fresh exact-protocol report |
| raw384/CLAHE/LC curated arms | Same explicit `curated_fd44`, saved raw/photometric full config | Each arm's original index/QA, own selected epoch coverage and saved descriptor/cache identity; no generic fallback |

Current code **and immutable6d3593f** already admit curated TTA explicitly. The
producer requires typed fd44 completion for nondefault TTA; final adapter delegates
to genuine `check_curated_completion`, retaining original tasks/receipts/PBS/
readonly milestone/config/source/cache/photometry and own role coverage. A family
string/QA0 alone is insufficient. New tests verify exact original-index/role/QA
pins and frozen rot4/.35 are forwarded by stage2 before any model/GT call; genuine
checker tests exercise constructed synthetic chains separately.

The old `FINAL_SELECTION_TTA.md` two-family sentence was stale and is corrected;
no verifier or allowlist was weakened. Historical source1f TTA reports remain
ineligible. Actual raw512 typed f13 report may be considered on its exact immutable
Old6d/Python3.12.14 runtime if original-byte stage1 recheck passes. A sanitized
aggregate is not that evidence. Current705 differs from6d in cli/unified/unified_cli
bytes: choosing that different source/runtime requires new fresh VAL, not retagging.
For CLAHE/LC, saved full-image photometry is preserved once before crop/TTA.
Held-out route computes it online; TRAIN+VAL3299 photometry cache does **not** cover
TEST and must not be passed off as a TEST cache. Budget its real preprocessing.

## Reviewable future commands — not executed

Freeze a chosen source/runtime **first**, then use only the immutable original
index's exact paths. For curated leader, set ROLE=training_best, SELECTED=BEST,
TTA=rot4 and THRESHOLD=.35 only if that protocol is actually selected from VAL.
The explicit variables below denote reviewed inputs, not inferred artifacts.
Other sources/roles use their own typed QA/protocol/terminal flags; never alias.

If no qualifying exact-source fresh VAL already exists:

```sh
python scripts/unified_validation.py --checkpoint "$SELECTED" --manifest "$MANIFEST" \
  --original-curated-index "$INDEX" --expected-curated-index-sha256 "$INDEX_SHA" \
  --checkpoint-role "$ROLE" --threshold "$THRESHOLD" --tile-size 512 --overlap 64 \
  --tta "$TTA" --device "$DEVICE" --cpu-threads "$THREADS" --output "$FRESH_VAL"
```

Create/recheck schema2 with genuine completion and own selected coverage:

```sh
python scripts/unified_selection.py create --checkpoint "$SELECTED" \
  --best-checkpoint "$BEST" --last-checkpoint "$LAST" --metrics-csv "$CSV" \
  --training-summary "$SUMMARY" --manifest "$MANIFEST" --validation-evidence "$FRESH_VAL" \
  --completion-family curated_fd44 --checkpoint-role "$ROLE" \
  --completion-qa "$ORIGINAL_QA" --expected-qa-sha256 "$QA_SHA" \
  --original-curated-index "$INDEX" --expected-curated-index-sha256 "$INDEX_SHA" \
  --tta "$TTA" --threshold "$THRESHOLD" --tile-size 512 --overlap 64 \
  --reviewer "$REVIEWER" --rationale "$VAL_DECISION" --output "$SELECTION"
```

Retain the whole selection SHA externally. `check` uses the same inputs, replacing
reviewer/rationale/output with `--selection` and `--expected-selection-sha256`.
Actual stage2 only after later explicit authorization; protocol comes from selection:

```sh
python scripts/unified_test.py --execute-reviewed-heldout \
  --selection "$SELECTION" --expected-selection-sha256 "$SELECTION_SHA" \
  --checkpoint "$SELECTED" --best-checkpoint "$BEST" --last-checkpoint "$LAST" \
  --metrics-csv "$CSV" --training-summary "$SUMMARY" --manifest "$MANIFEST" \
  --validation-evidence "$FRESH_VAL" --completion-family curated_fd44 --checkpoint-role "$ROLE" \
  --completion-qa "$ORIGINAL_QA" --expected-qa-sha256 "$QA_SHA" \
  --original-curated-index "$INDEX" --expected-curated-index-sha256 "$INDEX_SHA" \
  --device "$DEVICE" --cpu-threads "$THREADS" --output "$UNIFIED_TEST"
```

Original pipeline must use its OWN frozen evaluation source/runtime matching its
VAL report. If existing182 does not match that exact source identity/data, run new
fixed original VAL under that final source; never relabel182 or528. Then:

```sh
python scripts/evaluate_original_frozen.py freeze --validation-report "$ORIGINAL_VAL" \
  --manifest "$MANIFEST" --selected-unified-selection-sha256 "$SELECTION_SHA" \
  --rationale "$ORIGINAL_VAL_POLICY" --output "$ORIGINAL_POLICY"
python scripts/evaluate_original_frozen.py test --execute-reviewed-heldout \
  --policy "$ORIGINAL_POLICY" --expected-policy-sha256 "$POLICY_SHA" \
  --manifest "$MANIFEST" --data-root "$PREPARED_ROOT" --universal "$UNIVERSAL" \
  --vessel "$VESSEL" --disc "$DISC0" --av "$AV" --device "$DEVICE" \
  --cpu-threads "$THREADS" --output "$ORIGINAL_TEST"
```

It scores exactly the same eligible TEST photos/source/record/GT support while
preserving the fixed operator; source fingerprints can differ between pipelines.
Finally, saved-count comparison only:

```sh
python scripts/compare_reviewed_test.py --original-report "$ORIGINAL_TEST" \
  --original-policy "$ORIGINAL_POLICY" --trusted-original-policy-sha256 "$POLICY_SHA" \
  --candidate-report "$UNIFIED_TEST" --selection "$SELECTION" \
  --trusted-selection-sha256 "$SELECTION_SHA" --manifest "$MANIFEST" --output "$PAIRED_TEST"
```

Comparator binds BOTH policies to the same actual selection, requires TEST and
dynamic complete inventory, exact support/known-group attribution, all heads and
conditionalAV,2000/42. VAL-as-TEST/source/policy/checkpoint/protocol mismatches reject.
Keep dense/source adverse analyses beside pooled/macro results; no test-driven
threshold/checkpoint/TTA reselection. Historical original training exposure,
adaptive VAL development and unknown patient identity remain limitations.

## Tests actually performed in this audit

`tests` metadata below records source SHA, exact CPU1/CUDAhidden commands and
numeric exits. One initial invocation passed118 existing tests; six new metadata
cases failed because its synthetic output was inside its synthetic canonical
root (the production guard correctly refused). Only that fixture path was fixed,
then those six passed. Original failure logs are retained. No runtime source was
modified. Existing tests use constructed tiny weights/metadata and explicit
no-forward guards; no actual model/real GT/TEST data was decoded.
