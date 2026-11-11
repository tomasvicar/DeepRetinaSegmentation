# Vascular sampling 2: paired VAL evidence

**Vascular sampling 2 improves vascular measures over the completed sampling-1 control,
but vessel segmentation still regresses against the original models. This is VAL
evidence, not a final selection, clinical generalization claim or TEST result.**

The main matched-budget comparison is terminal E60 versus control terminal E60.
The original training-selected BEST E59 is reported separately; it is not relabeled
as E60 or used to claim an unbiased comparison with the fixed control. Both new
checkpoint roles have actual exposure to all 2,529 TRAIN photos and all 5,700
eligible annotation records. Source corpusv2 has 21 TRAIN selectors; the fixed VAL
domain contains 770 photographs, 993 annotation records and 612 known split groups.

## Measured quality

Five segmentation values below are pooled Dice; AV is conditional class accuracy.
Macro is source/head macro Dice. All measurements use the same per-photo GT
positives, valid support, deterministic annotation IDs and head/source attribution.

| Measure | Original nearest-corrected | Control E60 | Sampling2 LAST E60 | Sampling2 BEST E59 |
|---|---:|---:|---:|---:|
| Source/head macro | 0.667030 | 0.788763 | 0.798400 | 0.803898 |
| vessel | 0.770123 | 0.750472 | 0.756223 | 0.756713 |
| artery | 0.689522 | 0.677109 | 0.699348 | 0.718048 |
| vein | 0.697286 | 0.754314 | 0.773338 | 0.779477 |
| disc | 0.894808 | 0.934122 | 0.934293 | 0.933591 |
| cup | 0.435399 | 0.872808 | 0.876316 | 0.874871 |
| Conditional AV | 0.748566 | 0.788037 | 0.802107 | 0.810257 |

## Matched E60 differences and paired 95% intervals

| Measure | Sampling2 E60 − control E60 | Sampling2 E60 − original |
|---|---:|---:|
| Source/head macro | +0.009638 [+0.004376, +0.014147] | +0.131371 [+0.121440, +0.142223] |
| vessel | +0.005751 [+0.002260, +0.009339] | -0.013901 [-0.019490, -0.008463] |
| artery | +0.022239 [+0.012213, +0.033070] | +0.009826 [-0.006521, +0.025196] |
| vein | +0.019024 [+0.011870, +0.026380] | +0.076052 [+0.062083, +0.089602] |
| disc | +0.000171 [-0.008186, +0.005483] | +0.039485 [+0.034491, +0.044805] |
| cup | +0.003508 [-0.006961, +0.011026] | +0.440917 [+0.426591, +0.454459] |
| Conditional AV | +0.014071 [+0.006911, +0.021435] | +0.053541 [+0.038329, +0.067723] |

LAST vessel gains over control (+0.005751, CI above zero) but remains below the
original (−0.013901, CI below zero). LAST artery pooled improvement over the original
has an interval crossing zero. Disc/cup differences from control also cross zero.
The better macro alone does not establish improvement on every task or source.

## Remaining vessel regressions by annotation source

| Source | Original | Control E60 | LAST E60 | BEST E59 | LAST − original [95% CI] |
|---|---:|---:|---:|---:|---:|
| aria | 0.732888 | 0.711963 | 0.701207 | 0.703204 | -0.031681 [-0.043929, -0.020824] |
| avrdb | 0.872294 | 0.850534 | 0.954877 | 0.961054 | +0.082583 [+0.066338, +0.112235] |
| chasedb1 | 0.790873 | 0.762813 | 0.765268 | 0.764350 | -0.025604 [-0.037703, -0.018807] |
| drhagis | 0.793068 | 0.757151 | 0.757371 | 0.750594 | -0.035697 [-0.044812, -0.027987] |
| erlangen | 0.620504 | 0.634619 | 0.632641 | 0.632839 | +0.012137 [+0.000631, +0.027008] |
| hrf | 0.807503 | 0.785424 | 0.787748 | 0.785807 | -0.019755 [-0.030821, -0.013880] |
| iostar | 0.816907 | 0.805720 | 0.799440 | 0.801192 | -0.017466 [-0.031831, -0.002715] |
| rite | 0.813105 | 0.794785 | 0.791861 | 0.791351 | -0.021244 [-0.030872, -0.011294] |
| stare | 0.832768 | 0.824358 | 0.812751 | 0.806500 | -0.020017 [-0.062902, +0.007270] |

Every paired JSON also contains all five heads per source, full TP/FP/FN/valid
counts for both pipelines and their deltas, precision/recall, supported photos/groups
and paired Dice intervals. These tables do not hide unsupported or regressing heads.

## Exact V3 worst6ratio definition

This is the minimum of candidate/original ratios for **five pooled Dice values
plus conditional AV accuracy**, not six selected sources. LAST scores
**0.981950284732**, BEST **0.982586462423**;
both bottlenecks are vessel Dice. Every ratio is retained in the machine report.
No bootstrap interval for this minimum score was computed. A value below one
means at least one of the six criteria remains below the original reference.

## Provenance and limits

- `standalone-completion-summary.json` is explicitly a public-safe derived summary
  of the genuine `standalone_unified_completion_qa`, not a pool completion receipt.
  Original QA byte hash, source0b/config/cache/protocol identities, BEST/LAST hashes,
  own-epoch annotation/photo coverage and completed60 budget remain bound.
- The measured fresh report files are unmodified. Private operator paths, original
  supervisor text, host commands and operational logs are excluded from this branch.
- Reference original bytes are fixed at `182ad346…` (the d14 nearest-corrected
  original recipe), control report at `b3b7d7e9…`; complete SHA256 pins are in code
  and each comparison. Original decoding includes dedicated disc0 clipping the cup.
- `compare_vascular2_val.py` checks exact report/manifest/summary bytes and role
  bindings, then reuses unchanged `deepretina.comparison.compare_reports`.
  All four comparisons use the same source-stratified split-group bootstrap:
  2,000 draws, seed42, one paired draw shared by all heads and AV measures.
- Integer per-photo counts are nonnegative and bounded, pooled and source headers
  are reaggregated, per-photo GT-positive/valid counts and conditional scored/eligible
  pixels match exactly. AV exclusive-decoder consistency is checked by the shared helper.
- Known split groups preserve recoverable patient/photo aliases; full patient
  independence is not established. Intervals omit training-seed uncertainty and
  multiplicity correction. Development/selection used this VAL domain.
- Control and sampling2 have matching declared config except the newly explicit
  `vascular_sampling_weight` (old control default1, new2). Actual source differs
  to implement this coefficient; fresh control evaluation used CPU and sampling2
  CUDA. This is not a perfectly isolated causal or multi-seed study.
- Fresh BEST differs from saved BEST by only a few boundary-count decisions:
  pooled vessel TP+2/FP−2/FN−2 and vein FP+1; no GT/support/attribution change.
  Count-exact CPU/GPU/backend reproduction is not claimed.
- Historical original training exposure is incompletely recoverable and original
  preprocessing/photometry differs. No independent test blindness is established.
- No checkpoint, image, GT mask, TEST report or new NN forward was opened for this
  analysis. Only report JSON and manifest metadata were read. No model was selected.

## Reproduce the metadata-only comparison

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  uv run python scripts/compare_vascular2_val.py \
  --manifest "$CORPUS_V2_MANIFEST" --directory reports/vascular2-paired-val
```

The manifest must match SHA256 `17a2277c…`; outputs use exclusive creation.
Preserve prior outputs before a separately named repeat. Ten focused metadata
checks pass, including genuine role fixtures and rejection of pool retagging,
partial coverage, wrong checkpoint/source/sampling/protocol and altered source headers.
Ruff passes. No source gate or original completion schema was weakened.
