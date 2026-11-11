# Vascular sampling weight 4: actual completed-run VAL comparison

Both actual checkpoints passed the original strict standalone completion and own-coverage gates: **BEST E55** and **fixed terminal LAST E60**, each with **2,529 photos and all 5,700 annotation records contributing**. The training run finished 60 epochs; BEST is a separate validation-selected role. The existing bounded QA/validation wrapper returned an observed **0**, with strict QA and both fresh 770-photo VAL producers returning 0. No new inference was performed by this count analysis.

The BEST has higher pooled vessel, artery, vein and conditional AV scores than the current vascular2 BEST on this development VAL set. Its overall source-macro difference has an interval crossing zero, disc/cup tradeoffs remain, and vessels still trail the strong original reference. This is **not** final model selection, an all-task/all-source win or TEST evidence. No weights, registry entries or aliases are changed.

## Fresh point estimates

| Metric | Strong original | Vascular4 BEST E55 | Vascular4 LAST E60 |
|---|---:|---:|---:|
| vessel pooled Dice | 0.770123 | 0.760387 | 0.753186 |
| artery pooled Dice | 0.689522 | 0.731240 | 0.724778 |
| vein pooled Dice | 0.697286 | 0.792607 | 0.786561 |
| disc pooled Dice | 0.894808 | 0.930511 | 0.926244 |
| cup pooled Dice | 0.435399 | 0.860040 | 0.850235 |
| Conditional AV | 0.748566 | 0.822883 | 0.823125 |
| Source-macro all five | 0.667030 | 0.805977 | 0.800835 |

Worst-six ratio versus the pinned original: **0.987357 BEST**, **0.978007 LAST**; vessels are the bottleneck for both. This minimum is a point estimate, with no claimed bootstrap interval for the minimum.

## Paired changes against vascular2

Intervals below are paired source-stratified 95% percentile intervals from **2,000 draws, seed 42**, over the exact same **770 photos / 612 known split groups** and exact GT/source/record support. They are conditional on these frozen models, without multiplicity correction or seed-level/model-selection uncertainty. Recoverable groups do not establish complete biological patient independence.

| Metric | BEST55 − BEST59 (95% CI) | LAST60 − LAST60 (95% CI) |
|---|---:|---:|
| vessel pooled Dice | +0.003674 [+0.002030, +0.005638] | -0.003037 [-0.007063, +0.000614] |
| artery pooled Dice | +0.013192 [+0.006257, +0.020625] | +0.025430 [+0.017273, +0.033322] |
| vein pooled Dice | +0.013130 [+0.007760, +0.018998] | +0.013222 [+0.007175, +0.019476] |
| disc pooled Dice | -0.003080 [-0.009823, +0.001982] | -0.008049 [-0.014915, -0.002574] |
| cup pooled Dice | -0.014831 [-0.047695, +0.007009] | -0.026081 [-0.060101, -0.001583] |
| Conditional AV | +0.012626 [+0.007819, +0.018646] | +0.021018 [+0.014953, +0.027306] |
| Source-macro all five | +0.002079 [-0.004229, +0.007305] | +0.002434 [-0.003307, +0.008040] |

Compared with the strong original, BEST vessel Dice changes by -0.009736 [-0.014474, -0.005032]; LAST changes by -0.016937 [-0.022748, -0.011601]. The original vessel model therefore remains a material reference, even though the artery, vein, disc, cup and conditional AV point estimates improve.

LAST-versus-LAST is explicitly matched by terminal epoch, not proof of isolated causal effects. BEST55-versus-BEST59 is a comparison of two separate VAL-selected checkpoints, not matched-terminal evidence. BEST-vs-control LAST and cross-role comparisons are also retained, without silently renaming their roles. All eight results are available, including comparisons to original, control LAST, vascular2 LAST and vascular2 BEST.

## Per-source tradeoffs

Full per-source integer counts and paired intervals are retained in every comparison. The table gives BEST55 minus vascular2 BEST59 Dice changes for **all supported sources**; a dash means no support. DRIVE/RITE shared photos follow the existing A/V-first attribution to `rite`, not an invented additional independent sample.

| Source | Vessel | Artery | Vein | Disc | Cup |
|---|---:|---:|---:|---:|---:|
| aria | 0.005481 | — | — | -0.000081 | — |
| avrdb | 0.010353 | 0.040853 | 0.012751 | — | — |
| chasedb1 | -0.007482 | — | — | — | — |
| drhagis | 0.005427 | -0.015918 | 0.014109 | — | — |
| drishti | — | — | — | -0.002762 | 0.005863 |
| erlangen | 0.000022 | — | — | — | — |
| g1020 | — | — | — | -0.022096 | -0.074793 |
| hrf | -0.000195 | 0.008317 | 0.007827 | — | — |
| iostar | 0.000204 | 0.007190 | 0.021752 | 0.006000 | — |
| onhsd | — | — | — | 0.000753 | — |
| refuge | — | — | — | 0.006621 | 0.009593 |
| rimone | — | — | — | 0.004839 | 0.002170 |
| rite | 0.006728 | 0.001941 | 0.017985 | — | — |
| stare | 0.000021 | — | — | — | — |

For example, CHASE vessel Dice declines by about 0.00748 (interval below zero), and G1020 disc/cup decline by about 0.02210 / 0.07479 (both intervals below zero). REFUGE disc/cup improve by about 0.00662 / 0.00959 (both intervals above zero). AVRDB and HRF A/V improve on their supported heads. These contrasts prevent interpreting pooled vascular gains as a universal source-level improvement.

## Provenance and reproducibility

This uses immutable source0b, protocol29d3 and the genuine original standalone verifier4b568. Public completion summary is explicitly a projection of the original QA, not a pool receipt or a replacement original verifier artifact. Original QA, protocol, supervisor terminal, validation producer JSON and logs are preserved byte-for-byte privately; their actual SHA256 values remain in the reports. BEST checkpoint SHA starts `991b117a`, terminal LAST starts `a23ee273`; full values and exact own-coverage receipts are in the public summary. No model file is included.

`best-counts.json` and `fixed_terminal60-counts.json` preserve exact native per-photo counts while omitting evaluation runtime metadata; each names the full original producer SHA and has its own projection bytes. Statistical results can be reproduced from these count views with the existing `deepretina.comparison.compare_reports`, 2,000 draws / seed 42 / sources and AV enabled. Reference report hashes and complete input hashes are in all eight output files. The stricter standalone comparison CLI used the original private typed QA/protocol/terminal bytes, not a fabricated public summary.

Analysis was CPU-only on one local core (affinity 6, nice 17, OMP/MKL/OpenBLAS 1, CUDA hidden), and both comparison commands returned observed 0. No RGB, masks, model weights, GPU, TEST targets or test scores were opened. Threshold stayed 0.5 with the frozen FP32/tile512/overlap64/containment recipe. No inference, training, source mutation or extra study trial was launched. Original historical exposure/preprocessing and backend differences remain qualifications. Shared development VAL, selected epochs and repeated comparisons make these intervals descriptive rather than independent final-model evidence.
