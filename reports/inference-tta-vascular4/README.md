# Fixed-weight unified inference TTA on VAL

This opt-in experiment evaluated the same immutable vascular4 BEST55 checkpoint on all770 VAL photographs (612 known split groups), at25ppd, tile512/overlap64 and threshold0.5. All cases used FP32 on one GTX1060 6GB, CPU2/inter-op1. No training, TEST evaluation, cache build, threshold change, or default promotion occurred.

The checkpoint completed60 training epochs; its own best was epoch55. Checkpoint SHA`991b117a8fce9bd08c997a5de1e09a6dcd7048969cd6cb055b78787e5a6b6dfa`; manifest SHA`17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`. Geometry/photometry ran once before full-prepared-image transforms; joint vessel/AV/disc/cup probabilities were inverse-aligned and averaged before recovering conditional AV. See [the operator and API contract](../../docs/inference-tta.md).

| Mode | Macro dataset mean Dice | Change vs none (paired95% CI) | Conditional AV accuracy | Invocation seconds |
| --- | ---: | --- | ---: | ---: |
| none | 0.805973 | reference | 0.822879 | 204.60 |
| flips2 | 0.809128 | +0.003156 [+0.001669, +0.004872] | 0.829177 | 324.17 |
| rot4 | 0.812370 | +0.006397 [+0.004181, +0.008699] | 0.834518 | 553.15 |
| dihedral8 | 0.813082 | +0.007110 [+0.004973, +0.009423] | 0.835442 | 1001.73 |

All five pooled and macro-per-head point estimates improved in each ensemble. The highest observed aggregate was dihedral8; rot4 captured most of that observed gain with a shorter invocation. This is a quality/cost observation, not a promoted default.

| Mode | Vessel Dice | Artery Dice | Vein Dice | Disc Dice | Cup Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.760385 | 0.731228 | 0.792607 | 0.930511 | 0.860039 |
| flips2 | 0.762869 | 0.738592 | 0.797670 | 0.932004 | 0.862633 |
| rot4 | 0.764338 | 0.744870 | 0.801935 | 0.932933 | 0.863668 |
| dihedral8 | 0.764675 | 0.745964 | 0.803047 | 0.933065 | 0.863919 |

DRHAGIS artery and vein point estimates regressed in every ensemble. Each AV comparison there has only3 supported photographs/known groups and1951 defined bootstrap draws. The intervals include zero; the adverse point estimates remain part of the result.

| Mode | DRHAGIS artery Dice change (paired95% CI) | DRHAGIS vein Dice change (paired95% CI) |
| --- | --- | --- |
| flips2 | -0.010499 [-0.021311, +0.003504] | -0.007654 [-0.015872, +0.001955] |
| rot4 | -0.003540 [-0.030692, +0.012795] | -0.006237 [-0.018390, +0.011062] |
| dihedral8 | -0.000791 [-0.024093, +0.012267] | -0.003535 [-0.012605, +0.009922] |

Conditional AV accuracy changes versus none were +0.006298 for flips2, +0.011639 for rot4, and +0.012563 for dihedral8. Their paired95% intervals were respectively[0.004183,0.008428], [0.007822,0.015247], and[0.008352,0.016632]. Full per-source/head counts, intervals, defined draw counts, timings and provenance digests are in [results.json](results.json).

The fresh none report exactly matched the prior same-device512/64 report in all770 per-photo count/support/annotation traces, every metric except runtime, and inference protocol. The implementation/runtime source identity changed, and the new configuration representation explicitly adds the legacy defaults `mixed_precision=none` and `photometric_profile=raw`. Raw masks/probabilities were not saved for a bitwise comparison.

Every case completed with an observed child exit0; a separate observer recorded supervisor exit0. The four cases took about34m45s in total under a60min cap. Completed raw reports were preserved read-only, all original inputs and scripts retained their hashes, and no owned GPU processes remained. Operational paths/PIDs/UUIDs and raw per-photo evidence remain in the private workflow record.

Intervals use2000 source-stratified paired bootstrap draws of reviewed split groups, seed42. The three comparisons and all source/head intervals are exploratory and unadjusted for multiplicity. These are adaptive VAL observations from one trained checkpoint; the612 known groups do not establish complete patient independence. Fixed case order and warm filesystem can confound invocation times. These results are not held-out generalization, standalone original-protocol certification, or final model selection. `none` remains the default.

The sanitized code/tests passed166 focused synthetic CPU tests; the private controller additionally passed5 ownership/review/observed-exit tests. The operator implementation and decoder are shared across public prediction and evaluation paths.
