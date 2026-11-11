# Proposed native V3 BEST54 scalar threshold ablation

**Preparation only, no model forward/actual GT decode/TEST or launch.** Source318
is frozen and unchanged. New script `scripts/calibrate_native_v3_thresholds.py`
runs against that exact package API; its script SHA is separately pinned by a
private executionFalse packet. Checkpoint is genuine pool72 trial4 BEST54
`21c1b3c6126494b8534a022b2256ada38687c9194120f7be63bf66879687bdfb`.
Original typed pool receipt/index/own5700coverage remain operator prerequisites;
no standalone QA is fabricated and these derived reports are not freeze evidence.

Choose one explicit TTA mode **none / rot4 / dihedral8** after the currently
running four-mode sweep. Do not extend that sweep or silently use another mode.
For each of the770VAL photos, call `predict_probabilities` once under the chosen
mode and then decode the identical FP32 probability array at predeclared global
thresholds **.25,.30,.35,.40,.45,.50,.55,.60**. One probability inference withrot4
or8 contains4or8 neural views; it is not falsely called one neural view. The
original saved training configuration stays separate from actual threshold/TTA
inference protocol. No probability cache is persisted.

Every photo's0.5 counts must exactly match the native318 evaluator counters using
an injected **same** probability array (no extra NN calls). Valid/GT-positive
support and conditional A/V correct/scored/eligible counts must be threshold
invariant. The scalar changes vessel gating and disc/cup too, so inspect adverse
heads/source rows rather than interpreting a vessel gain alone.

The objective is one minimum of six pooled candidate/original ratios: five head
Dice and conditional A/V accuracy, using unchanged reference
`182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836`.
Select its maximum across the eight declared thresholds, ties closest0.5. This
is exploratory adaptive VAL tuning, not final model/protocol selection. Default
threshold remains0.5. Nonlinear worst-six minimum gets a point estimate only;
confidence intervals are explicitly for each head/source Dice and conditionalAV.

Saved-count comparisons reuse existing strict `comparison.compare_reports`:
2000source-stratified paired draws, seed42, exactly612knownsplitgroups, every
nonbaseline threshold vs0.5 plus exploratory selected threshold vsoriginal.
They verify all770photo/record/source/GTsupport identities. Intervals are
conditional on the adaptively chosen checkpoint/grid and do not remove multiple
comparison or seed uncertainty. Known groups are not proven complete independent
biological patients. Original historical exposure/preprocessing/backend limits
remain. No inference gain or changed default is claimed by this proposal.

Seven synthetic CPU counter checks passed in1.68s: same-array threshold effects,
conditional invariance, native counter parity for all three selectable modes,
nonfinite/FP64 rejection, Falsepacket admission and exactsix-ratio objective.
All model calls are intercepted and no corpus image/mask is opened. The unchanged
prior vascular2 script supplies pure counter/aggregation helpers, pinned at
`7be1227ac7d41247f847c169aaf43a061b0c4ac60363a9346b506ca8495bf147`;
its old model attribution is explicitly replaced in the new derived view.

The separate private packet requests Old CPU2/interop1, nice15/idle I/O and the
same Old GPU guard/shared lock, at most1800s inclusive of setup/cleanup (science
1650s). It stays executionFalse and TTA-unreviewed. No source/hardware/runtime
fallback, new job, install or training is part of this preparation.
