# Separate vascular2 BEST59 BatchNorm experiment

The completed TRAIN-only BN calibration did **not** improve overall vascular2 BEST
quality. Vessel gain is small with an interval crossing zero; macro and conditional
AV accuracy decrease. Keep the original parent; no model/default is replaced. This is a separate explicit
source0b/parent81113 experiment. The earlier source6b/parent1067 CPU-only helpers,
source trees, original checkpoint and original completion receipt are unchanged.

Recipe: original vascular_sampling_weight2 retained;8192 TRAIN patches, seed42,
batch4,384px, coverage-first all2529 photographs, basic augmentation, two workers,
RGB/255−.5. Actual PyTorch2.6 `torch.optim.swa_utils.update_bn` resets cumulative BN
statistics. No supervised loss, gradient or optimizer updates occur. TRAIN labels
are read by the existing valid-mask/foreground/source-task sampler; this is allowed
TRAIN usage, not a claim that the loader never opens labels. Only54 tracked BN
buffers may change, and every other parameter/buffer must remain byte-identical.

The original parent is BESTE59 from a genuinely completed60 run, with actual
5700/5700 annotation exposure in that selected checkpoint. Parent epoch/steps and
completed parent-run60 are separate facts. Derivation removes stale training VAL
scores and optimizer state, records the exact parent/recipe, and marks prediction
only. Parent completion QA certifies parent bytes, never newly derived weights.

The execution uses an idle GTX10606GB, FP32 without autocast, CPU2/interop1,
two workers, low CPU/I/O priority and a45-minute owned-process-group deadline.
Canonical prepared data are read directly: this host has no derived cache; cacheNone
is explicit and no8GB transfer is required. The prepared bytes/configured geometry
are unchanged. No validation/test data influence BN updates. Later fresh VAL scoring
necessarily reads VAL labels. TEST is never decoded/scored.

The frozen0b `unified_validation.py` expects an original training-selected score;
we do not fabricate one for the derived checkpoint. The separate research producer
uses the same frozen0b model/inference/evaluation functions, data/source checks and
fixed threshold.5/tile512/overlap64, verifies exact non-BN parent equality again,
and emits `vascular2_bn_recalibration_research_validation`, not standard selection
or pool completion evidence. There is no TEST or training-resume entry point.

Two new explicitly named scripts and16 tiny synthetic CPU tests cover actual
update_bn semantics, unchanged non-BN tensors, restoration on failure, cumulative
statistics, safe derived checkpoint reload, parent59 role, stale-score removal and
false-derivation rejection. Operator commands, hosts, process IDs and live logs
remain solely in the private operational repository. Actual completion and paired count checks are recorded below.


## Actual outcome: preserve the parent

The original bounded supervisor observed both processes exit0: TRAIN BN403.05s,
fresh VAL175.40s, total578.47s. Exact8192 samples and2529 distinct TRAIN photographs
were delivered. Independent postcompletion safe CPU loading confirms exactly54
BN state entries change and all74 other state entries remain byte-identical;
all floating state is float32, all BN batch counters are2048, and stale VAL scores
and optimizer state are absent. Parent59/120832 gradient steps remain distinct
from the completed parent-run60 claim. Neither original parent nor source changed.

| Measure | Original parent BEST59 | Derived BN | Difference [paired95% CI] |
|---|---:|---:|---:|
| Source/head macro | 0.803898 | 0.802282 | -0.001616 [-0.002647, -0.000284] |
| vessel | 0.756713 | 0.757220 | +0.000507 [-0.000413, +0.001369] |
| artery | 0.718048 | 0.708269 | -0.009778 [-0.012063, -0.007533] |
| vein | 0.779477 | 0.775658 | -0.003818 [-0.005521, -0.002182] |
| disc | 0.933591 | 0.933386 | -0.000205 [-0.000712, +0.000267] |
| cup | 0.874871 | 0.877252 | +0.002381 [+0.001069, +0.004414] |
| Conditional AV | 0.810257 | 0.807196 | -0.003061 [-0.004977, -0.001071] |

Vessel delta+0.000507 has a CI crossing zero; artery, vein and conditional AV
regress, while cup improves. Macro decreases with its CI below zero. This defined
recipe therefore does not supply a balanced improvement over its parent.
No alternative BN recipe was chosen after inspecting these results.

Against nearest-corrected original models, vessel Dice remains lower:
0.757220 versus0.770123, difference−0.012903 [−0.018031,−0.007969]. Exact V3 worst6ratio
is0.983245021558, bottleneck vessel. This small point increase over the parent’s
0.982586462423 does not establish improvement across all six criteria or erase
vessel regression. No CI for this minimum-of-six objective was computed.

`actual-val770/paired-vs-parent.json` and `paired-vs-original.json` include pooled
and per-source5-head confusion counts, differences, Dice/source-macro intervals,
and conditional/end-to-end AV counts and intervals. Both use2000 identical paired,
source-stratified split-group draws, seed42,770photographs/612known groups.
Source-head integer headers reaggregate exactly, reference positive/valid domains,
record IDs, annotations and source attribution match exactly. Split groups preserve
known aliases but do not prove complete patient independence. These exploratory
VAL intervals omit training-seed uncertainty and multiplicity correction.

Source0b/evaluation recipe is fixed, but parent fresh evaluation used RTX4070 and
derived used GTX1060; device/runtime differences are explicit and not causally
isolated. Incomplete original-model historical exposure and different original
preprocessing/decoding remain qualifications of comparisons to originals.
No TEST, held-out quality claim, release-weight publication, selection/default
change or training resume was performed.

Derived checkpoint SHA256:
`5be2fbce7bd91b3eafa34db26d60d9ecbe2bf862faf7085188927de48e45429a`.
Calibration SHA256:
`60a4edd5c97179559b271cf475c72b9cdc99ce756875f798b279375dcd88dbc1`.
Actual fresh VAL SHA256:
`6ed8101bdc23f312f1d273e81372de16ff6bfd89e79be186d43e407fafd9b0e2`.
Only reports/hashes are committed; derived weights stay in the separate read-only
experiment output. The original typed parent completion QA is never retagged as
certification of derived bytes. Twenty-one focused tests and Ruff pass.
