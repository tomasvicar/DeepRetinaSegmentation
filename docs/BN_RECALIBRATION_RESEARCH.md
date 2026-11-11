# TRAIN-only BatchNorm recalibration: prepared controlled research

This branch prepares one possible post-training statistics intervention for the
fixed compact base12/depth4/batch4 model. It does not launch the actual8192-image
calibration or claim improved quality. Original checkpoints and frozen source6b
are untouched. No VAL/TEST arrays or model predictions were examined here.

## Actual E52 versus E60 metadata observation

`reports/bn-recalibration-control60/state-drift.json` was computed from exact safe
`weights_only=True` CPU byte snapshots of the actual best52 and last60 files.
All model state shapes/dtypes and configurations match. Parent last SHA:
`10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2`.

| State group | Relative L2 change versus E52 | Maximum absolute change |
|---|---:|---:|
| Learnable parameters |0.0250136255|0.0726317167|
| BN running mean/variance buffers |0.1113289343|23.6400681|
| BN batch counters |0.1538461538|16384|

All2208 BN-stat elements changed. The larger relative buffer drift is consistent
with a BN-statistics hypothesis, but these heterogeneous quantities do not prove
causality. Parameters also changed. The proposed intervention holds parameters
fixed and is an empirical diagnostic; only fresh VAL can establish its measured
quality effect. It must not claim recovery of an earlier score before evaluation.

## Exact prepared recipe

`scripts/recalibrate_bn.py` requires an isolated process, explicit execution flag,
CPU-only/CUDA hidden, the exact frozen6b executable source digestcb5 and exact
parent last60 SHA. It imports `UnifiedDataset`, `CoverageSampler`, `VerifiedCache`
and `load_unified` ONLY from that frozen source checkout, not this research code
package. No training loop or loss is copied.

-8192 delivered TRAIN samples, seed42, batch4, workers2, patch384;
-coverage-first shuffled2529-photo pass, then the same frozen
 `dataset_task_balanced` weighted extras using its shared sampler/loader generator;
-foreground probability0.5; TRAIN targets may guide crop/observer selection;
-augmentation`basic`: existing source D4 rotate/flip retained; no affine or
 photometric augmentation; images are centered exactly once as`RGB/255-0.5`;
-pinned canonical manifest17a227..., content00d9..., full train/val cache18c...,
 strict cache pin and no canonical fallback. Only TRAIN rows are iterated or
 decoded by the calibration loader; no VAL/TEST array is loaded or scored.

The frozen VerifiedCache constructor performs its normal complete encoded-byte
integrity check, which can hash encoded assets from other partitions; it does
not decode their arrays. This transport-integrity read is distinct from using
VAL/TEST images/GT to calculate BN statistics. `require_coverage` is called on
TRAIN rows only. Source support scanning/observer/crop selection uses TRAIN only.

The actual installed PyTorch2.6.0+cu124 `torch.optim.swa_utils.update_bn` source
was inspected and its function SHA pinned:
`4d67111e1e01591e8b3475e0e6353eb92e349907a6f02d06b827bcec3998c6ee`.
See `update-bn-reference.json`. It is decorated`no_grad`, resets BN running
statistics/counters, sets BN momentum to`None` for cumulative per-batch averaging,
uses train mode during the pass, then restores momenta. Equal batches of4 avoid
unequal-final-batch weighting. The wrapper additionally restores all original
module modes and momenta even on failure. No optimizer is created and no gradient
update is performed. The compact architecture has no dropout layer to introduce
an additional training-mode stochastic intervention.

Every learnable parameter AND every non-BN state buffer must remain identical
in raw tensor bytes, including signed-zero representation. Model state keys and
finiteness are checked. Only tracked BN`running_mean`, `running_var`, and
`num_batches_tracked` may differ. Actual delivered parent indices, not worker
prefetch, count toward8192 and the required2529-photo coverage. Per-photo visits
and before/after BN hashes are reported. No calibration quality metric is invented.

## Derived checkpoint meaning

The output is `derived-predict-only.pt`, never`best.pt` or`last.pt`. The exact
parent training config/epoch60/gradient steps/coverage remain inherited facts;
the calibration recipe is a separate field (do not rewrite original mild
training augmentation as basic). Lineage records parent SHA,60 gradient epochs,
zero calibration gradient updates, prediction-only and no training resume.
Stale top-level validation scores/metrics and optimizer/RNG training state are
removed. Parent source/data/cache/geometry provenance is retained. All5700
annotations' prior gradient-training contribution belongs to the parent, not a
new gradient update in this calibration.

The frozen `load_unified` schema remains suitable for prediction, but current
fresh evidence/final-selection/completion gates require original training-score
and QA semantics and are NOT claimed to accept the derived payload. Root can
review a separate clearly labelled research VAL measurement with parent and
calibration provenance. Do not restore fake score fields, relabel the derived
SHA as original QA.last, or claim inherited QA certifies modified BN buffers.
No calibration registry, final-selection adapter or TEST route is added here.

## Future execution, only after root operational review

The explicit execution flag below is an operational guard under existing user
authorization, not a new user permission flow. It has NOT been run on real data.
Use an owned process group/deadline wrapper; CPU4 on physical cores8,10,12,14,
nice15/ioniceidle. Stop the owned group on timeout and preserve partial evidence;
full8192 CPU throughput is not measured by these tests.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 taskset -c 8,10,12,14 nice -n 15 ionice -c 3 \
/path/to/active87/.venv/bin/python scripts/recalibrate_bn.py \
  --source-root /path/to/DeepRetinaSegmentation_v2_training_6b3dfad \
  --checkpoint /immutable/E60/milestone/last.pt \
  --checkpoint-sha256 10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2 \
  --manifest /canonical/corpusv2/manifest.json --cache /cache18c \
  --output /new/private/bn-calibration-control60 --cpu-threads 4 \
  --execute-train-calibration
```

Output directory and checkpoint/
JSON files are exclusive. The script checks parent bytes and frozen source
again after the pass; it never overwrites the parent. Actual full calibration
and any fresh VAL remain pending review. Tiny synthetic CPU tests only cover
buffer-only changes, cumulative averaging, centering/delivered coverage,
mode/momentum cleanup on failure, rejection of non-BN side effects, exact repeat,
modified updater rejection and stale-score/predict-only lineage semantics.

Prepared verification:9 tiny CPU tests passed in1.91s, including safe unified
checkpoint save/load with odd rectangular prediction equality. Ruff check/format
and syntax compilation pass. No actual8192-sample calibration was executed.

## Prepared honest research VAL producer (not executed)

`scripts/validate_bn_recalibration.py` is a separate proposed research route for
root review. It never restores a parent validation score or invokes a final
selection/TEST gate. It checks external parent/derived/sidecar byte pins, actual
calibration coverage/recipe, original versus derived config/gradient history,
absence of stale scores/optimizer state and independently rechecks every non-BN
state byte against the parent. Only then does it use frozen6b
`evaluate_unified_model(..., split='val', threshold=.5)` with the original config
and pinned cache18c. Full770 photo identity/source/data leases are retained.

The FLAT output has `kind=bn_recalibration_research_validation`, actual fresh
per-photo/header counts and source/runtime/parent/calibration provenance,
`selection_authorized=false`, `test_evaluation_enabled=false`. It contains no
inherited training score and no standard `unified_validation_evidence` receipt.
Generic metadata-only paired counts comparison can consume the flat metrics;
that does not mint a final-selection artifact. This proposed producer has NOT
been run on real VAL arrays. Additional synthetic tests reject changed
parameters, stale scores, wrong derived SHA, missing coverage and changed recipe.
Combined prototype/producer tests:15 pass in2.14s; no quality measurement yet.

## Actual reviewed TRAIN calibration

The once-only CPU4 calibration completed successfully in709.9437s. It delivered
8192 samples covering every2529 TRAIN photo, minimum calibration visits1.
Independent parent-versus-derived byte checks confirmed all non-BN states are
bitidentical;54 BN state keys changed and every BN batch counter is2048. The
parent retains actual60-gradient-epoch/all5700-annotation history; calibration
adds zero gradient updates. Derived SHA is
eecfebac1f9ee77f26de49654d1caf0dd1c5e35a733900bfce85afc3970cabe3.
The original parent was unchanged and the derived model/sidecar/original terminal
were copied offhost to an immutable backup outside Git/OwnCloud. No model binary
is added to this research branch. Clean aggregate evidence is in
`reports/bn-recalibration-control60/actual-train8192/summary.json`.

This successful statistics pass establishes no quality improvement. Root then
approved a separate once-only research VAL770 run with its own20min CPU budget;
that report and any paired comparisons remain distinct from this TRAIN summary.
