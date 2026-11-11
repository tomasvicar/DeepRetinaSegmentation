# Separate vascular2 BEST59 BatchNorm experiment

A bounded TRAIN-only BN calibration and subsequent fresh VAL experiment is running;
no quality gain is claimed before those measurements. This is a separate explicit
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
remain solely in the private operational repository. Actual results will be added
only after observed terminal completion and paired count checks.
