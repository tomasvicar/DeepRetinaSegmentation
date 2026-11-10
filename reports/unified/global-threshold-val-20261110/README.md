# Proposed global-threshold VAL experiment

This is a pre-TEST plan, not authorization to run inference. No forward,
installation, training, test evaluation or default change has been performed.

After the original control finishes its planned 60 epochs and terminal QA passes,
freeze one independent copy of its **best** checkpoint. Its selected epoch may be
earlier than 60. Record its SHA256 and the completed last60 proof in an approved
pre-execution addendum. Never evaluate thresholds on a moving checkpoint or use
different weights for different candidates. Plan checkpoint SHA is deliberately
null until these bytes exist; evaluation is blocked until this pin is supplied.

The six candidates are 0.35, 0.40, 0.45, **0.50**, 0.55 and 0.60. Evaluate the
fresh 0.50 reference first, followed by the others in the declared order. Produce
six independent enriched reports using the existing `scripts/unified_validation.py`
producer, with a single frozen source/runtime/device and the same 770 validation
photos/993 selected annotation records. No new wrapper is needed. Existing producer
checks bind checkpoint bytes, configuration, evaluation source, canonical data,
selected annotations and inference protocol. It has no test split option.

The producer package bytes are pinned to
`3779e1b5e9577d6870cf5e9a3ee48fb2271941dbbca12891f7a7f002075c4c9c`
(base `4b6c07e`). The addendum must record the final immutable checkout commit,
the same package fingerprint and CLI script hash, and the actual locked runtime.
The training source remains `3157bb9`; this separate producer includes the
subsequent enriched evidence checks. The producer currently reads canonical
images/masks directly; this plan does not imply evaluation uses the training
derived cache. Canonical fingerprints remain unchanged.

## Selection and interpretation

Primary selection maximizes the existing five-head source macro Dice
`macro_dataset_mean_dice`. When scores are within 1e-12 of the maximum, prefer
0.50; otherwise prefer the threshold closest to 0.50, then the lower value.
Preserve the complete fresh 0.50 report regardless of the selected threshold.
Report every candidate, all five pooled and source-macro metrics, source-level
counts, vascular-only and disc/cup macro summaries, and AV conditional/end-to-end
accuracy. Secondary summaries explain tradeoffs; they do not silently replace
the declared primary selector or introduce an undeclared veto.

A global threshold changes vessel, disc and cup together. Cup positivity is
`P(cup)>=t AND P(disc)>=t`, so a cup score increase may come from disc containment
or changed cup support. Artery/vein class selection is the same conditional
argmax for every threshold; only the vessel gate changes. Conditional AV accuracy
on identical known GT pixels must therefore remain unchanged. Verify its exact
per-photo correct/labelled counts across all six reports; drift requires an
investigation before selection. Improved artery/vein Dice or end-to-end accuracy
from thresholding is a segmentation/gating change, **not improved artery-versus-vein
discrimination**.

The all-five selector can improve disc/cup enough to offset a vascular regression.
Publish vascular-only source macro, vessel recall, artery/vein Dice and all source
changes beside the primary score. A single global threshold cannot separately
optimize these objectives; the protocol deliberately does not add per-head or
per-dataset thresholds. Validation tuning over six choices is part of model
selection, not independent confirmation of superiority.

All six reports must agree on checkpoint/source/runtime/data, geometry and all
photo/annotation/GT support inventories; only the threshold may differ. Preserve
legacy25 raw RGB normalization and checkpoint tiling without overrides. After
reviewing all VAL evidence, freeze the selected weights plus inference protocol
and retained 0.50 reference before TEST is opened. No post-TEST retuning.

## Six planned manual commands

These commands are documentation only and have **not been executed**. Parent
approval and a complete addendum are prerequisites. Set `PYTHON` to an existing
uv-frozen Python executable, `EVAL_SOURCE` to the immutable producer checkout,
`CHECKPOINT` to the independently frozen best checkpoint, `MANIFEST` to canonical
61032be2 data, and `REPORTS` to a fresh directory outside canonical data.
`DEVICE` and `CUDA_VISIBLE_DEVICES` must designate the same reviewed device for all
six. Reserve it only after ongoing 60-epoch training has completed; do not overlap
an existing GPU job. Record identical thread/numerical flags and UUID/driver.

```bash
export PYTHONPATH="$EVAL_SOURCE/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-050.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.50
"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-035.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.35
"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-040.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.40
"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-045.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.45
"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-055.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.55
"$PYTHON" "$EVAL_SOURCE/scripts/unified_validation.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --output "$REPORTS/threshold-060.json" --device "$DEVICE" --cpu-threads 2 --threshold 0.60
```

The six existing CLI invocations are independently recorded evidence producers,
not a job array or scheduler. An existing report path is refused. No new wrapper
or inference implementation was introduced, so no wrapper-specific CPU test is
needed; only this JSON plan's candidates, pins and command correspondence were
checked without importing Torch or loading checkpoints.
