# Experimental unified RGB model

This branch implements one shared network with five logits: vessel, artery
conditional on vessel, vein conditional on vessel, disc, cup. The default compact
UNet has 1,106,597 parameters (base width12, depth4, BatchNorm). This is a verified training/prediction pipeline for an experimental model.
The bundled `unified-full-data-baseline` completed60 epochs on all2,221 eligible
TRAIN photographs across17 sources, with all5,281 annotation records contributing.
Its validation-selected checkpoint is epoch56; optimization continues and a final
new model has not yet been selected. Functional tests and short screens do not establish final quality. Published original
models remain the reference until the full-data comparison is complete.

## Data and geometry

Use the canonical manifest produced by `scripts/prepare_unified_data.py`, then
apply the benchmark split/exclusion audit before training. Each record contains
`id`, `dataset`, `task`, `split`, `image`, `mask`, `source_id`, and preferably
`photo_id`. Image/mask paths are relative to the manifest directory. NPZ targets
and validity arrays have native task channels: vessels1, AV3, disc/cup2. Optional
FOV intersects validity. Unknown annotation channels never become background.
`usable_for_training=false` and explicit training exclusions are respected.

Canonical images use the verified original physical preparation:25pixels/degree,
raw RGB without local contrast normalization, `preprocessing_profile=legacy25_raw`.
The student receives RGB/255−0.5. Full-data mode requires explicit profile/scale,
rejects modern maximum-dimension caps and every `preparation_preview=true` row.
Raw-photo prediction calls the same original preprocessing, then restores the
inclusive FOV crop and original resolution. Supply the dataset or camera FOV
angle for appropriate geometry; prediction JSON records the chosen parameters.

`source_id` constrains patient/source split membership. It does **not** merge
photographs. `photo_id` merges identical prepared RGB aliases, checked against
actual decoded pixels. Paired eyes remain separate photos. During training a
photo contributes one randomly selected vascular bundle (one AV observer with
its consistent vessel target, or one binary vessel observer) and one paired
disc/cup observer. Thus alternate binary observers remain usable without mixing
inconsistent vessel and AV references. Validation deterministically prefers AV
when available and chooses the first sorted observer. Observer-specific results
must identify this reference choice.

## Objectives and decoding

Vessel, disc and cup use masked BCE plus optional soft Dice, averaged over
annotated objectives. Artery/vein use two-class cross entropy only where both
class labels are valid and exactly one reference class is positive. Unknown,
ambiguous and unclassified vessel pixels provide no AV classification gradient.
At prediction, every detected vessel pixel receives artery/vein argmax; there is
no independent class confidence rejection. Cup is restricted to predicted disc.

Tiling and geometry restoration blend joint AV probabilities Pv×Qa and Pv×Qv,
then recover conditional class probabilities. This prevents low-vessel-confidence
background tiles from dominating the class decision. Public probability arrays
contain [Pv,Qa,Qv,Pdisc,Pcup]; Qa+Qv=1, including neutral0.5/0.5 outside FOV.
`decode_probabilities` applies vessel gating and anatomical hierarchy.

## Training and coverage

```bash
uv run python -m deepretina.unified inspect data/fair-unified/manifest.json
uv run deepretina train-unified \
  --manifest data/fair-unified/manifest.json --output runs/unified-all-sources \
  --purpose full_data --epochs 60 --epoch-samples 8192 --stop-after-epoch 1
uv run deepretina train-unified \
  --manifest data/fair-unified/manifest.json --output runs/unified-all-sources \
  --resume runs/unified-all-sources/last.pt
```

Set epoch samples to **max(8192, number of eligible training photos)**. Full-data
mode enforces a shuffled pass through every photo before source/task-balanced
replacement extras; a smaller budget or disabled coverage pass is rejected.
Task balancing derives actual supported objectives from masks. Dataset grouping
can be explicitly overridden by `sampling_group`. The recommended first run is
60epochs, batch4, patch384, foreground sampling0.5, mild paired augmentation,
AdamW and cosine schedule. The supplied configs/unified.toml uses LR0.0003,
a promising screened starting point; the historical default is0.001. Measure
epoch1 time before forecasting the full budget. Neither is a validated final optimum.

`coverage.json` records actual supervised samples delivered to successful
optimizer updates, unique photos, contributing annotation IDs, and per-dataset
photo coverage. Worker-prefetched samples do not count. A crop missing all
supervision falls back to an annotation-centered crop. A complete photo pass
does not guarantee every alternative observer was selected; annotation counts
make that distinction explicit. Aliased photos can appear in multiple dataset
coverage entries, so dataset totals need not sum to the global count.

`best.pt` is validation-selected and safe to load with `weights_only=True`;
`last.pt` additionally preserves optimizer, scheduler, all RNG states and actual
coverage for exact epoch-boundary continuation. Resume requires the original
manifest hash, configuration, executable training-source fingerprint and run directory. `--stop-after-epoch` is an
operational pause, leaving the planned cosine budget unchanged. New CLI code does
not migrate existing experiments across source revisions: older pinned training
checkouts keep running/resuming through their own unchanged sources, while
their checkpoints remain usable for prediction. `--max-steps`
is intended for functionality checks and may stop before photo coverage is
complete. Check the exported coverage rather than assuming completion.

Public `train-unified` accepts a flat `--config settings.json` or `settings.toml`
containing `UnifiedConfig` field names. Unknown fields and nested tables are
rejected. `--device`, `--purpose`, `--epochs`, `--epoch-samples`, and `--max-steps`
override config values. Resume restores the saved config and audited content pin
before applying explicit overrides; the trainer still rejects changes to the
planned budget/configuration. Only device and runtime stop-after-epoch can vary
under the existing continuation contract. Progress logs go to stderr; stdout is
the JSON command result. Optional `--expected-content-sha256` enforces an external
corpus pin. Full-data training validates the canonical audit even without that
option. Functionality checkpoints lacking a corpus pin cannot use public validation.

## Inference and evaluation

```bash
uv run deepretina predict \
  --checkpoint runs/unified-all-sources/best.pt --input fundus.png \
  --output predictions/example --dataset drive
```

The public checkpoint dispatcher preserves original and pilot model routes.
Unified prediction requires raw RGB and canonical25 pixels/degree geometry.
Provide `--fov` for a single image if available, `--dataset` or
`--field-of-view-degrees` for the camera. Folder inputs use per-photo automatic
FOV masks and cannot share one `--fov`. `--pixels-per-degree` must equal25.
Legacy prepared arrays/DICOM, raw AV proxy/MATLAB, vessel-input and `--normalized`
options are rejected for unified models.

Checkpoint `tile_size` and `overlap` apply by default. Explicit overrides are
validated, actually passed to tiled inference, and recorded with checkpoint
values in `prediction.json`. Whole-view models retain their trained whole-view
resize/padding policy and reject tiling overrides rather than ignoring them.
No new unified bundled aliases or final model weights accompany these commands.
The model and current quality remain experimental.

```bash
uv run deepretina evaluate --checkpoint runs/unified-all-sources/best.pt \
  --manifest data/fair-unified/manifest.json --split val \
  --output metrics/unified-validation.json
```

Public validation reuses the checkpoint config and requires exactly its manifest
SHA and audited prepared-content fingerprint, including after relocation. It
rejects unpinned functionality checkpoints and changed inputs before writing
metrics. `evaluate` defaults to test for older pilot routes, so **specify
`--split val`** for unified models. Public train/test split evaluation is blocked;
there is no file-hash shortcut for a frozen-selection record. A reviewed record
schema/protocol must be implemented before test evaluation becomes available.
Low-level evaluators remain research APIs and retain their documented ancestry
and selection obligations.

Outputs include original-size probabilities, five masks, an AV color map,
overlay and geometry metadata. `load_unified`, `predict_probabilities` (CHW RGB
in[0,1]), `restore_probabilities`, `decode_probabilities`,
`evaluate_unified_model`, and `read_unified_sample` support reusable tooling.
Evaluation reports pooled `mean_dice` and `macro_dataset_mean_dice` (mean supported dataset Dice per channel, then mean available channels). Dataset attribution follows each selected annotation bundle, including coannotated aliases. It also reports per-channel metrics and reference support, per-dataset/photo
results, gated AV accuracy and conditional AV accuracy on known class-labeled
vessel pixels. Per-channel AV Dice uses that same known-vessel support; it does
not measure all background false positives. Test evaluation belongs after model
selection; historical teacher exposure and previously viewed tests must be
reported explicitly. A future legacy warmstart is a separate ancestry-qualified
alternative, not the default compact baseline.

## Validation selection and full-data search

The evaluator reports pooled `mean_dice` and `macro_dataset_mean_dice` separately.
`selection_metric` chooses which validation score selects `best.pt`; the default
`mean_dice` preserves functionality-run compatibility. Full-data search plans
explicitly select `macro_dataset_mean_dice`, giving each supported dataset equal
weight within an output. Checkpoints/CSV retain both scores and the selection
score. Resume requires the same declared metric. See [UNIFIED_HPO.md](UNIFIED_HPO.md)
for audited17-source planning, fixed scheduler budgets and commands.

## Controlled original-topology alternative

`architecture="legacy_universal"` selects the original universal linear-width
UNet (32/64/96/128, depth4, BatchNorm, no dropout), expanded to five outputs:
1,304,421 parameters. Its widths/depth/norm are fixed; the compact architecture's
`base_channels`, `depth`, and `norm` settings do not change this alternative.
Actual architecture metadata is exported separately from the requested config.
From-scratch initialization uses no historical weights and remains eligible for
an independently partitioned new-model experiment. It has not been GPU trained
or demonstrated to improve accuracy.

Optional historical initialization is deliberately separate and exploratory:

```toml
architecture = "legacy_universal"
legacy_initialization = "src/deepretina/pretrained/legacy-single-net-universal.pt"
allow_historical_exposure = true
```

This accepts only the safe converted original universal checkpoint. Its vessel,
disc and cup head parameters are copied exactly to outputs0,3,4; conditional
artery/vein weights and biases start at zero (uniform conditional probability).
The original backbone and BatchNorm state are copied without alteration. Tests
compare the actual distributed checkpoint's three initial logits bit for bit on a divisible rectangular input and its original288×288 tile. Very small padded inputs can differ by floating-point rounding because the final convolution has five rather than three output channels; tests enforce close agreement there.

Historical teacher exposure to current splits is **unknown**. Such descendants
are marked `independent_model_eligible=false`, carry `accuracy_claim=false`, and
may be evaluated only on train/validation for exploratory analysis; the unified
evaluator refuses their test evaluation. Explicit authorization acknowledges
this limitation and does not make the initialization eligible for the final
independent model. Keep it outside the main unified HPO candidate plan. Lineage
includes original source provenance, converted-checkpoint SHA256, training-index
hash/count and the output mapping. Original labels may also contain known
historical defects, including UOA cup semantics; ancestry is no quality proof.

Student checkpoints remain self-contained: inference and epoch-boundary resume
restore all five heads plus lineage and never reopen/download the historical
checkpoint. The original checkpoint path is provenance, not a runtime dependency.
For the new student only, an input wrapper fixes height/width padding for odd,
rectangular and very small images before delegating to the unchanged original
forward. The historical reference implementation and golden behavior stay intact.

## Report preservation and checkpoint identity

Public unified validation refuses an existing output file before loading the model
or evaluating, and publishes with exclusive creation to preserve even a report
created concurrently. Choose a new metrics filename for a new measurement.
Prediction and validation include `checkpoint_sha256` of the exact immutable byte
snapshot loaded with `weights_only=True`; replacing a live `best.pt` during
inference cannot change that report's checkpoint identity.

Each `per_photo` entry includes `av_conditional_counts`: `correct`,
`scored_pixels`, `eligible_pixels`, and `end_to_end_correct`. Counts use the same
supported known-class reference-vessel mask as the header. Unified conditional
classification is scored even where the vessel detector misses a reference
vessel, so conditional correctness can exceed end-to-end correctness. Per-photo
correct/scored sums reconstruct header `av_conditional_accuracy`, while
end-to-end/eligible reconstructs `av_accuracy`. These counts support paired
uncertainty analysis without changing decoding or training objectives.
