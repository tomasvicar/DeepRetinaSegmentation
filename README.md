# DeepRetinaSegmentation · Leipzig

**Status:** original trained models are bundled with verified prepared-input
inference. Three additional single-dataset pilots remain explicitly preliminary.
A completed corpus-v1 unified baseline is bundled (17 TRAIN sources, 2,221 photos); model optimization and final comparison remain in progress.
The agreed deliverable is recorded in [GOAL.md](docs/GOAL.md).

A Python package for retinal **vessel segmentation**, **artery/vein classification**,
and **optic disc/cup segmentation**, with one workflow from dataset ZIPs to predictions.
The branch preserves five original trained checkpoints and three explicitly named
pilot checkpoints. Original architectures, weights and prepared-input inference
have been checked numerically against the historical implementations. The newer
pilots are small single-dataset experiments; they are not replacements for the
original models. The separately named full-data unified baseline is also ready for inference.
See [the project goal and completion criteria](docs/GOAL.md).

## Install

Check out `for_Leipzig`, then use [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv sync --frozen --dev
uv run deepretina doctor
uv run deepretina --help
```

The lockfile uses PyTorch 2.6 with CUDA 12.4 on Linux, compatible with the GTX 1060/1070
and RTX 4070 Ti SUPER used here. A CUDA-capable driver is needed for GPU training;
`--device cpu` works for development. `uv` creates the environment and installs dependencies.
No encoder weights or data are downloaded by the application.

## Predict immediately with included models

The models are part of the repository and Python package. You do not need
access to the author's ownCloud, training data or checkpoint directories:

```bash
uv run deepretina models
uv run deepretina predict --model legacy-vessels --input /path/to/fundus.jpg --output predictions/vessels
uv run deepretina predict --model legacy-universal --input /path/to/fundus.jpg --output predictions/disc_cup
```

Add `--device cpu` to work without a GPU and `--dataset drive` (or the correct
source name) to select source-specific geometry. For an unknown source, supply
`--field-of-view-degrees` explicitly. The original single-net default uses the
source-inferred raw profile; `--normalized` selects its original local-contrast
profile. The historical training choice is not recoverable from the weights.

The original A/V network classifies an existing vessel segmentation. With the
original prepared four-plane NumPy input:

```bash
uv run deepretina predict --model legacy-av --prepared-array input.npz --output predictions/av
```

For a raw photograph, explicitly opt into the Python approximation of the old
MATLAB preprocessing. Without `--vessel-mask`, this runs the original vessel model
first:

```bash
uv run deepretina predict --model legacy-av --av-raw-proxy \
  --input /path/to/fundus.jpg --dataset drive --output predictions/av
```

For the separately calibrated MATLAB-style Python profile, provide the original
FOV mask. Its generic center/radius calibration is computed in Python. A known
`--dataset` or explicit positive
`--field-of-view-degrees` is required:

```bash
uv run deepretina predict --model legacy-av --av-raw-matlab \
  --input /path/to/fundus.jpg --fov /path/to/fov.png \
  --dataset drive \
  --output predictions/av --device cpu
```

An optional `--fov-diameter-pixels 538` supplies an explicit measured diameter
and takes priority over estimation (538 belongs only to the measured DRIVE
reference photograph). Add `--vessel-mask /path/to/vessels.png` to classify
supplied original-frame vessels; otherwise the original vessel model runs first.
The two raw profile flags are mutually exclusive. Empty/full or unusable FOVs
fail closed when estimating; RIM-ONE, Erlangen, ROC and RIGA require explicit
calibration because their original source uses special branches. These errors
are rejected before vessel inference or output creation. See the
[provided-FOV calibration scope](docs/MATLAB_FOV_CALIBRATION.md). Runtime remains entirely Python/uv; MATLAB
scripts in the repository only generate test references.

[The calibrated profile evidence](docs/MATLAB_AV_PROFILE.md) records numerical
residuals and a synthetic counterexample. Actual saved-model checks on two real
DRIVE/HRF inputs had **zero vessel-gated class disagreements**; this limited
result does not establish universal parity or historical training-file provenance.

Both raw A/V modes are **not certified pixel-equivalent to MATLAB**.
Prepared-input model arithmetic is verified separately. See
[original model details](docs/LEGACY_MODELS.md), [A/V input contracts](docs/LEGACY_AV.md)
and [preprocessing evidence](docs/LEGACY_PREPROCESSING.md).

| Included model | Provenance / outputs |
| --- | --- |
| `legacy-universal` | Original trained single_net: vessel, disc, cup |
| `unified-full-data-baseline` | Corpus-v1 completed60-epoch five-output baseline (17 TRAIN sources, 2,221 photos); selected E56 on validation, no TEST yet |
| `legacy-vessels` | Original trained vessel network |
| `legacy-disc-0`, `legacy-disc-1` | Two original trained disc networks |
| `legacy-av` | Original trained A/V classifier; requires vessel support |
| `pilot-vessels` | New DRIVE-only pilot; test vessel Dice 0.7988 |
| `pilot-av` | New RITE-only pilot; labeled-vessel class accuracy 0.7219 |
| `pilot-disc-cup` | New Drishti-only pilot; disc/cup Dice 0.8798/0.7648 |

Aliases `vessels`, `disc_cup` and `av` select the corresponding original models.
Files are included in the package and checked against SHA256 before loading.
Custom checkpoints use `--checkpoint path/to/best.pt`. Outputs include masks,
probabilities, overlay and provenance metadata at the input photograph geometry.
The pilot metrics are not a controlled comparison with the original models;
[EXPERIMENTS.md](docs/EXPERIMENTS.md) records their limited scope. Dataset photographs
and ground-truth masks are not distributed.

## Prepare data

The team shares the `fundus_databases` folder through ownCloud. Set `--data-root` to
your local copy of that folder (on the author's machines,
`/var/home/tom/ownCloud/fundus_databases`). Keep source archives outside Git.
Read them directly, without extracting all archives:

```bash
uv run deepretina prepare \
  --data-root /path/to/ownCloud/fundus_databases \
  --datasets drive rite drishti \
  --output data/leipzig --max-size 1024
uv run deepretina inspect data/leipzig/manifest.json
```

All **31 historical dataset selectors** are migrated. List their source paths, tasks and
validation scope with `uv run deepretina datasets`. This includes the original eight
supervised sources plus ARIA, AVRDB, G1020, RIGA, RIM-ONE, STARE, UoA-DR, Erlangen
and the auxiliary/image-only collections. UBMI accepts an external archive or directory;
its private source is not included.
V2 additionally decodes native DRIONS, GAMMA and `idrid_disc` annotations;
BIOMISA remains quarantined unless its audited export-role policy is explicit.
The [V2 append preparation guide](docs/CORPUS_V2_PREPARATION.md) describes the
verified 21-source TRAIN corpus (2,529 photos) and independent retraining path.
See [portable geometry and release scope](docs/V2_RELEASE_INTEGRATION.md).
See [dataset formats and annotation rules](docs/DATASETS.md) before mixing databases.
For a bounded integrity preview, add `--limit 3`; this limits source groups per dataset
and marks the output as a preview. It is not the complete training corpus. Override a
source location with `--source ubmi=/path/to/private/data`.
Preparation preserves official test sets, groups shared DRIVE/RITE images and paired subjects,
and creates validation only from training data. Prepare related datasets **together**.

Prepared images retain aspect ratio and are at most `--max-size` pixels on their longest side.
Categorical labels use nearest-neighbour resizing. The manifest and NPZ files describe
per-channel labels and which pixels are actually annotated; unknown A/V labels are ignored.
Use a new or empty output directory for each preparation; the manifest contains exactly the
requested datasets. Existing prepared data are never overwritten.

## Optional source FOV for raw original A/V

`--av-raw-matlab-source` explicitly enables the Python source FOV/photometry
profile with automatic FOV or a supplied source-compatible `--fov`. It accepts the
existing raw A/V vessel/camera options. Numerical photometry residuals and measured
counterexamples remain; it is not certified MATLAB parity and changes no default.
See [scope, examples and limitations](docs/MATLAB_SOURCE_AUTO.md).

## Train and use one unified model

The Python pipeline can train one RGB network for vessel, artery/vein, disc and
cup, using partial labels and all suitable annotated sources. New full-data model
selection is still in progress; the bundled original models remain available.
See the [complete unified quickstart](docs/UNIFIED_QUICKSTART.md) for canonical
data preparation, patient/photograph split audit, training, resume and prediction.
A colleague can rebuild all 21 training sources directly from shared RAW archives with
`scripts/prepare_training_data.py`; no author-prepared parent is needed. See the
[retraining scope](docs/RETRAINING.md) for the original-topology option and native
architecture limitations.
The public entry point is `uv run deepretina train-unified`; a starting preset is
`configs/unified.toml`. Validate a checkpoint with `deepretina evaluate --split val`
and predict with `deepretina predict --checkpoint ...`. Experimental unified
test evaluation remains blocked until final selection is frozen and reviewed.

## Train the pilot architecture

Use a new output directory for each run:

```bash
uv run deepretina train --manifest data/leipzig/manifest.json \
  --config configs/vessels.toml --output runs/vessels
uv run deepretina train --manifest data/leipzig/manifest.json \
  --config configs/av.toml --output runs/av
uv run deepretina train --manifest data/leipzig/manifest.json \
  --config configs/disc_cup.toml --output runs/disc_cup
```

TOML presets reproduce the measured 30-epoch pilot settings; they are starting points
for further validation. Explicit options override the preset:
`--epochs 2 --max-steps 10 --base-channels 8 --patch-size 128` makes a short smoke run.
`--max-steps` limits the **total** optimizer steps, not steps per epoch. CUDA defaults to the
first visible GPU; use `CUDA_VISIBLE_DEVICES=0` or `--device cuda:1` to select a device.

Training uses aligned random crops, rotations and flips. Validation uses complete images
with overlapping tiles and a fixed threshold of 0.5. `best.pt` is selected using validation
mean Dice; `last.pt` records the last epoch. `run.json`, `metrics.csv` and `validation.json`
record settings, manifest hash and scores. Checkpoints store state dictionaries and primitive
metadata, and load with `weights_only=True`. Original weights are distributed in a separate safe, converted format. Unrecoverable source annotations remain
in the manifest with quality notes and `usable_for_training=false`; training excludes them.
Disc-only sources never train the absent cup. Channel-supervision flags travel with the
checkpoint; unsupported binary channels are all-zero placeholders explicitly marked in
prediction metadata and the probability NPZ.

Independent output channels are:

| Task | Channels |
| --- | --- |
| `vessels` | vessel |
| `av` | vessel, artery, vein |
| `disc_cup` | disc (including cup), cup |

Presets use validation-selected losses from the pilot: BCE for A/V and BCE+Dice for
vessels and disc/cup. `--loss` allows controlled comparisons.
Select losses and settings on validation; evaluate the final selected configuration on test.

## Evaluate and predict pilot checkpoints

```bash
uv run deepretina evaluate --checkpoint runs/av/best.pt \
  --manifest data/leipzig/manifest.json --split test --output runs/av/test.json
uv run deepretina predict --checkpoint runs/av/best.pt \
  --input /path/to/fundus.jpg --output predictions/example
```

`--input` also accepts a directory of images. Predictions contain named binary PNG masks,
`probabilities.npz`, a colour overlay and `prediction.json`. Raw probabilities remain available
for downstream analysis. Binary outputs enforce cup within disc and resolve artery/vein
conflicts within the vessel mask. Outputs retain the input image dimensions. The checkpoint
records preparation scale so raw images are resized consistently before inference and restored
to original geometry afterward. Tiled inference bounds GPU memory; tune `--tile-size` and
`--overlap` consistently for comparisons.

Evaluation reports channel Dice, IoU, precision/recall, counts and per-image scores within valid
annotated pixels. Undefined empty-class scores are `null`. AV scores distinguish classification
of annotated vessels from missed vessel detection. Prepared-resolution scores are not directly
comparable with historical original-resolution or differently masked evaluations.

## Development and evidence

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

- [Repository and branch map](docs/REPOSITORY_MAP.md): original code, checkpoints and defects.
- [Migration notes](docs/MIGRATION.md): what changed and what remains unsupported.
- [Datasets](docs/DATASETS.md): encodings, splitting and all migrated adapters.
- [Migration audit](docs/ADAPTER_MIGRATION_AUDIT.md): exact historical loader coverage.
- [Supervised additions](docs/ADAPTERS_SEGMENTATION.md) and
  [auxiliary collections](docs/ADAPTERS_AUXILIARY.md): real-data checks and label caveats.
- [Experiments](docs/EXPERIMENTS.md): real runs, machines and measured results.
- [Work log](docs/WORK_LOG.md): investigation and implementation decisions.

The original model ports preserve historical architecture and inference. Their
prepared-input numerical checks do not establish new quality scores or resolve
all historical preprocessing uncertainty. Pilot and unified-model quality claims
require controlled comparisons. Generic `evaluate` handles pilot checkpoints and pinned unified validation.
Original-model evaluation needs the matching geometry and annotation protocol
and is not silently routed through another model family's preprocessing.

Hyperparameter search runs, candidate results and new-model selection are tracked on
`experiments/leipzig-hpo`. The verified training/prediction pipeline is included here;
experimental results do not replace the bundled original models.
