# DeepRetinaSegmentation · Leipzig

A Python package for retinal **vessel segmentation**, **artery/vein classification**,
and **optic disc/cup segmentation**, with one workflow from dataset ZIPs to predictions.
The `for_Leipzig` branch trains a compact RGB U-Net from scratch. It needs neither MATLAB
nor the unavailable pretraining corpus. Historical implementations remain in Git history.

## Install

Use [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv sync --frozen --dev
uv run deepretina doctor
uv run deepretina --help
```

The lockfile uses PyTorch 2.6 with CUDA 12.4 on Linux, compatible with the GTX 1060/1070
and RTX 4070 Ti SUPER used here. A CUDA-capable driver is needed for GPU training;
`--device cpu` works for development. `uv` creates the environment and installs dependencies.
No encoder weights or data are downloaded by the application.

## Prepare data

Keep licensed source archives outside Git. Read them directly, without extracting all archives:

```bash
uv run deepretina prepare \
  --data-root /var/home/tom/ownCloud/fundus_databases \
  --datasets drive rite drishti \
  --output data/leipzig --max-size 1024
uv run deepretina inspect data/leipzig/manifest.json
```

Available adapters: `drive`, `rite`, `hrf`, `drhagis`, `chasedb1`, `iostar`, `refuge`, `drishti`.
See [dataset formats and annotation rules](docs/DATASETS.md) before mixing databases.
Preparation preserves official test sets, groups shared DRIVE/RITE images and paired subjects,
and creates validation only from training data. Prepare related datasets **together**.

Prepared images retain aspect ratio and are at most `--max-size` pixels on their longest side.
Categorical labels use nearest-neighbour resizing. The manifest and NPZ files describe
per-channel labels and which pixels are actually annotated; unknown A/V labels are ignored.
Use a new or empty output directory for each preparation; the manifest contains exactly the
requested datasets. Existing prepared data are never overwritten.

## Train

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
metadata, and load with `weights_only=True`. Historical pickled models are a different format.

Independent output channels are:

| Task | Channels |
| --- | --- |
| `vessels` | vessel |
| `av` | vessel, artery, vein |
| `disc_cup` | disc (including cup), cup |

Presets use validation-selected losses from the pilot: BCE for A/V and BCE+Dice for
vessels and disc/cup. `--loss` allows controlled comparisons.
Select losses and settings on validation; evaluate the final selected configuration on test.

## Evaluate and predict

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
- [Datasets](docs/DATASETS.md): encodings, splitting and supported adapters.
- [Experiments](docs/EXPERIMENTS.md): real runs, machines and measured results.
- [Work log](docs/WORK_LOG.md): investigation and implementation decisions.

This cleanup changes architecture and preprocessing; it does not establish superiority over
published or legacy checkpoints. A runnable smoke test demonstrates functionality, while
claims about model quality require controlled held-out comparisons. No dataset files or model
weights are committed to this repository.
