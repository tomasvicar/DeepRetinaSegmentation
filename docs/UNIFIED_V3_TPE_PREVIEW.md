# Unified TPE preview: completed run, BEST E58

`unified-v3-tpe-preview` is the **first completed native TPE proposal**, not the
subsequent running proposal. It is an experimental validation-selected preview,
not the final HPO winner or a TEST-qualified replacement for the original models.

```bash
uv run deepretina predict --model unified-v3-tpe-preview \
  --input /path/to/fundus.jpg --dataset drive \
  --output predictions/tpe-preview --device cpu
# Optional exploration: four rotated views, with approximately four times the work.
uv run deepretina predict --model unified-v3-tpe-preview \
  --input /path/to/fundus.jpg --dataset drive \
  --output predictions/tpe-preview-rot4 --device cpu --tta rot4
```

Choose the actual source/camera, or supply `--field-of-view-degrees` for unknown
sources. Use a new/empty output directory. Default TTA remains `none`; the original
model aliases and all eleven previously bundled checkpoints are unchanged.

The compact UNet has 1,965,637 parameters and five outputs: vessel, conditional
artery/vein, disc and cup. Its raw 25-pixels/degree RGB pipeline uses one shared
network; outputs return to the original photograph geometry. Inference is FP32,
including on CPU and Pascal GPUs. Optional TTA averages aligned joint vessel/class
probabilities before decoding, as described in [UNIFIED_TTA.md](UNIFIED_TTA.md).

## Python API

Run this with `uv run python` after installation. This uses the same
[DRIVE TRAIN photo/FOV pair](PREDICT_ORIGINAL.md#obtain-a-matching-photo-and-fov)
as the original pipeline. Unlike the CLI's FOV path, this API takes a binary
**array** for `fov`. Use a separate new/empty output directory.

```python
from pathlib import Path
import numpy as np
from PIL import Image
from deepretina.bundled import bundled_checkpoint
from deepretina.unified import predict_unified

with Image.open("inputs/drive21/21_training_mask.gif") as mask:
    fov = np.asarray(mask.convert("L")) > 0
with bundled_checkpoint("unified-v3-tpe-preview") as checkpoint:
    outputs = predict_unified(
        checkpoint, Path("inputs/drive21/21_training.tif"),
        Path("predictions/drive21-tpe-api"),
        dataset="drive", fov=fov, device="cpu", tta="none",
    )
```

The bundled checkpoint checksum is verified before loading. The function exports
masks, probabilities, overlay and metadata; its return value describes the output
files. The saved photometric profile is applied automatically: do not pre-normalize
or apply CLAHE to the input photograph.

## Training and completion provenance

The run completed 60 epochs on corpus V2: **21 TRAIN sources, 2,529 photos and all
5,700 annotation variants**. Its validation-selected BEST is E58 and independently
covers all 2,529 photos and 5,700 variants (minimum visits 60 and 1 respectively).
This is distinct from claiming the BEST weights are the terminal E60 weights.
The original standalone completion QA and both fresh validations are pinned in
[the release provenance](../reports/release/unified-v3-tpe-preview/provenance.json)
and [completion summary](../reports/release/unified-v3-tpe-preview/original-completion-summary.json).

The bundled checkpoint is byte-for-byte the original BEST58, SHA256
`20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`,
8,894,002 bytes. It already contains no optimizer/training state; it was neither
repacked nor quantized. File size alone is not evidence of model compression.

[The recipe](../configs/unified-v3-tpe-preview.toml) matches the actual first TPE
configuration, including BF16 training, except `device="auto"` replaces the
machine-specific `cuda:0`. It does not use the six parameters of the later proposal.

```bash
uv run deepretina train-unified --manifest /prepared/corpus-v2/manifest.json \
  --output runs/tpe-retrain --config configs/unified-v3-tpe-preview.toml
# CPU or GTX 1060/1070 training: explicit FP32 override (slower, different arithmetic).
uv run deepretina train-unified --manifest /prepared/corpus-v2/manifest.json \
  --output runs/tpe-retrain-fp32 --config configs/unified-v3-tpe-preview.toml \
  --device cpu --mixed-precision none
```

Exact historical scores are not promised across devices or precision changes.
This training recipe requires the prepared corpus; prediction needs no training
data. No encoder weights or datasets are downloaded by prediction.

## Measured validation and cost, with separate scopes

Fresh Ada validation without TTA on all 770 VAL photographs produced macro
source/head Dice **0.8108954**, vessel Dice **0.7628822** and conditional A/V accuracy
**0.8249115**. Original reference vessel Dice is **0.7701234**, so vessel quality
still regresses. Historical exposure and preprocessing/backend differences limit
comparison with the original reference; these are interim VAL results, not TEST.

A separate matched Old-GTX1060 study measured all modes on the same 770 photographs
and fixed threshold 0.5. Its `none` macro is 0.8108974, slightly different from Ada;
no global bitwise parity is claimed. `rot4` macro is **0.8180552** and `dihedral8`
**0.8188538**. TTA has adverse source/head points: RIM-ONE cup drops approximately
0.00418 for `rot4` and 0.00252 for `dihedral8` relative to its same-device `none`;
both paired intervals cross zero. Eight views also have slightly worse pooled cup
Dice than four views. These exploratory, adaptive VAL comparisons do not establish
patient-independent or universally improved performance.

| Pipeline/mode | Resident CPU median seconds per photograph |
| --- | ---: |
| Composed original four-network reference | 59.59 |
| TPE `none` | 8.75 |
| TPE `rot4` | 33.20 |
| TPE `dihedral8` | 65.53 |

These are one raw DRIVE **TRAIN** photograph with official FOV, CPU affinity one
core/thread, FP32, one warmup and three repetitions; they exclude model loading
and file I/O. Eight views are slower than the original reference in this example.
Do not attach the 8.75-second `none` cost to eight-view quality. This is not a GPU
speed or population latency claim. Full evidence and uncertainty:
[TTA VAL results](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/tpe72-inference-tta-val),
[matched CPU costs](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/benchmark/tpe72-tta-cpu-drive21).

`--tta rot4` is an optional quality/cost exploration, not a changed default or a
final selection. HPO continues; no held-out TEST was used for this release.

## Optional scalar threshold calibration

A later eight-threshold sweep on the same BEST58/rot4 model and all770 VAL
photographs selected **0.35** by the minimum of six pooled ratios against the
original reference. It reused the same probabilities for every threshold;
conditional A/V accuracy therefore stayed unchanged. To try that measured recipe:

```bash
uv run deepretina predict --model unified-v3-tpe-preview \
  --input /path/to/fundus.jpg --dataset drive --fov /path/to/fov.png \
  --tta rot4 --threshold 0.35 --device cpu \
  --output predictions/tpe-preview-rot4-threshold035
```

Use the photograph's actual camera/dataset and matching original-frame FOV.
This exploratory setting achieved pooled vessel Dice **0.7694639**, macro
source/head Dice **0.8220992** and conditional A/V accuracy **0.8361133**.
Vessel Dice still trails original0.7701234 by−0.0006595, with paired95% interval
[−0.0065302,+0.0055169]; this establishes neither superiority nor equivalence.
Some source/head results regress. Defaults remain threshold0.5/TTA`none`; this
does not make the preview a final or TEST-qualified model. The CPU timings above
were measured at0.5 and are not a new timing measurement of0.35.

These all-source vessel scores pool107 annotated VAL photos, including20 AVRDB
photos with **positive vessel support only**: background false positives cannot
be scored there. On the remaining87 densely supported photos/55 known groups,
original vessel Dice is **0.7613266**, while this BEST58/rot4/.35 pipeline scores
**0.7482361**. The difference−0.0130905 has paired95% interval
[−0.0185577,−0.0087907]. The close all-source point estimates therefore conceal
a clear dense-domain vessel regression. See the
[support-specific analysis](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/dense-vessel-support-20261112)
for counts, source results and historical exposure/backend limitations.

The [full threshold report](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/first-TPE-BEST58-rot4-threshold-20261111)
retains all eight settings, counts, source regressions and uncertainty. The
[all-head comparison](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/all-head-candidate-comparison-20261111)
keeps the original and other completed candidates separate.
