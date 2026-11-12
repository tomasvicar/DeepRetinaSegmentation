# Selected raw512 unified model

`unified-raw512-selected` is a separately named CPU inference export of the
completed curated raw512 run's **BEST epoch50**. Its own checkpoint covers all
2,529 TRAIN photographs and 5,700 annotation records from21 sources. Training
completed60 epochs; this selected artifact is not the terminal checkpoint.
The compact UNet has base16/depth4/batch normalization and five outputs.

```bash
uv run deepretina predict --model unified-raw512-selected \
  --input fundus.png --dataset drive --fov official-fov.png --output predictions/unified-cli
```

```python
import numpy as np
from PIL import Image
from deepretina import predict_bundled_unified

with Image.open("official-fov.png") as image:
    fov = np.asarray(image.convert("L")) > 127
predict_bundled_unified(
    "unified-raw512-selected", "fundus.png", "predictions/unified-api",
    dataset="drive", fov=fov,
)
```

Both routes inherit the frozen **rot4 / threshold0.35 / tile512 / overlap64**
protocol when those four options are omitted. Prediction uses FP32 without
mixed-precision autocast, saved raw `legacy25_raw` geometry at nominal25px/degree,
and joint A/V probability fusion before conditional decoding. Supply the correct
camera profile and official FOV where available; nominal geometry is not clinical
calibration. Explicit inference overrides are a different protocol.

On770 VAL photographs the selected protocol's macro Dice is0.82283410 and
conditional A/V accuracy0.85119431. Dense-vessel Dice is**0.75131596** versus
original**0.76132681**, on87 photographs/55known groups/8sources. The paired
95% interval for the difference is[−0.01435434,−0.00656358]; precision and recall
are also lower. [Exact current6808 saved-count verification](../reports/release/raw512-current6808-dense-20261112/README.md)
confirms this interval with2,000 source-stratified known-group draws/seed42.
Original vessels remain preferable for the measured vessel
priority. Native V3 has a stronger artery result. This is a balanced alternative,
not an all-task replacement, a global optimum or a clinical claim. Exploratory
VAL intervals do not establish complete patient independence. The subsequent frozen TEST is reported below.

## Frozen TEST on856 photographs

Both original and unified executions and their outer observer completed with
exit0 at14:17:29UTC on12November2026. The same856 photos/annotation records/
source attribution/GT support were checked; no model selection or tuning followed.
The exact raw512 protocol was frozen before execution.

| Metric | Original | Selected unified |
| --- | ---: | ---: |
| Vessel pooled Dice | 0.787840 | 0.786565 |
| Artery pooled Dice | 0.695384 | 0.784665 |
| Vein pooled Dice | 0.706572 | 0.832444 |
| Disc pooled Dice | 0.892788 | 0.946856 |
| Cup pooled Dice | 0.454491 | 0.885680 |
| Conditional A/V accuracy | 0.751861 | 0.860869 |
| End-to-end A/V accuracy | 0.641228 | 0.752940 |
| Dataset/head macro Dice | 0.681858 | 0.839614 |

Artery, vein, disc and cup pooled differences have positive paired95% intervals.
Pooled-vessel difference−0.001275 has an interval crossing zero; this is not
evidence of equivalence. The dense-vessel supplement excludes only positive-only
AVRDB:102photos/71known groups, original**0.780768** versus unified**0.776548**,
difference**−0.004220**, paired95% interval**[−0.006315,−0.002161]**. Both precision
and recall are lower. Original vessels remain the recommended measured vessel
reference. Higher macro/other heads do not remove this remaining deficit.

Intervals use2000 source-stratified group draws, seed42 (696known TEST groups
overall). Known groups do not prove complete patient independence; historical
exposure of original weights is unknown. Full per-source results and adverse
subgroups are retained in the TEST report. The selected original521b was tested;
the CPU deployment292e has separate exact TRAIN integration evidence.

[Full TEST report, source results and exact count/exit evidence](../reports/release/selected-raw512-TEST-20261112/README.md)
retain adverse subgroups and all precision values. Root independently rechecked
the two original report hashes, both exits,856-photo support and all five pooled
confusion totals.

The selection binds original checkpoint SHA
`521b8881d1b2c69a7395be6687a8b86581838e3f5dbb004a150333d454132cea`
and selection SHA
`caef7716a5656a151cf6c0ce1e9b8cd694aa9f13620cef36f14abcd0a6368e80`.
The packaged CPU copy has the distinct artifact SHA
`292ea5999a1fc97a767c1c34bb7b4558040267372c5b128f403f2dce50529b29`.
All model tensor names/dtypes/shapes/bytes and inference/configuration metadata
are preserved; only CPU storage copies were made. It is marked inference-only,
not a training-resume checkpoint. This BEST already contained no optimizer fields;
the small byte-size change is not parameter compression. Formal selection/TEST
continues to use the original bytes, never this renamed artifact.

All12 prior packaged files, entries and aliases remain available unchanged.
In particular `vessels`, `av` and `disc_cup` retain their original models.
See [export and verification evidence](../reports/release/unified-raw512-selected-20261112/README.md),
[packaging rules](FINAL_MODEL_PACKAGING.md) and
[the frozen selection](../reports/release/raw512-selection-20261112/README.md).

[Paired GPU timing on two fixed TRAIN inputs](../reports/release/selected-raw512-two-TRAIN-GPU-timing-20261112/README.md)
measures the original selected checkpoint with this exact rot4/.35 protocol,
not the separately exported package file. Resident pipeline time and model
loading are reported separately, with input IO excluded. Two images do not
establish general speed.
