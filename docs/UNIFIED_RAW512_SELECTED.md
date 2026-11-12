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
VAL intervals do not establish complete patient independence. **Held-out TEST
was pending at packaging**; no TEST result is claimed here.

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
loading are reported separately; two images do not establish general speed.
