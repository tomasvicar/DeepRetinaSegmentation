# Original artery/vein model

This is a numerical port of the trained `Ves_Class` v9 model, rather than a newly
trained substitute. The safe checkpoint is `models/legacy-av-v9.pt`; conversion
and numeric evidence are recorded in
[`av_conversion.json`](../reports/legacy/av_conversion.json).

The owned original is
`fundus_databases/models/Ves_Class_AV_UNet/net_v0_0_9.pt`, 8,159,748 bytes,
SHA256 `75c5048aed6adb9f783f8fabb056185c4eb81c23be45b3a152794f59dc8cb1c0`.
Its architecture is `Network.Net` from `artery_vein_class` commit
`7bb0539b90dfce2ed6eaeef1de812072be1155d0`. The filename v9 does **not** identify
the attention network in `Network_v9.py`. Both owned v8 and v9 are recognized by
SHA256; the v9 artifact is independently checked here.

The residual U-Net has encoder widths 4→32→64→128→256, decoder
256→128→64→32, heads 32→128→3, and 2,030,379 parameters. Blocks normalize inputs
with BatchNorm, apply two convolutions and a residual addition. The model retains
the original transposed convolutions, center cropping with round-half-to-even,
and nearest final upsampling. It is implemented with PyTorch alone.

## Input and prediction contract

The input is one already-preprocessed image with four channels: RGB and an
existing binary vessel segmentation in the same pixel frame. This model
classifies supplied vessel pixels; it does not independently segment vessels.
The vessel mask can come from the original universal vessel model or an explicit
manual mask. Keep that provenance when interpreting predictions.

Historical inputs were at 25 pixels/degree with local contrast normalization and
CLAHE. The model itself computes each image/channel's global mean and sample
standard deviation, then divides without an epsilon. No additional RGB offset,
scale, clipping or per-tile normalization is applied by this port. DICOM input
preserves stored pixel values, including its vessel-channel scale; rescale slope
and intercept are not part of the old `pixel_array.astype(float32)` path.

A linear RGB rescale by 16, without new quantization, gives identical logits in
the numeric check. This does not make 8-bit and 12-bit quantization equivalent:
rounding to 255 and rounding to 4096 can produce different normalized values.
Likewise, similar CLAHE implementations are not automatically pixel equivalent.

The library exposes:

```python
from deepretina.legacy.av import load_av, predict_av_input

model, metadata = load_av("models/legacy-av-v9.pt", device="cpu")
# inputs: NumPy float32 [4,H,W] or Torch [1,4,H,W], already prepared.
result = predict_av_input(model, inputs, device="cpu", original_size=(height, width))
```

`prepare_input(rgb_hwc, vessel_hw)` and `predict_av(model, rgb, vessel, ...)`
provide stored-value stacking convenience. `predict_av_input` integrates directly
with `load_vesclass_dicoms` or the preprocessing module's four-channel stack.
Height and width must be at least 16. Constant channels, including an empty or
full vessel mask, make the original standardization undefined; the prediction
wrapper reports that condition rather than emitting NaN predictions. The model's
forward method itself retains the original mathematical behavior.

Outputs are:

- `prepared_logits`: three original logits at prepared geometry.
- `prepared_conditional_probabilities`: two-channel artery/vein softmax.
- `prepared_classes`: 0 background/tie, 1 artery, 2 vein at prepared geometry.
- `classes`: the same class IDs restored by nearest interpolation when requested.

The historical loss uses only logits 1 and 2; its background logit is not used
for classification. Decoding compares artery/vein logits strictly, preserves ties
as background, and removes pixels where the supplied vessel channel is zero.
A three-class softmax would change this contract.

Original restoration uses direct nearest resizing of a uniformly scaled **full
frame**, matching OpenCV `INTER_NEAREST`. If a caller supplies a cropped input,
it must invert that crop geometry explicitly rather than stretching it across
the original photograph. Full-image inference is required: patch inference changes
both global channel statistics and context.

## Safe conversion and verification

`convert_av(source, destination)` accepts the known owned v8/v9 artifact hashes
by default. It loads original module state using `torch.load(weights_only=True)`
and explicit aliases for five `Network` container classes and the exact basic
PyTorch layer types. Historical imports and arbitrary pickle globals are not
executed. The output contains primitive metadata and a state dictionary, so normal
prediction requires neither the old module names nor an unsafe pickle loader.
An explicit `require_known=False` supports a separately verified owned artifact
without relaxing the pickle-global allowlist. Unknown original v1–v7 forward
semantics are not silently certified by the v9 parity results.

The pinned original `Network.py` is preserved unchanged under
[`tests/reference/av`](../tests/reference/av). Tests extract only its five model
classes into an isolated reference namespace. An independent CenterCrop adapter
was checked against actual torchvision 0.21.0; torchvision is not a runtime
requirement. All saved tensors and CPU logits match exactly for input geometries
31×35, 63×77, 64×80 and 129×173. The maximum absolute error is zero.
Nearest class restoration is also compared directly with OpenCV, and a forbidden
pickle-global test verifies that arbitrary deserialization callbacks do not run.

Run the normal checks with:

```bash
uv run pytest -q tests/test_legacy_av.py
```

The original owned checkpoint is required for the original-pickle numeric test;
set `DEEPRETINA_AV_CHECKPOINT=/path/net_v0_0_9.pt` on another machine.
The independently executed torchvision reference environment used:

```bash
uv run --with torchvision==0.21.0 python -m pytest -q tests/test_legacy_av.py
```

That temporary environment resolved PyTorch 2.6.0+cu124 and torchvision
0.21.0+cu124. It did not change the project's lock or runtime requirements.
The parity report identifies exact source/model hashes, dtype, input shapes,
seed, software version and output hashes. This validates model arithmetic and
stored-input inference. It does **not** certify the raw MATLAB CLAHE/resampling
pipeline: original preprocessed DICOM/MATLAB goldens were not supplied. Prepared
DICOM mode retains the exact old input contract; raw-image surrogate preprocessing
must be labeled accordingly.

## Explicit calibrated raw CLI profile

`uv run deepretina predict --model legacy-av --av-raw-matlab` selects the
separate calibrated Python profile. Supply `--input`, `--fov`, `--dataset`
(or `--field-of-view-degrees`), and `--output`. The generic provided-mask center
and radius are estimated unless a positive `--fov-diameter-pixels` overrides
them. Empty/full masks and special source branches fail closed without an explicit
calibration; see [provided-mask scope](MATLAB_FOV_CALIBRATION.md).
An optional original-frame `--vessel-mask` bypasses automatic original vessel
prediction. The raw mode cannot be combined with prepared arrays, DICOMs,
`--prepared`, or `--av-raw-proxy`. See the complete [README command](../README.md)
and [numerical qualifications](MATLAB_AV_PROFILE.md).

Output metadata records `raw_av_preprocessing`, its actual profile name,
`raw_av_geometry`, and the vessel source. The calibrated profile keeps
`matlab_parity_certified=false` and
`historical_training_file_provenance_verified=false`. Classes/overlays are
restored to the original photograph dimensions. The earlier proxy retains its
`raw_av_proxy` metadata for compatibility. Native prepared inputs and original
single-net normalization remain separate routes.

## Output support in raw A/V prediction

`vessel.png` retains the original-frame vessel mask supplied to A/V preparation.
The classifier gates its output on the nearest-resized mask in its own prepared
grid; restoring that support to the original frame can move boundary pixels.
The exact restored gating mask is `vessel_support` in `probabilities.npz`.
Artery/vein outputs are mutually exclusive and gated by that restored support.
Nested vessel prediction metadata links only to the retained `vessel.png`;
temporary probability, overlay and metadata files are discarded.
