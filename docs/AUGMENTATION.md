# Optional augmentation trials

These helpers belong to the separate HPO branch. They do not alter the stable
`for_Leipzig` models or their default training recipe.

`CropDataset` continues its existing random crops, quarter turns and horizontal
flips for every augmentation mode. `TrainConfig.augmentation="basic"` preserves
that recipe. The `mild` and `strong` candidates additionally call
`deepretina.augment.augment(image, target, valid, level, generator=None)`.

Inputs are CHW floating tensors: RGB image values in `[0,1]`, per-channel targets,
and a matching boolean validity tensor. Output shapes and dtypes are retained,
all outputs are contiguous, and input tensors are never modified in place.
An explicit CPU `torch.Generator` or the global seeded CPU torch RNG controls all
parameter/noise draws. This fits DataLoader worker seeding without a separate
NumPy random stream. `CropDataset` owns a separate CPU augmentation generator in
each DataLoader worker, seeded from the worker seed plus a fixed offset. Optional
augmentation draws therefore do not advance the crop/quarter-turn/flip RNG stream,
so matched basic/mild/strong trials retain the same sampled crops. Mild and strong
use the same random-draw template. Tensor computation may run on CPU or CUDA; the explicit
generator must be a CPU generator. Bitwise reproducibility is checked on CPU;
backend/hardware changes can affect floating-point interpolation.

| Operation | Mild | Strong |
|---|---|---|
| Additional affine probability |0.75|0.75|
| Physical rotation |−10° to +10°|−20° to +20°|
| Isotropic zoom |0.90–1.10|0.80–1.20|
| Photometric probability |0.90|0.90|
| Additive brightness |−0.06 to +0.06|−0.15 to +0.15|
| Contrast multiplier |0.90–1.10|0.80–1.20|
| Gamma exponent |0.90–1.10|0.80–1.20|
| Independent channel multiplier |0.975–1.025|0.92–1.08|
| Gaussian noise standard deviation |uniform 0–0.01|uniform 0–0.025|

Affine parameters are applied jointly to the photograph, every target channel and
every validity channel using a single sampling grid. The grid compensates for
height/width differences so rotation remains a physical rotation on rectangular
images. Photographs use bilinear interpolation and reflected padding. Targets and
validity use nearest-neighbour interpolation with zero/false padding. An explicit
outside-domain check makes every new padding pixel invalid for supervision.
Categorical labels remain exact; a known vessel with unknown A/V class stays
unknown after transformation. Photometric changes affect only the image, and the
final image is clamped to `[0,1]`. Elastic deformation is not included.

The lower-level `warp_joint(..., angle_degrees=..., scale=...)` exposes deterministic
geometry for tests and controlled experiments; scale above 1 zooms in. `augment`
with `level="basic"` is an identity helper because the dataset already supplies
its original spatial augmentation.

## Verification and interpretation

Seven regression cases pass, covering exact quarter-turn geometry, padding
validity, unknown A/V support, categorical labels, input immutability, seeded
reproducibility and finite bounded image output. Invalid geometry, dtypes, levels
and nonfinite image values fail clearly.

A training photograph from each of DRIVE, RITE and Drishti was transformed with
basic/mild/strong at seed 3, producing nine checked examples. The contact sheet is
the local, Git-ignored `reports/hpo/augmentation/real_examples.png`; accompanying shapes, validity counts,
seeds and finite/category checks are in `reports/hpo/augmentation/sanity.json`.
Overlays show valid vessel/disc in red, vein/cup in blue, and ignored regions with
a dark checkerboard. AV vessel support is green. These artifacts check alignment
and appearance; they do not establish an accuracy gain.

The profiles are experiment candidates. Extra channel variation can hurt A/V
classification, and stronger geometry/noise can hurt thin vessels. Compare them
against `basic` using the same training images, seed, optimizer-update budget and
validation split; report elapsed time as well as validation metrics. Select the
profile using validation data and keep the test split sealed until the final
choice. Only measured gains should be promoted into a stable training recipe.
