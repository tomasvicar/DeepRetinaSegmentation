# Optional automatic MATLAB-source RGB preparation

`prepare_vesclass_matlab_source_auto` composes the measured automatic FOV
first-cutoff backend with the existing calibrated resize/photometry preparer.
It is an explicit Python API; public CLI/defaults and frozen evaluation/training
sources are unchanged. No MATLAB installation is required at runtime.

```python
from deepretina.legacy.matlab_source_auto import prepare_vesclass_matlab_source_auto

prepared = prepare_vesclass_matlab_source_auto(
    raw_rgb_uint8, binary_original_frame_vessel_mask, dataset="stare"
)
inputs = prepared.inputs()  # existing four-plane stored-value contract
boundary = prepared.source_fov.calibration_boundary  # original frame
processing = prepared.source_fov.processing_fov       # original frame
resized_processing = prepared.fov                     # prepared frame
```

The result inherits the existing PreparedVesClassProxy interface. Its extra
`source_fov` retains the distinct calibration boundary, processing mask and
calibration. The composed preparer passes **that processing mask and the explicit
measured2×radius** to the existing preparer. It never estimates another radius
from a filled processing mask. Metadata identifies automatic calibration, its
actual center/radius, source diagnostics and both mask hashes; it does not label
the automatic estimate as a caller-supplied measurement.

The profile is fixed at25pixels/degree with the existing25photometry parameters.
Field angles are literal unambiguous user-owned7bb loader parameters, including
STARE35°, Drishti30°, DiaRetDB50°, E-Ophtha40° and RIDB45°. REVIEW(60/50) and
RIGA(45/35) require an explicit angle. Source loaders that require special provided
masks (RIM/Erlangen) are rejected by this automatic composition; their separate
provided-mask policy is not silently replaced. Unknown datasets need an explicit
angle. Supplied histogram edges/thresholds are deliberately absent from this API.

This is the explicit souradnice`fov=0` branch. It does not claim to replay official
provided-FOV policies of other dataset loaders. All geometry remains full frame,
with source scalar bicubic resize/ceil output shape and nearest categorical
resampling. Original single_net Python normalization is untouched.

## Measured complete-chain gate

The exact7bb image_adjustment source is followed by its separately invoked
local_contrast_and_clahe helper and independently rounded `uint16(round(RGB*4096))`
storage contract. This is not a replay/certification of every historical exporter
or proof of which training files were used. Vessel inputs are synthetic patterns
for geometry tests; no annotations are read.

| Input scope | Prepared shape | Resize max error | Normalized max error | Stored12 unequal / max units |
| --- | --- | --- | --- | --- |
| STARE train im0002 | 815×943 | 9.99e−16 | 6.36e−5 | 390 /1 |
| Drishti official train002 | 641×749 | 5.55e−16 | 8.57e−5 | 354 /1 |
| DiaRetDB0 image-only001 | 1005×1308 | 7.77e−16 | .06624 | 74,165 /271 |
| Synthetic STARE, explicit12° fixture | 350×437 | 2.35e−15 | 7.63e−5 | 27 /1 |
| Synthetic E-Ophtha, explicit12° fixture | 250×312 | 2.22e−15 | 4.45e−7 | 1 /1 |

Every measured calibration, raw/nearest-resized processing FOV and vessel mask
matches exactly. Four additional E-Ophtha/RIDB synthetic cases verify the source
half-factor cutoff and end-to-end masks/boundaries/calibration, including fallback
behavior. The numerical RGB residuals above remain explicit; no blanket parity
is claimed and `matlab_parity_certified` stays false.

## DiaRet first divergence and bounded operator study

On the **same MATLAB-resized RGB**, distance is exact. The first difference is
Sobel gx(max1.5259e−4)/gy(max2.2888e−5), followed by query-coordinate error(max.00708),
extrapolation9.41e−5, Gaussian background1.44e−6 and V-before-CLAHE.001655.
Feeding the **same MATLAB V** to Python CLAHE yields max4.44e−16, so a histogram
snap/epsilon would hide the upstream issue rather than fix it.

For this actual shape/input, MATLAB Gaussian auto equals explicit spatial
bitwise; explicit frequency differs only2.89e−15. The bounded Sobel study uses
Drive, HRF and DiaRet reference single-distance arrays plus six synthetic masks.
Double accumulation then one float32 cast, both separable float32 orders,
OpenCV Sobel and SciPy float32 correlate each disagree with MATLAB on at least
one nonconstant case. The existing operator differs at fewer entries in these measured cases, but is
also imperfect.
**No speculative operator replacement,720-order search or per-image rule was
introduced.** Exact single-output accumulation/traversal remains unresolved.

## Actual A/V numerical sensitivity, not quality

One paired DiaRet measurement uses the same actual dedicated-vessel prediction
from that raw image for both unchanged converted V9 forwards. There is no GT,
threshold tuning, tiling/downsampling or model/weight change. The common prepared
support has59,805 vessel pixels; its normalized fourth-plane difference is zero.

Stored RGB differences yield normalized-input max.20781, logit max.20369
(mean.00031839) and conditional-probability max.06658 (mean.00005007).
**Gated classes disagree at zero pixels**, both prepared and nearest-restored
native geometry. Prepared artery/vein counts are45,044/14,761; native counts are
59,325/19,338 on both sides. This one controlled observation is not accuracy,
clinical equivalence, unseen-data performance or a universal parity guarantee.
Continuous outputs are observably different.

The unchanged full-forward pair completed in7.92seconds at CPU2/nice10, CUDA
hidden, kernel peakRSS2,776,178,688bytes, under the explicitly permitted6GiB/300s
bound. An initial2GiB attempt was stopped before returning the first AV result;
the raised limit changed only the practical memory allowance. No feature-lifetime
rewrite was used. An earlier launcher metadata-key error occurred before vessel
forward and was corrected to the public crop_size/border arguments.

Reports, exact weight/support hashes, stage tables, resource receipt and durable
reproduction scripts are in [matlab-source-auto](../reports/legacy/matlab-source-auto/README.md).
The module remains optional pending review and does not enable a held-out scorer.

Verification:1890 focused CPU tests passed (5.25seconds), including the four
half-factor cases, two stored-RGB synthetic chains, default source angles and
no-second-calibration spy. Ruff/format are clean. Component/report/test-log hashes
are in [verification.json](../reports/legacy/matlab-source-auto/verification.json).
