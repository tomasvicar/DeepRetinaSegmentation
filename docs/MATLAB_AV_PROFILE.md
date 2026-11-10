# Calibrated MATLAB AV preprocessing (experimental)

`prepare_vesclass_matlab_calibrated` is a separate Python profile, available through
the explicit `predict --av-raw-matlab` CLI option. It requires the original-frame
FOV. A supplied diameter overrides the verified generic provided-mask calibration;
without one, Python estimates the source center/radius from that mask.
Existing U-Net Python normalization,
prepared AV routes and default profiles remain available. Runtime uses Python/uv;
MATLAB is used to generate numerical test references.

```python
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated

prepared = prepare_vesclass_matlab_calibrated(
    rgb_uint8,
    binary_vessels,
    fov=original_fov,
    # Optional explicit fov_diameter_pixels=538 overrides generic calibration.
    dataset="drive",
    pixels_per_degree=25,
)
# prepared.image: HWC uint16 stored12; prepared.vessel: matching binary mask
# prepared.inputs(): four stored-value input planes for the original AV model
```

The optional diameter 538 belongs only to the measured DRIVE reference photograph; it is not a dataset-wide calibration. Masks must match the original RGB grid. Floating vessel probabilities must be thresholded explicitly. Unknown dataset angles need `field_of_view_degrees`. Image-based FOV extraction and special dataset branches, including stereoscopic cases, remain outside the verified scope. Automatic radius estimation uses only the supplied mask and fails closed for special datasets without an explicit diameter.

The implementation follows user-owned `image_adjustment.m` and `local_contrast_and_clahe.m`: full-frame scalar bicubic resize with antialiasing and ceil output dimensions, nearest masks to the output shape, eroded provided FOV, float32 distance/Sobel extrapolation, replicate Gaussian background subtraction, HSV CLAHE, and uint16 rounding of normalized RGB ×4096. Photometric constants stay at the source's 25-pixel/degree setting even when an explicit outer geometry uses 35. No MATLAB toolbox source was inspected or copied. Public references are [imresize](https://www.mathworks.com/help/matlab/ref/imresize.html), [imgaussfilt](https://www.mathworks.com/help/images/ref/imgaussfilt.html), and [adapthisteq](https://www.mathworks.com/help/images/ref/adapthisteq.html). CLAHE details and separate tests are in [MATLAB_CLAHE.md](MATLAB_CLAHE.md).

## Reference results and limits

[comparison.json](../reports/legacy/matlab-av-profile/comparison.json) records MATLAB R2026b Update1 goldens and their hashes. Real photographs/arrays stay outside Git; small synthetic resize fixtures are committed. Reproduce with `uv run python reports/legacy/matlab-av-profile/compare_profile.py --golden-dir PATH --output REPORT.json`. MATLAB generator scripts accompany the report; they call public APIs only.

| Reference | Prepared H×W | FOV disagreements | Stored12 MAE inside FOV | Maximum difference |
|---|---:|---:|---:|---:|
| DRIVE 25 | 1222×1182 | 0 | 0.00010970 | 26 |
| HRF 25 | 1073×1610 | 0 | 0.00000388 | 1 |
| DRIVE 35 | 1710×1655 | 0 | 0.00000310 | 1 |

All 24 synthetic bicubic cases agree within 6e-15; nearest masks are bitwise identical. Real bicubic RGB maximum errors are below 1.7e-15. Distance fields are bitwise identical. Float32 Sobel accumulation still differs at some background pixels, producing small extrapolation/background errors. DRIVE has one changed V histogram bin; injecting reference V reduces its stored12 maximum error to 1. HRF has no changed bins. These are measured residuals, not a universal tolerance guarantee.

A deliberately retained smooth full-frame synthetic image is a counterexample: stored12 MAE 2.2523, maximum 37, despite only floating-point-scale Gaussian differences near a histogram boundary. MATLAB's automatic Gaussian domain equals its explicit spatial domain in this probe; its frequency result differs by up to 1.55e-15. Thus the observed issue is not explained by an auto-domain switch. CLAHE histogram quantization can amplify roundoff in nearly flat regions. No ad-hoc epsilon or bin snapping is applied. The circular-FOV counterpart has MAE 0.00716/max37; a constant-image case is stored12 exact.

The current profile is therefore **not certified bitwise MATLAB parity**, and historical training-file provenance is not verified. The training source points to 25N; later 35 exports do not establish what a particular saved checkpoint saw. The actual-model sensitivity gate below was completed before adding the explicit CLI route. Fixed synthetic correctness tolerances in tests do not weaken these limitations.

## Actual saved-model sensitivity gate

[model-forward.json](../reports/legacy/matlab-av-profile/model-forward.json) compares the actual safe-converted legacy V9 checkpoint (`82428903…f270b`) on MATLAB stored RGB versus Python stored RGB. Each pair receives the **same nonconstant real manual vessel mask**, nearest-resized to the common prepared grid: DRIVE 21_manual1 and HRF 01_g. This isolates preprocessing differences; manual vessels are an input, not a segmentation-quality benchmark. The checkpoint hash is unchanged after inference. The immutable source87 AV port ran on Ada CPU only, nice5, four threads, CUDA hidden; two pairs took approximately 18 seconds total.

| Reference | Standardized input max error | All-logit MAE / max | Conditional probability max error | Conditional argmax differences (all / FOV / vessels) | Gated class differences |
|---|---:|---:|---:|---:|---:|
| DRIVE25 | 0.0257933 | 9.05e-6 / 0.0120920 | 0.00602061 | 4 / 3 / 0 | 0 / 108109 vessel pixels |
| HRF25 | 0.00132072 | 3.36e-7 / 0.000664234 | 0.000257850 | 0 / 0 / 0 | 0 / 115420 vessel pixels |

Per-logit-channel errors, input mean errors, probabilities, vessel hashes and source references are retained in the JSON. `compare_model_forward.py` reproduces the gate with external goldens and original archives, without changing either. The two measured vessel-gated classifications are identical; logits are not identical. These results do not certify arbitrary images, image-based FOV extraction, historical checkpoint preprocessing provenance, or standalone raw prediction quality. See the [CLI example](../README.md) and [AV input contracts](LEGACY_AV.md) for usage. The CLI also has an actual bundled-model test covering calibrated preparation, native output dimensions and profile metadata.

The generic provided-mask center/radius stage is available as a separate
[calibration helper](MATLAB_FOV_CALIBRATION.md). The complete raw golden and
intermediate-stage generation procedure is preserved in
[REPRODUCING.md](../reports/legacy/matlab-av-profile/REPRODUCING.md).

Output `raw_av_preprocessing.fov_calibration` distinguishes
`estimated_from_provided_fov` from `explicit_diameter`. The former stores
`method=souradnice_generic_binary_sobel`, one-based center coordinates, radius
and edge count. An explicit diameter stores `method=caller_supplied_diameter`,
its half-radius, and null center coordinates; no automatic estimate is implied.
The same metadata is returned by the library profile.
