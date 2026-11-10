# Provided-FOV calibration against MATLAB

The experimental `legacy.matlab_fov` module reproduces the **generic provided-mask** branch of the user's original `souradnice.m` at `7bb0539b90dfce2ed6eaeef1de812072be1155d0`. It performs no image thresholding, morphology-based FOV discovery, bounding-box diameter substitution, or production default change. It can supply a diameter to the separately calibrated AV preprocessing API:

```python
from deepretina.legacy.matlab_fov import calibrate_provided_fov
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated

calibration = calibrate_provided_fov(original_binary_fov, dataset="drive")
prepared = prepare_vesclass_matlab_calibrated(
    raw_rgb_uint8, original_binary_vessels, fov=original_binary_fov,
    dataset="drive", fov_diameter_pixels=calibration.diameter_pixels,
)
```

This is an explicit experimental choice. A measured/supplied calibration retains priority: callers that already have one should pass it directly and need not run this estimator. CLI defaults and the existing explicit-diameter contract are unchanged on this branch. The returned center uses MATLAB one-based row/column coordinates. The original variable `prumer` means **radius** despite its name; the Python `diameter_pixels` property returns twice that value.

The source first applies default Sobel `edge` to the provided FOV. Center coordinates are positive-half-up rounded modes of midpoint coordinates from rows/columns containing at least two edge pixels; tied modes use their smallest value. Radius is one plus the rounded modal edge distance from that center. This is faithful to the source, not a robust circle/ellipse fit. For example, our clipped circle with generating radius35 yields source radius20; the elliptical case yields radius23. Such masks can be physically unsuitable even when numerical replication is exact.

The binary edge implementation uses replicate-border normalized Sobel gradients, twice RMS gradient magnitude as the automatic threshold, and discrete directional nonmaximum suppression with asymmetric plateau ties and zero exterior gradient magnitude. It was independently derived from the public [MATLAB edge API](https://www.mathworks.com/help/images/ref/edge.html) and black-box calls on synthetic masks. No proprietary toolbox source was inspected or copied. The implementation intentionally accepts binary masks only, not arbitrary grayscale edge inputs.

## Measured gate

[comparison.json](../reports/legacy/matlab-fov-calibration/comparison.json) records every stage and input-mask hash. MATLAB R2026b Update1 ran CPU-only, nice5, singleCompThread. Results:

| Input | Edge pixels | Center row,column | Radius / diameter | Edge and gradient disagreements |
|---|---:|---:|---:|---:|
| DRIVE21 official FOV | 1588 | 301,279 | 269 / 538 | 0 |
| HRF01_g official FOV | 4931 | 1167,1762 | 1633 / 3266 | 0 |

Both real default thresholds also match exactly. All **73 synthetic masks** have bitwise-equal edge, unthinned edge, and gradient arrays; all defined centers/radii match exactly. Cases cover odd/even circles, off-center and boundary-touching/cropped masks, an ellipse, holes, rectangles, checkerboard, empty/full, and 60 additional seeded sparse/dense/random ellipse probes. Eighty-six focused tests check the small committed synthetic fixtures, degenerate failures, and excluded source branches; no MATLAB or real images are needed at test runtime.

Empty/full and small threshold-suppressed masks cannot define source edge intersections; MATLAB errors and Python raises a documented `ValueError`. Python does not invent a calibration. RIM-ONE, Erlangen (including video), ROC and RIGA use special original branches and are explicitly rejected by this **generic** helper. They require separately verified handling or an explicit calibration. Automatic FOV extraction and arbitrary-mask universal parity are not certified. The remaining photometric roundoff and historical file-provenance limitations in [MATLAB_AV_PROFILE.md](MATLAB_AV_PROFILE.md) still apply.

## Reproduction after scratch cleanup

The [durable raw/stage builder instructions](../reports/legacy/matlab-av-profile/REPRODUCING.md) recreate real DRIVE/HRF/DRIVE35 goldens from original data and user-owned helpers. A fresh single-thread MATLAB rebuild was checked against all seven original real/stage files: every decoded array matched bitwise; [builder-reproduction.json](../reports/legacy/matlab-av-profile/builder-reproduction.json) retains the evidence. MAT-file headers can differ between runs.

To reproduce the additional mask gate:

1. `uv run python reports/legacy/matlab-fov-calibration/generate_inputs.py /ABS/SCRATCH/fov_inputs.mat` creates synthetic masks.
2. Add the report directory and original helper directory to the MATLAB path, change to that scratch directory containing the real `drive_golden.mat` and `hrf_golden.mat`, then run `generate_goldens`. It uses only public `edge` and the original user-owned `souradnice`. Keep CUDA hidden and `-singleCompThread`.
3. `uv run python reports/legacy/matlab-fov-calibration/package_goldens.py --golden-dir /ABS/SCRATCH --fixture-dir /ABS/NEW_FIXTURE --report /ABS/comparison.json` compares both stages/calibrations and packages only synthetic arrays. Do not overwrite the committed fixtures casually; review new MATLAB-version differences first.

No photographs, full real arrays, or original source data are copied into Git or modified.
