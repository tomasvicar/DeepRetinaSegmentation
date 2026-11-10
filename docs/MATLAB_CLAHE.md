# Isolated MATLAB AV CLAHE profile

This experimental module is separate from the historical Python single_net
normalization. No production route is automatically changed. The package remains
Python/uv; MATLAB was used only as a licensed CPU numerical oracle.

`deepretina.legacy.matlab_clahe.adapthisteq_uniform` implements the normalized
floating2D, uniform-distribution, full-range subset needed by the original AV
pipeline. It computes float64 results and rejects integer/color/nonfinite/out-of-
range input. `av_matlab_clahe_v` applies the user-owned source parameters:
256bins, ClipLimit0.005 and tile counts rounded from height/width divided by
`250/(55/25)`. This photometric kernel is fixed25px in that source, including its
later35px geometry experiment. The source's training path names25N, but that
alone does not prove provenance of every historical training file.

```python
from deepretina.legacy.matlab_clahe import av_matlab_clahe_v

v_after = av_matlab_clahe_v(v_before)  # normalized double-valued V plane
```

The source-profile helper intentionally rejects tiny images that would produce
fewer than2tiles; the general function supports tiny inputs with explicit valid
tile counts. This is not an implementation of every MATLAB image class,
nonuniform distribution or Range option.

## Algorithm and provenance

The implementation uses public [adapthisteq documentation](https://www.mathworks.com/help/images/ref/adapthisteq.html),
[Zuiderveld's published CLAHE description](https://www.cse.unr.edu/~bebis/CS474/StudentPaperPresentations/1994%20-%20CLAHE.pdf),
and deterministic black-box calls to MATLAB R2026b Update1. **No proprietary
MathWorks toolbox source was read or copied.** The generated scripts use only
public functions and user-owned original processing parameters.

The numerical conventions independently established by these calls are:

- nearest normalized histogram-bin assignment;
- centered symmetric padding to even contextual-tile sizes;
- minimum clip count `ceil(tile_pixels/bins)` plus the rounded normalized excess allowance;
- mass-preserving capped redistribution, cumulative mapping and bilinear interpolation between tile centers;
- full nominal output range without global stretching to the observed extrema.

A final clamp removes floating-point convex-interpolation overshoots of a few
ulps. This avoids returning values above1 to later storage validation.

## References and measured tolerances

[Comparison JSON](../reports/legacy/matlab-clahe/comparison.json) records all stage
errors and source-array hashes. [Reproduction script](../reports/legacy/matlab-clahe/compare_goldens.py)
reads external golden arrays; [MATLAB generator](../reports/legacy/matlab-clahe/generate_goldens.m)
generates deterministic synthetic cases and the additional HRF stage arrays.
Real photographs and full-resolution arrays remain in local scratch outside Git.
The committed52KBfixture contains **only synthetic arrays**, with its SHA in
`tests/reference/matlab_clahe/metadata.json`.

All80synthetic MATLAB goldens pass at absolute tolerance1e-14: constants, ramps,
random inputs, odd rectangles,2×3tiny geometry,8/256bins, clip0/0.005/0.1/1,
and multiple tile grids. Identical real MATLAB V inputs produce:

| Dataset | V shape | Tile grid | Mean absolute error | Maximum |
|---|---|---|---:|---:|
| DRIVE25 |1222×1182|11×10|6.45e-17|4.44e-16|
| HRF25 |1073×1610|9×14|4.79e-17|4.44e-16|

Every pixel is within1e-12; this is numerical agreement with these R2026b
references, not a bitwise or unknown-historical-runtime guarantee.

## Whole-chain residuals: no raw-pipeline certification

Replacing only CLAHE in the existing Python proxy leaves material errors.
All following errors are in uint16 storage counts after rounding RGB×4096,
measured within the provided FOV:

| Input/remaining operations | DRIVE MAE / max | HRF MAE / max |
|---|---:|---:|
| Identical MATLAB illumination-correctedRGB, newCLAHE+PythonHSV |0.000005/1|0/0|
| MATLAB-resizedRGB, otherwise existingPythonphotometry+newCLAHE |16.54/189|17.25/1841|
| Existing rawPythonproxy with newCLAHE |58.60/1120|67.87/3316|
| Diagnostic photometric candidate from MATLAB-resizedRGB |0.000544/36|0.001264/37|

The diagnostic candidate is report-only: single-precision distance/gradient
coordinates, linear interpolation and the documented MATLAB Gaussian support
`2*ceil(2*sigma)+1` with replicated borders. On identical extrapolated input,
that Gaussian matches to MAE~1e-16/max1.56e-15. Its remaining corrected-V bin
mismatches are1pixel for DRIVE and3for HRF, with input differences≤4.2e-7.
CLAHE's discrete bin decisions can amplify these tiny upstream differences;
nearest-bin discontinuities are not evidence that the CLAHE stage is wrong.

The [public Gaussian defaults](https://www.mathworks.com/help/images/ref/imgaussfilt.html)
explain the former support mismatch. Extrapolation/interpolation and scalar-scale
bicubic raw resizing still need their own measured treatment. Boundary rounding
in HSV/storage also prevents casually calling floating agreement bitwise parity.
No production activation or claim of full raw-AV MATLAB parity follows yet.
Existing single_net Python preprocessing remains unchanged and its golden tests
continue to pass. Prepared original DICOM remains the exact AV input reference.
