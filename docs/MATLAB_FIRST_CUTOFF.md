# Optional MATLAB FOV first-cutoff contract

The FOV source does **not** consume a full multithresh vector. In the user-owned
revision`7bb0539b90dfce2ed6eaeef1de812072be1155d0`, all four occurrences in
[`matlab_preprocessing/souradnice.m`](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/7bb0539b90dfce2ed6eaeef1de812072be1155d0/matlab_preprocessing/souradnice.m#L10)
are immediately reduced with`min`:

| Source lines | Required low threshold |
| --- | --- |
| 10–11 | DiaRetDB/REVIEW: min(multithresh(gray,4))×0.1 |
| 12–13 | Drishti/IDRiD/MESSIDOR/RIGA/ROC/VICAVR/AVRDB: min(multithresh(gray,5)) |
| 14–15 | STARE: min(multithresh(gray,3)) |
| 16–17 | E-Ophtha/RIDB: min(multithresh(gray,5))×0.5 |

The upstream source loaders actually pass automatic FOV (`fov=0`) in this ref:
`load_stare.m:30`, `load_diaretdb0.m:10`, and
`load_drishtigs.m:12` (training; test caller is line32, not used by this study).
Their camera angles are35°,50° and30° respectively; those are field angles,
not pixels-per-degree settings.

Line29 uses only`gray>prah1`; lines72–75 can replace it with the histogram fallback
cutoff. No later source operation uses any other threshold. The enclosing
[`image_adjustment.m`](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/7bb0539b90dfce2ed6eaeef1de812072be1155d0/matlab_preprocessing/image_adjustment.m#L1)
converts RGB to double at line2 before calling souradnice at line3. That is the
measured input contract, distinct from calling rgb2gray directly on uint8.

## Explicit API; defaults stay unchanged

```python
from deepretina.legacy.matlab_firstcut import first_cutoff_source
from deepretina.legacy.matlab_fov_source import (
    automatic_fov_source,
    source_rgb2gray,
)

first = first_cutoff_source(source_rgb2gray(raw_rgb_uint8), 3)
print(first.value, first.effectiveness_status)

result = automatic_fov_source(
    raw_rgb_uint8,
    dataset="stare",
    threshold_backend="source_first_cutoff",  # explicit opt-in
)
processing_fov = result.processing_fov
calibration_boundary = result.calibration_boundary
calibration = result.calibration
```

`first_cutoff_source` requires finite float64 grayscale in[0,1] and integer N3/4/5.
It returns only the unscaled first cutoff, its possibly fractional bin coordinate
and measured method/status. It does not expose a purported complete threshold
clone. The FOV backend applies the source dataset factor, component/morphology
policy and any source histogram fallback. Its diagnostics record the actual
backend and cutoff metadata. Mixing supplied oracle thresholds with this backend
is rejected. Degenerate arbitrary-threshold policies fail closed.

The existing default`threshold_backend="oracle"` remains guarded for N3/4/5
without explicitly supplied measured thresholds. The first-cutoff study changed no public default, frozen evaluation3779,
training3157 or canonical corpus. A later explicit opt-in CLI is documented in
[MATLAB_SOURCE_AUTO.md](MATLAB_SOURCE_AUTO.md); it does not promote this backend to a default.
The method's`full_multithresh_clone` and`matlab_parity_certified` stay false:
finite measurements do not prove equivalence for every possible input/runtime.

## Independent oracle evidence

Two independent synthetic batches cover uniforms, multimodal distributions,
sparse/discrete levels, dense histograms, power skew and narrow intensity ranges:
600 cases with seed11827 and720 with seed90173, each across N3/4/5. After the
failed-search correction below, **all1320 first cutoff values are bitwise equal**
to actual MATLAB R2026b outputs. The source factors produce zero pixel-threshold
decision disagreements. These counts compare actual floats, not rounded indices.
Full vectors are599/600 and718/720 equal; those higher-cut differences are unused
by the FOV caller and are deliberately not patched.

The first batch exposed a real error in the earlier analysis: comparing rounded
bin coordinates hid22 fractional first-cutoff differences and14 pixel decisions.
When optimization fails with empty classes, the source returns the **continuous
uniform initial positions**, including63.75 for N3 and42.5 for N5, with negative-
infinity effectiveness. Rounding that fallback was wrong. The optional method
now preserves these values and records`negative_infinity` status without
inventing a finite score. Regression tests retain the fractional counterexample.
The old rounded-coordinate report remains historical evidence, not the current
floating-cutoff contract.

Eighteen synthetic spatial RGB cases match initial/closed/filled masks,
boundaries, processing masks and every defined calibration/error. Three real
source frames cover the actual N3/N4/N5 caller branches:

| Raw frame scope | Shape | Source center(row,col), radius | Disagreements |
| --- | --- | --- | --- |
| STARE im0002, canonical train membership | 605×700 | (290,345),325 | 0 |
| DrishtiGS002, official train | 1752×2045 | (879,1025),1025 | 0 |
| DiaRetDB0 image001, image-only calibration input | 1152×1500 | (556,743),717 | 0 |

First/final cutoff floats, all measured mask stages and calibration agree exactly
on those frames. No ground-truth annotations, network forwards, test-quality
metrics or GPU execution were used. Real images/intermediate arrays remain in
external scratch; Git retains hashes, numerical reports and synthetic-only RGB
fixtures/histogram counts. The histogram fixture reconstructs sufficient counts
and min/max, rather than storing the1320 original synthetic sample arrays.

[Detailed report](../reports/legacy/matlab-fov-source/firstcut-contract-report.json)
records caller lines, seeds, actual oracle-file hashes, the earlier measurement
correction and per-frame stage differences. Reproduction scripts are in the
[oracle guide](../reports/legacy/matlab-fov-source/README.md).

This completes the measured first-cutoff requirement for the optional FOV module,
not every part of raw prediction. Full-frame image_adjustment resize, processing
FOV versus calibration boundary, photometry and historical exporter provenance
remain separately checked contracts. A future composed profile must pass its
measured diameter explicitly when preparing RGB, rather than calibrating again
from its hole-filled processing FOV. Public activation requires separate review.

Verification:1869 focused CPU tests passed on Ada, nice10/CUDA hidden/thread1
(5.05seconds), with Ruff/format clean. Component/runtime/log hashes are recorded
in [firstcut-verification.json](../reports/legacy/matlab-fov-source/firstcut-verification.json);
that receipt is not a full-checkout attestation or a model-quality result.
