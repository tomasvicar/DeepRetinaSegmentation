# Optional MATLAB source FOV stages

This is a separate experimental API, not a replacement for public calibration or
single_net Python preprocessing. No CLI/default, canonical data, model weights,
training source checkout or running experiment is changed. The provided-special
stage is implemented and measured below; automatic image-derived FOV is still
incomplete and is not certified or activated.

`legacy.matlab_fov_source.provided_fov_source(mask,dataset=...)` returns separate
`calibration_boundary`, `processing_fov`, `calibration`, `closed_mask` and
`source_branch` fields. Calibration is computed from exactly the source boundary,
not necessarily the returned processing mask. Coordinates are MATLAB one-based.
The original name prumer means radius; diameter is twice radius. Degenerate
source calibrations raise instead of inventing a robust fit.

## Provided branches and measured gate

| Original source branch | Calibration boundary | Returned processing FOV |
|---|---|---|
| generic | default Sobel edge of supplied mask | supplied mask |
| RIM-ONE | supplied mask itself | whole-frame ones |
| Erlangen / video | left/right outer vertical rails | supplied mask, normally ones |
| ROC / RIGA | Sobel of MATLAB rows/cols 20:end-20 embedded into zero canvas | supplied mask |

Original load_rimone supplies a **left-half rectangular wireframe**, not a dense
left-half retina mask. `rimone_loader_mask(shape)` reproduces it and rejects odd
width as the source indexing requires an integer half-width. Passing a dense mask
would be a different source input and can yield a very different radius. The
optional API intentionally retains this behavior. Erlangen even retains an empty
provided processing mask while calibrating from its artificial rails; that is a
source fact, not endorsement of that input for photometric prediction.

The public generic helper still rejects all four special families without explicit
calibration; importing this module does not relax that guard. To experiment with
later photometry, a caller must deliberately pass both result.processing_fov and
result.calibration.diameter_pixels, not recalibrate the processing mask.

[provided-comparison.json](../reports/legacy/matlab-fov-source/provided-comparison.json)
records sixteen public-MATLAB/source oracle cases (fourteen synthetic plus two
actual source image shapes). Every defined boundary, processing mask, center and
radius matches exactly. Clipped/empty source failures remain failures. Synthetic
fixtures contain only small generated masks; real image pixels are not in Git.
Source inputs were one Erlangen PNG (770×1000) and one RIM-ONE r3 stereo JPEG
(1424×2144), read from ownCloud ZIPs without modifying them. The RIM loader's
wireframe and Erlangen loader's full mask were constructed exactly as source.
The provided branch uses image dimensions, not RGB values; actual source bytes
and paths are hashed in the report for provenance. This does not certify the full
photometric pipeline or interpret source calibration as physical camera truth.

The oracle uses unchanged user-owned souradnice.m at 7bb0539b and public toolbox
calls only. It ran on Ada CPU 1, singleCompThread, nice 10, CUDA hidden, with a 90 s
wall limit per invocation. No proprietary toolbox source was read or copied.
Reference generators and fixture packager accompany the report.

Real-source calibrations: Erlangen center386,501/radius501; RIM-ONE center713,537/radius537, all MATLAB one-based. These are source outputs, not new camera-angle validation.

## Remaining automatic branch: source map and exactness risks

The source first converts incoming RGB to grayscale and chooses low/high cutoffs,
selects 8-connected low-mask components touched by a high-mask pixel, closes with
strel disk(10), optionally retries using a 250-bin histogram quantile, and fills
holes. **Calibration uses final_closed before hole fill**, whereas the returned
processing FOV is the filled mask (except RIM full-frame override). Returning only
a filled mask and then recalibrating it would lose source semantics.

| Source identifiers | Initial low threshold |
|---|---|
| diaretdb, review | min(multithresh(gray,4))×0.1 |
| drishtigs, idrid, messidor, riga, roc, vicavr, avrdb | min(multithresh(gray,5)) |
| stare | min(multithresh(gray,3)) |
| eophtha, ridb | min(multithresh(gray,5))×0.5 |
| others | 0 |

High threshold is 0.4×max(gray) for inspireavr, otherwise 0.8×max(gray). Fallback
expected-black fraction overrides: ARIA/G1020=.1343, DiaRetDB=.1634, STARE=.269,
Drishti=.1568 (or.2929 for width>2200), E-Ophtha=.45. Allowed white-fraction
error is.1, except Drishti .05 and E-Ophtha .15. Other sources compute their
expected fraction from the same initial closed mask, so that fallback condition
is normally self-satisfied. The retry uses the first histogram cumulative count
strictly greater than expected-black×pixel-count and the corresponding LEFT edge;
no adaptive loop or robust improvement should be silently added.

Primary docs establish [bwlabel8-connectivity](https://www.mathworks.com/help/images/ref/bwlabel.html),
[imfill4-connectivity](https://www.mathworks.com/help/images/ref/imfill.html), and
[strel's default four-line disk approximation](https://www.mathworks.com/help/images/ref/strel.html).
An exact Euclidean disk or default SciPy connectivity is not automatically faithful.
[Multithresh](https://www.mathworks.com/help/images/ref/multithresh.html) uses a
locally optimal search for N>2; the source uses N=3/4/5. A global multi-Otsu optimum
may improve the objective while disagreeing with MATLAB. It must not be declared
an exact drop-in. Threshold quantization/ties, morphology padding and histogram
bin edges require public-API oracle stages before an automatic port is activated.

Next bounded oracle batches: grayscale/footprint/connectivity on tiny synthetic
inputs; zero-low-threshold source families; fallback histogram cases; then N3/4/5
threshold outputs and a few representative real inputs. Keep thresholds, selected
components, closed mask, filled mask, boundary and radius as separate stages.
If local-search equivalence cannot be recovered cleanly, report the measured gap
and keep those profiles explicit/uncertified rather than masking it with epsilon
or substituting a better optimizer. No automatic-profile parity is claimed here.

UOA is another call-site distinction: its MATLAB loader strips the timestamp
AFTER image_adjustment, whereas the separately verified single_net Python loader
has its own earlier masking policy. Do not mix these two family contracts.

Validation:26 focused CPU cases passed on Ada, nice10, one thread, CUDA hidden.
Ruff/format checks ran remotely; no local heavy test or MATLAB process was run.
Rebuild instructions are in [the oracle README](../reports/legacy/matlab-fov-source/README.md).

## Automatic source stages: measured subset, not a complete automatic port

The separate `automatic_fov_stages` / `automatic_fov_source` study API now
implements the original double RGB-to-gray coefficients, 8-connected seeded
components, default `strel('disk',10)` approximation, closing and four-connected
hole filling. Closing pads nine background pixels before dilation/erosion and
crops afterwards. Plain SciPy closing on the original frame disagrees at image
edges. The measured disk approximation is 19×19, rather than a Euclidean radius10
disk. Seed membership deliberately retains label zero if a high seed selects it.
These are source semantics, including awkward cases.

Three grayscale, five morphology and two seeded-component synthetic goldens are
bitwise equal. Seven automatic zero-low-threshold scenes match every measured
stage, processing mask and defined center/radius when the MATLAB histogram edges
are supplied explicitly. Two scenes reproduce source calibration failure. A hole
counterexample calibrates radius20 from the boundary before hole filling; using
the filled processing mask would change the answer. This is why both arrays are
retained. `MatlabSourceFOV.diagnostics` identifies supplied oracle edges and
thresholds explicitly.

Two algorithms remain unresolved and fail closed:

- Source multithresh N3/4/5 profiles require a measured threshold vector for
  studies. MATLAB documents a search-based locally optimal algorithm for N>2;
  substituting global multi-Otsu would be a different method.
- The source fallback `histcounts(gray,250)` automatically chooses floating bin
  edges. A guessed edge rule failed the nonzero-background G1020 synthetic
  counterexample: source cutoff `.002`, guessed cutoff `1/255`, with 11,339 mask
  disagreements. That implementation was removed. Fallback requires measured
  251-element edges, checks their coverage, and records `supplied_oracle_edges`.
  Nonfallback N0 source profiles run without oracle parameters.

No public raw-AV route imports this module. These measurements establish stage
agreement on the listed cases, not full raw-image preprocessing, historical
export provenance, or complete automatic-FOV parity. In particular RIM-ONE's
calibration boundary, returned full-frame processing FOV, any later crop, and
exporter handling of an all-ones mask must be verified separately against the
chosen original caller revision. Erlangen video branches need caller evidence.

The numerical report is
[`automatic-comparison.json`](../reports/legacy/matlab-fov-source/automatic-comparison.json).
Public operator documentation:
[rgb2gray](https://www.mathworks.com/help/matlab/ref/rgb2gray.html),
[bwlabel](https://www.mathworks.com/help/images/ref/bwlabel.html),
[strel](https://www.mathworks.com/help/images/ref/strel.html),
[imfill](https://www.mathworks.com/help/images/ref/imfill.html),
[multithresh](https://www.mathworks.com/help/images/ref/multithresh.html), and
[histcounts](https://www.mathworks.com/help/matlab/ref/double.histcounts.html).
The footprint, coefficients and error counts above are direct R2026b oracle
measurements; no proprietary toolbox implementation was inspected.
