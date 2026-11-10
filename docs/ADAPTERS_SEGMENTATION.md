# Legacy vessel and optic nerve adapters

The registry in `src/deepretina/adapters/segmentation.py` migrates the eight
remaining supervised-data readers from the historical MATLAB loaders and their
`single_net` Python counterparts. It uses the common read-only `Source` interface
and the shared preparation writer; it needs NumPy, Pillow and SciPy, installed by
`uv sync`. No MATLAB, OpenCV or scikit-image runtime is required.

| Selector | Local source | Decoded annotation support |
|---|---|---|
| `aria` | ARIA.zip | Both BDP/BSS vessel annotators for all three categories; disc outlines for control and diabetic images |
| `avrdb` | AVRDB.zip | Sparse binary vessel lines plus artery/vein line labels, including misspelled filenames |
| `g1020` | G1020.zip | LabelMe disc/cup polygons; missing channels remain unsupervised |
| `onhsd` | Onhsd.zip | Four clinician radial disc annotations; paired eyes and observers share patient group |
| `riga` | RIGA directory | Six overlaid disc/cup outlines per primary photograph, with explicit recovery quality |
| `rimone` | RIM-ONE.zip | Release 3 average and both expert disc/cup masks, annotated left stereo view only |
| `stare` | STARE.zip | Both AH/VK vessel masks where supplied; remaining photographs are image-only records |
| `uoadr` | UoA_DR.zip | Vessel/disc JPEG masks and optic-disc center markers, including the swapped vessel/disc files of image 72 |

Releases 1/2 files bundled in RIM-ONE.zip are not the input of the historical
`load_rimone.m`, which explicitly targets release 3. The adapter follows that
scope and retains all release 3 average/expert raster masks. Other releases are
not silently treated as independently annotated release 3 examples.

## Common annotation rules

All adapters preserve the photograph coordinate frame. The shared writer performs
aspect-preserving image resizing and nearest-neighbour label resizing. Their metadata
names the original `source_image` so observer variants share one prepared RGB file.
`source_id` groups observers, paired eyes where identifiable, and related records
before splitting. Filename-derived patient grouping is conservative; it does not
establish patient independence when the source lacks patient identifiers.

For disc-only annotations, `task=disc_cup` still has two channels, but cup validity
is entirely false. Missing or unusable labels never become supervised all-background
examples. Failed source annotations retain an explicit quality description and
all-false validity; the shared writer marks these records as unusable for training.
Their original annotation resources remain available for review.

Metadata includes `annotation_resources` with original source paths. Structured
JSON/MAT coordinates are in the **original image frame**, not the resized prepared
frame. The common writer preserves structured resources and unsuccessful source
rasters; successfully decoded raster annotations are represented by their prepared
masks and original path references. This avoids copying many gigabytes of redundant
source images while keeping failures reviewable.

## Dataset-specific decoding

**ARIA.** BDP/BSS masks use every nonzero source pixel, as in the original reader.
The disc/fovea outline conversion uses binary closing with a radius 21 Euclidean
disk, hole filling, and the largest connected component. This follows the existing
MATLAB/single_net algorithm; SciPy and MATLAB boundary-pixel morphology need not be
pixel-identical. Alternative `_dfs`, `_dfd` and unsuffixed outline names are paired
by existence, instead of the MATLAB loop-index special cases. Numeric subject
prefixes stay together across multiple images of the same category.

**AVRDB.** Source vessel drawings are dark pixels on white (`red<128`);
artery drawings use `blue<128`, vein drawings `red<128`, matching the newer Python
reader. Accepted class spellings include `artery`, `arteries`, `atertries`, `vein`,
`veins`, `veinds` and `veisn`. Sparse line drawings do not establish exhaustive
background truth: vessel validity is restricted to marked positive vessel pixels,
and A/V validity to pixels belonging to exactly one class on vessel support.
Crossings, uncertain class pixels and all unmarked pixels are ignored. These data
cannot independently teach a model to distinguish vessels from background. Vessel
metrics on marked positive support do not measure full-image segmentation accuracy.

**G1020.** LabelMe x/y coordinates are rasterized in their native zero-based image
frame using Pillow polygons. The old Python reader subtracted1 without evidence
that LabelMe coordinates are MATLAB indices; that shift is not retained. Three
source JSONs duplicate the label `disc` (`image_1956`, `image_2085`, `image_2507`).
The strictly nested pairs in `image_1956` and `image_2507` are resolved by filled
area and nesting, recording the correction. In `image_2085`,2.44% of the inner
polygon lies outside the outer polygon, so the record is excluded from training
and retained with its JSON for review instead of guessing a corrected label. They are not assigned by the number of polygon
points. Repeated `discLoc` boxes do not define segmentation masks and are ignored.
Missing cup/disc channels remain invalid. Cup pixels outside a supplied disc remain
in the target for traceability and are excluded from cup validity.

**ONHSD.** `ONHCentre` and 24 `ONHEdge` radial distances reconstruct the disc with
angles `pi/12 * [1..24]`. Centers are converted from MATLAB's one-based coordinates,
vertices rounded, and polygons rasterized with Pillow. AnsuONH, Bob, David and Lee
remain separate observations. Nonfinite/nonpositive radii or collapsed rounded
vertices are rejected as annotations, preserving their original center/radial MAT
files and a review reason. A zero-radius annotation is not a normal negative disc.
The real 99-image archive contains 53 such invalid clinician annotations among 396.

**RIGA.** The archive supplies original photographs and marked photographs rather
than clean segmentation maps. Recovery starts from the legacy single_net operation
`original_green - marked_green >15`. Eight-connected line components must enclose
an area after hole filling. If the original thresholded contour is open, the
smallest successful diamond closing radius 1, 2 or 3 is tried, following the bounded
morphology used by the legacy MATLAB conversion. The selected radius is recorded. Disc/cup assignment uses filled area and nesting rather
than line length. A small tolerance (at most 1% of cup pixels, with 2-pixel floor)
allows raster boundary irregularities; retained cup is intersected with disc and
the count is recorded. Open curves, merged unrecoverable contours and ambiguous
pairs are flagged for review. The reader never invents an ellipse or substitutes an
unverified predicted mask. All six observers, including failed reconstructions,
remain separate records grouped by the primary photograph. The original marked
raster is retained for failed recovery. This is an explicitly checked recovery
algorithm, not a claim of exact agreement with hand-created RIGA masks.

**RIM-ONE release 3.** Photographs and masks retain the full stereo frame. Only the
left half is valid; the second view has no supplied mask and must not be a negative
training region. Both experts use the source `-1-` filename marker; that marker does
not select the expert number. Average masks and both expert masks are retained
separately, grouped by patient prefix across left/right eyes. Source cup pixels
outside the corresponding disc are excluded from cup validity rather than silently
expanding a clinician's disc annotation.

**STARE.** AH and VK annotations remain separate. Missing annotation for a photograph
produces `task=images`, `target=None`; it is available for prediction or an explicitly
chosen image-only workflow and excluded from supervised training.

**UoA-DR.** Masks use red-channel `>127`. Image 72 has vessel `.2.jpg` and disc
`.1.jpg`; all other images use vessel `.1.jpg`, disc `.2.jpg`. This follows the
MATLAB image72 correction, which the single_net loader omitted.

**Important legacy-label correction:** `.3.jpg` is an **optic-disc center marker**,
not a cup segmentation. The dataset's bundled primary publication, *Comparative
Analysis of University of Auckland DR Database*, Section3, describes the three
sub-images as “manually segmented retinal vessel, optic disk and optic disk center”.
The actual source rasters match that description. Both old MATLAB and single_net
loaders mistakenly named the center marker a cup. The new adapter supplies disc
supervision with **cup validity entirely false**. The original `.3.jpg` is preserved
as an auxiliary `optic_disc_center` raster (`preserve=True`), together with its
marker centroid in original zero-based x/y coordinates. The centroid is a summary
of the supplied marker, not a claim of a new cup boundary.

The same publication describes the vessel masks as manually segmented ground truth;
this adapter retains their supplied dense-mask semantics. Thin-looking trees alone
are not treated as evidence that all unmarked pixels should be ignored. The legacy
bottom-left timestamp region (`rows>=round(.97H), columns<round(.25H)`) is ignored
through FOV validity. The photograph is retained unchanged; that region does not
become artificial black training input.

## Verification

Regression tests cover missing-channel validity, G1020's nested duplicate labels,
ONHSD coordinate conversion and invalid radii, RIGA closed-contour recovery and
failure detection, UoA image 72's swapped masks/timestamp exclusion, AVRDB sparse
support and misspelled filenames, and RIM-ONE stereo validity/observer grouping.
Executed real-data preparation counts and any rejected annotations are recorded
below after the full conversion passes; unsupported or failed annotations are not
counted as successfully decoded labels.


### Completed full real conversions at longest side 128 (2026-11-10)

| Source | Records | Usable labels / image-only / rejected |
|---|---:|---|
| ARIA | 406 | 286 vessel annotations  + 120 disc annotations; all usable|
| AVRDB | 200 | 100 sparse vessel  + 100 sparse AV annotations; all usable on marked support|
| ONHSD | 396 | 343 valid clinician disc annotations ; 53 rejected radial annotations retained|
| RIM-ONE release 3 | 477 | 159 photographs × average/expert1/expert2; all usable|
| STARE | 417 | 40 vessel annotations on 20 photographs ; 377 image-only photographs|
| UoA-DR | 400 | 200 vessel + 200 disc-only records; center markers retained |

Outputs are in `data/migration/segmentation-quick-128-v2` and
`data/migration/segmentation-rest-128-v2`, outside Git. The latter
contains a superseded UoA decoding pass; corrected UoA outputs were fully regenerated
in `data/migration/segmentation-uoadr-128-v3` (114 seconds) and replace those UoA rows.
All 2,296 records passed
array shape/type/range and source-group split checks; usable disc/cup pixels obey
cup-within-disc validity. No usable record had an entirely empty valid target.
Invalid source records are listed separately above and excluded from that claim.
G1020 full conversion completed in 545 seconds: 1,020 records, 1,019 usable,
with `image_2085` explicitly invalid and its original JSON retained. It supplies
1,018 valid disc channels and 793 valid cup channels (missing channels stay invalid);
no usable record had an entirely empty target. Output:
`data/migration/segmentation-g1020-128-v3`. Including G1020, the verified total is
**3,316 records** across seven adapters: 2,885 usable dense annotations, 54
quarantined dense annotations, and 377 image-only STARE photographs. Every corrected
UoA disc record has an all-false cup channel; all 200 original center-marker JPEGs
and all 200 derived marker centroids were checked. RIGA full-pass
results are recorded below.

The bounded RIGA recovery fallback was checked against a synthetic reference:
a small missing contour pixel is recovered with disc/cup IoU above 0.98, while a
large missing arc remains rejected. The actual BinRushed1-Corrected image 13's six
annotators all failed the original unclosed threshold and recover two nested
closed regions using radius 1 or 2. This establishes improved annotation recovery;
it is not a measured model-accuracy improvement.


The recovery implementation also uses an exact nonzero bounding-box crop with
six zero-support pixels around the ink before morphology (maximum diamond radius
three, dilation followed by erosion). It restores the original full-frame masks.
This is not an estimated optic-disc crop. Forty-eight actual successful annotation
pairs spanning all seven RIGA folders produced **bitwise identical** masks and
identical quality metadata before/after this optimization. Twelve actual rejected
pairs remained rejected. Measured morphology/recovery time for these 60 pairs
fell from 11.16 seconds to 1.67 seconds; timings exclude annotation I/O and the shared
writer. Reports are `data/migration/riga-crop-equivalence.json` and
`data/migration/riga-crop-equivalence-rejected.json`.

### Final RIGA pass and complete eight-adapter inventory

The full final RIGA conversion completed in **1,607.95 seconds** at longest side
128: **749 original photographs × six observers = 4,494 records**. Of these,
**2,681 annotations reconstructed two checked nested closed regions** and are
usable; **1,813 are explicitly quarantined**, with all-false validity and their
original marked rasters preserved for review. Unrecoverable line annotations are
not reported as successfully reconstructed segmentation masks.

Usable RIGA records split into 1,673 training, 508 validation and 500 test observations;
quarantined records split into 1,021/392/400. Every original photograph and all its
observers stay in one split. Among successful reconstructions, 2,422 required no
closing, 111 used radius 1, 109 radius 2 and 39 radius 3. All 4,494 records passed shape,
dtype/range and source-group integrity checks. Every usable record retained both
nonempty disc and cup at the deliberately small smoke resolution; every rejected
record's preserved raw annotation was checked to exist. Final output and report:
`data/migration/segmentation-riga-128-v4` and
`data/migration/riga-final-integrity.json`.

Across a fixed 1,113-record comparison against the unclosed recovery algorithm,
692 existing successful records remained **bitwise identical** in target,
validity and FOV arrays. Bounded closing recovered 136 previously rejected records,
while 285 remained quarantined. Usable annotations increased 692→828 (+19.7%);
no previously successful reconstruction was lost. This establishes an annotation
recovery improvement, not a model-accuracy gain. The comparison is recorded in
`data/migration/riga-bounded-recovery-comparison.json`.

All eight migrated adapters have now completed full real-source passes:
**7,810 records = 5,566 usable dense annotations + 1,867 explicitly quarantined
dense annotations + 377 image-only photographs**. The quarantined dense total is
1,813 RIGA outlines, 53 ONHSD radial annotations and one ambiguous G1020 JSON.
Decoded labels, missing channels, source quality and auxiliary center annotations
remain distinguished in the manifests.
