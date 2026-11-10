# Corpus v2 archive audit: proposal, not current training data

The existing immutable corpus v1 contains17 selected dense-supervision sources.
It does **not** cover every suitable annotation in all available archives. Source3157,
canonical6103, its split memberships and current HPO remain unchanged. This audit
reads archive central directories first; real annotation inspection is confined to
TRAIN/unsplit proposed-training resources. No TEST masks or scores are read.

[Reproducible central inventory](../reports/data/corpus-v2-archive-audit/central-inventory.json)
records archive sizes, logical/transport counts and name candidates. It does not
call every mask-like filename suitable dense supervision. Duplicated long/short
archive transport paths must not produce duplicated patients or records.

| Source | Available relevant annotations | v2 disposition |
| --- | --- | --- |
| DRIONS-DB |110 fundus photos,220 ordered36-point OD contours, two observers | Native disc-only decoder; cup invalid; train-only without patient crosswalk. Physical field angle unresolved. |
| GAMMA |100 official TRAIN fundus photos and100 disc/cup masks;25600 OCT slices and100 diagnostic overlays are separate | Native fundus decoder only, official training/train-only; patient crosswalk and physical acquisition angle unresolved. |
| BIOMISA | Fundus_Disc/Cup raster labels exist in addition to OCT layers | Audit matching, actual format and semantics before admission. Sample `.jpg` is actually PNG RGBA with filled white region; named Cup is larger than named Disc. No silent rename by size. |
| IDRiD (registered but excluded from v1 dense sources) |54 official TRAIN OD TIFFs and27 official TEST OD TIFF names | Missed disc-only supervision. TRAIN decoding can be added; TEST membership remains held out and labels unopened in this audit. |
| ACRIMA |705 cropped images with filename glaucoma classification | No native dense vessel/AV/disc/cup annotations found. |
| Kaggle |1000 images in disease-category folders | Classification labels do not supply dense five-head GT. |
| DB_DBMI_Canon_Healthy |87 fundus images | No dense annotation resources in archive. |
| HEI-MED | Lesion GT, ONH location, machine-segmented vessels | Lesions/ONH points are not disc segmentation; machine masks are not human vessel GT. |
| Other registered sources | Covered by existing adapters/auxiliary preservation | Recheck individual target semantics rather than treating registration as dense coverage. |
| `ziajP7YB` | Starts as ZIP but missing readable central directory | Inventory incomplete; do not claim this damaged/incomplete export was audited or repair original in place. |

The [DRIONS creator page](https://www.ia.uned.es/~ejcarmona/DRIONS-DB_2.html)
confirms110 scanned600×400 fundus photos and independent OD contours from two
experts. Bundled `_Leeme.txt` gives36 X/Y points; its MATLAB viewer plots those
coordinates over `imshow`, establishing one-based pixel centers. Rasterization
must declare its polygon convention; it is not recovered vendor binary GT.

The [GAMMA organizer paper, section2.1](https://arxiv.org/html/2202.06511)
describes paired eyes among276 patients,300 samples, three100-sample challenge
partitions and specialist-fused disc/cup labels. Patient IDs are not recovered
from anonymous sample numbers. It identifies Topcon TRC-NW400 and an unspecified
KOWA camera, but does not establish each photograph's field angle. Do not silently
assign45° or claim verified25px/degree calibration. Native raw adapters can retain
RGB/GT without physical scaling; legacy25_raw v2 preparation requires an explicit,
documented calibration policy before a training-ready manifest is approved.

BIOMISA inspection must distinguish its fundus masks from OCT labels, and PNG
content from the misleading `.jpg` extension. Raw labels and contradictory cases
remain preserved/flagged, rather than being guessed into negative supervision.

Exact decodedRGB photo identities and known patient groups must be checked against
all immutable v1 memberships. Any v1 validation/test alias stays held out or is
quarantined; new unknown-patient sources stay train-only. An exact photo check
cannot prove absence of re-encoded/resized aliases or unknown patients. A future
v2 must state those remaining limits and cannot silently overwrite v1 artifacts.
