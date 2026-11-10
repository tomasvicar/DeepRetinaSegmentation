# Auxiliary and image collection adapters

The Python adapters in `src/deepretina/adapters/auxiliary.py` replace the legacy
MATLAB collection loaders. They read archives or directories through `Source`;
raw archives remain unchanged. Their `ADAPTERS` mapping contains the selectors
below and accepts `(source, emit)`.

`task="images"` means an image has **no vessel/disc/cup training target**. Such
images do not become negative segmentation examples and do not require the
missing historical pretraining database. Auxiliary annotations retain their
original meaning. The common preparation writer copies `annotation_resources`
into the prepared output; metadata also records original resource paths.

| Selector | Image selection | Labels retained |
| --- | --- | --- |
| `diaretdb0` | `diaretdb0_fundus_images` PNGs | Lesion `.dot`/groundtruth resources; original script's image-number diagnosis heuristic, explicitly named `legacy_diagnosis` |
| `diaretdb1` | `ddb1_fundusimages` PNGs | Separate lesion groundtruth resources and original healthy-image-number list |
| `eophtha` | EX, MA and healthy subject folders | Lesion group, available EX/MA annotation resources; grouped by subject folder |
| `erlangen` | Actual PNG/TIFF/JPEG `FundusImages`, excluding FOV files | **Manual vessel supervision** from corrected-mask green channel; unmatched originals stay image-only |
| `fire` | Original A/P/S image pairs in `Images` | Supplied FOV and registration control points `(x1,y1,x2,y2)`; pair grouping |
| `heimed` | JPEG originals in `DMED` | GND/meta/map/derived-resource references; derived vessel images are not asserted to be manual masks |
| `idrid` | C. Localization originals | Original optic-disc/fovea centers and CSV resources; official train/test preserved |
| `inspireavr` | `org` JPEGs | Both observers' scalar AVR measurements and GT table |
| `messidor` | Original TIFFs | Retinopathy grade and macular edema risk from XLS tables, using `xlrd` |
| `rcrgbma` | `Original` images | Both experts' microaneurysm mask resources |
| `review` | HRIS, VDIS, CLRIS and KPIS | Sparse vessel-edge positions for three observers and original tables |
| `ridb` | Original JPEGs | Identity grouping from filename prefix; repeated captures remain together |
| `roc` | Training/test image filenames | Microaneurysm point, radius, lesion and probability fields from consensus XML; official split preserved |
| `vicavr` | `images/original` | Expert vessel points, calibers and class values; no dense masks inferred |
| `ubmi` | User-supplied `Gacr*` original images | Optional existing `ImageAnalysis/VesselsSeg/*whole.png` resources for inference, original geometry and hysteresis FOV |

Unsplit collections emit `split="unsplit"`, allowing the common writer to create
deterministic grouped partitions. No patient identities are inferred where the
source does not supply them. Erlangen groups by patient across visits and eyes;
FIRE by pair; E-Ophtha by subject folder; RIDB by recognition identity.

Erlangen's actual archive differs from the old hard-coded MATLAB JPEG glob:
originals are PNGs with patient/visit/eye identifiers. Pairing first uses all of
those fields. A legacy collapsed identifier is accepted only if it identifies a
single corrected mask. Ambiguous matches raise an error. Corrected masks use
green-channel thresholding and nearest-neighbor resizing if needed; images with
no corrected mask retain `task="images"`.

All decoded coordinates refer to **original image pixels**. Resizing a prepared
image does not silently rescale sparse measurements; consumers must apply the
recorded preparation geometry when drawing those measurements. Sparse widths,
center points, lesion masks and scalar AVR values are distinct annotations and
are not interchangeable with dense vessel, artery/vein, disc or cup labels.

The local VICAVR files contain numeric class `3` at 31 measurement points despite
the header documenting only `0` (unclassified), `1` (artery), and `2` (vein).
The adapter preserves `3` as `undocumented_source_value`; it does not guess a
class or discard those measurements.

UBMI is an internal filesystem interface, not a claim that its raw data are
available here. Supply the original `Gacr*` folders and corresponding JPG/PNG
photographs. A missing collection or ambiguous whole-mask pairing raises an
actionable error. Optional whole-vessel segmentations have unverified manual
provenance, so UBMI stays image-only. The FOV uses grayscale hysteresis with the
historical image-specific intensity thresholds, largest-component selection and
hole filling. Historical local-contrast transformation and fake two-slice
NIfTI/DICOM exports are superseded by ordinary RGB images and metadata.

Synthetic regression tests cover every selector, missing-layout errors, mask
pairing, source grouping, annotation preservation, official splits and the
absence of fabricated segmentation labels. Real archive decoding checks and
local preview examples are recorded under `data/migration/auxiliary_validation`;
those artifacts are excluded from Git. UBMI is validated synthetically because
its original raw files are not present in the supplied database directory.

## Verified local archive inventory (2026-11-10)

A full decoding pass visited every emitted image in all fourteen locally supplied
archives, checked RGB dtype/shape and supervised-mask/FOV geometry, parsed the
available annotation tables, and wrote previews of the first three records.
This is an adapter/data-integrity check, not a model accuracy experiment.

| Selector | Unique originals decoded | Supervised vessels | Split notes |
| --- | ---: | ---: | --- |
| `diaretdb0` | 130 | 0 | Unsplit |
| `diaretdb1` | 89 | 0 | Unsplit |
| `eophtha` | 434 | 0 | Identical repeated originals deduplicated; duplicate basename copies had matching archive CRC/size |
| `erlangen` | 146 | 106 | Remaining 40 originals have no matching corrected manual vessel mask |
| `fire` | 268 | 0 | Registration pairs stay together |
| `heimed` | 169 | 0 | Existing derived vessel resources stay auxiliary |
| `idrid` | 413 | 0 | 310 official training, 103 official testing |
| `inspireavr` | 40 | 0 | Both observer measurements checked |
| `messidor` | 1,200 | 0 | Every image matched an XLS diagnosis row |
| `rcrgbma` | 250 | 0 | 500 expert microaneurysm-mask references |
| `review` | 16 | 0 | Includes the two KPIS images omitted by the old loader |
| `ridb` | 100 | 0 | Repeated identity captures grouped |
| `roc` | 100 | 0 | 50 official training, 50 official test |
| `vicavr` | 58 | 0 | 11,448 preserved expert point measurements |
| `ubmi` | Not locally available | 0 | Filesystem contract and optional-resource handling tested with synthetic originals |

The full-pass JSON records retain the resource-reference counts seen at execution.
A subsequent small reporting fix deduplicates FIRE's repeated transport-prefix
copies of the same control-point file; a dedicated regression test verifies one
reference per image while keeping different lesion categories separate.
