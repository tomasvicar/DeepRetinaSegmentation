# Complete legacy adapter migration audit

Audit date: 2026-11-10. This inventory covers the exact `load_*.m` union from
`master`, `artery_vein_class`, `seg_videa`, `segment_sada01`, `segment_sada02`,
and `single_net`, plus the concrete `single_net/datasets_loaders/Loader*.py`
classes. There are **35 MATLAB paths**, **14 Python loader paths**, and **31**
canonical dataset selectors. `LoaderGeneric` is infrastructure, not a dataset.
RITE is the separate selector for the old DRIVE A/V loader. UBMI is included
even though its complete external source collection is unavailable locally.

The machine-readable source map is
[`coverage.json`](../src/deepretina/adapters/coverage.json). Its status fields
distinguish inventory, implementation and actual validation. A registered selector
alone does not demonstrate successful data conversion. Tests keep every original
loader represented and prevent sparse/image-only sources from being advertised
as dense vessel labels.

## Dataset contracts

| Selector | Required source | Historical annotation semantics and pitfalls |
| --- | --- | --- |
| `aria` | `ARIA.zip` | Two vessel annotations (BDP/BSS); disc/fovea markups need disc-only contour recovery. Preserve each annotator and shared image source. |
| `avrdb` | `AVRDB.zip` | Separate vessel, artery and vein JPEG line annotations on white; handle spelling variants and mark sparse supervision explicitly. |
| `chasedb1` | `CHASEDB1.zip` | Two vessel annotators; paired child eyes must share partitions. |
| `diaretdb0` | `DiaRetDB0.zip` | Fundus PNGs used by legacy; lesion annotations/diagnoses are auxiliary and not vessel negatives. |
| `diaretdb1` | `DiaRetDB1.zip` | Fundus PNGs used by legacy; separate lesion masks/MAT are auxiliary. |
| `drhagis` | `DRHAGIS.zip` | Binary vessel masks; new RGB A/V optional in supplementary ZIP; old grayscale outputs half resolution. |
| `drishti` | `Drishti-GS.zip` | Disc/cup consensus soft maps; explicitly document binary threshold. |
| `drive` | `DRIVE.zip` | Official training/test image and vessel/FOV mask pairs. |
| `eophtha` | `E-Ophtha.zip` | EX/MA and healthy photographs; lesion masks are auxiliary, not vessel/disc labels. |
| `erlangen` | `Erlangen.zip` | FundusImages PNG; corrected RGB vessel TIFF green channel; match patient/visit/eye and keep repeated views together. |
| `fire` | `FIRE.zip` | Registration photograph pairs and official FOV mask; correspondence TXT is auxiliary. |
| `g1020` | `G1020.zip` | LabelMe-like JSON polygons; missing cup/disc is invalid supervision, not empty background. |
| `heimed` | `HEI-MED.zip` | DMED photographs; edema masks/metadata auxiliary. Ignore generated previews and DICOM gzip when selecting originals. |
| `hrf` | `HRF.zip` | Binary vessel/FOV masks; optional new RGB A/V; numeric grouping across disease categories is conservative. |
| `idrid` | `IDRiD.zip` | Legacy selected C.Localization photos; archive A lesion masks/B grading/C localization CSV are auxiliary, not vessel targets. |
| `inspireavr` | `INSPIRE-AVR.zip` | Original photographs and scalar AVR GT; no dense A/V reference. |
| `iostar` | `IOSTAR.zip` | Binary vessel/FOV plus RGB A/V; TIFF colors may contain compression artifacts. |
| `messidor` | `MESSIDOR.zip` | Original TIFF photographs and XLS grading, with duplicate nested ZIPs; do not count nested duplicates again. |
| `onhsd` | `Onhsd.zip` | Disc center + 24 polar radii in clinician MAT files; convert one-based center to Python coordinates, retain four annotators. |
| `rcrgbma` | `RC-RGB-MA.zip` | Original photographs plus microaneurysm PNG labels; auxiliary lesion masks are not vessel labels. |
| `refuge` | `REFUGE.zip` | 255 background,128 disc,0 cup; official400/400/400 partitions. |
| `review` | `Review.zip` | Sixteen photographs; observer TXT contains sparse paired vessel-edge coordinates/widths, not complete dense masks. |
| `ridb` | `RIDB.zip` | Image-only legacy exporter; no supplied dense reference. |
| `riga` | `RIGA/` | Extracted prime photographs plus six drawn-contour images. Recover closed regions, order by filled area/nesting, report failures. |
| `rimone` | `RIM-ONE.zip` | Legacy r3 stereo frame with cup mask; archive also has disc masks. Retain original full frame or record half-image transform. |
| `rite` | `RITE.zip` | A/V on DRIVE source photographs; shared source IDs prevent cross-dataset leakage. |
| `roc` | `ROC.zip` | Photographs and XML microaneurysm consensus points (not MAT in supplied ZIP); preserve official training/test. |
| `stare` | `STARE.zip` | All-image PPM corpus and AH/VK vessel annotations. Unlabeled images stay image-only; upstream gz variants optional. |
| `ubmi` | `UBMI/ or Sada_01/ (external)` | Legacy own-image inference folders, not A/V GT. Full Sada_01 collection unavailable; explicit external input contract required. |
| `uoadr` | `UoA_DR.zip` | Vessels/disc/cup JPEGs; image72 swaps .1/.2; mask burned-in footer. Preserve patient/source identity. |
| `vicavr` | `VICAVR.zip` | Original photographs plus three experts sparse coordinate/width TXT/HTML; no complete dense A/V mask. |

## Independent local evidence

- Archives are read-only inputs. ZIP central-directory inventories were inspected
  without full extraction. Several contain both full shared-drive prefixes and
  short duplicate exports; raw member count is not unique image count.
- Erlangen contains146 FundusImages PNGs and108 corrected vessel TIFFs, plus512
  average-image TIFFs. A sample image and mask are770x1000 RGB. Corrected masks
  contain black, green and white pixels; the old green-channel extraction includes
  both green and white. The old JPG glob does not match these supplied PNGs.
- RIGA currently exists as an extracted tree:5550 files,3933 TIFF and1617 JPEG,
  with749 `prime` photographs. No nested archives were observed in that tree.
  Prime counts: MESSIDOR460; BinRushed1-Corrected50, BinRushed2/3 each47,
  BinRushed4 51; Magrabia male/female each47. These are inventory observations,
  not successful contour-extraction counts.
- A sampled RIM-ONE r3 stereo image and cup reference both measure1424x2144.
  The reference is already a full-frame binary mask. Never stretch it across a
  cropped half image or invent labels for an unannotated half.
- STARE supplied ZIP contains expanded PPM photographs and `labels-ah`,
  `labels-vk`, with no gzip members. Filename matching must distinguish original
  and annotated PPM files. Some originals have no manual reference.
- REVIEW supplied TXT includes image/segment IDs and right/left edge x/y pairs
  for three observers. These are sparse width references; treating all other
  image pixels as background would fabricate dense supervision.
- ROC supplied archive contains200 JPG members with duplicate exports and two
  XML files. Consensus annotations are microaneurysm points; no MAT is present.
- ONHSD includes large quantities of unrelated algorithm output MAT files.
  Select only `Images` and the matching `Clinicians` center/radii resources.
  Duplicate polygon points or nonpositive radii require explicit quality handling.
- G1020 valid ZIP contains1020 JPG plus1020 JSON plus one CSV. The extensionless
  corrupt duplicate is unnecessary. Duplicate polygon labels in old code were
  guessed by point count; a faithful rewrite must record/resolve ambiguity using
  geometry, rather than silently declaring missing anatomy negative.
- Full UBMI/Sada_01 layout is not supplied. Five source photographs and manual
  vessel inputs under `from_pascal` are annotation-input examples, not recovered
  artery/vein labels or the entire legacy dataset.

## What complete migration means

Every historical source loader needs an executable Python path for the original
images and annotations it actually consumed. Image-only loaders should produce
image records and preserve auxiliary references, with no invented all-zero target.
Partially annotated supervised tasks need false validity for missing channels.
Multiple annotators need distinct records tied to the same source split; their
outputs must not be overwritten by a common filename. Sparse labels need explicit
support masks or separate auxiliary resources, not false full-image negatives.

Legacy scale/FOV/normalization helpers are replaced by the documented Python
preprocessing contract; numerical equivalence to every MATLAB heuristic is not
claimed. The large missing pretraining corpus remains unnecessary. Existing
external data must never be modified while preparing local outputs.

## Integration verification

`tests/test_adapter_coverage.py` protects the exact historical loader inventory,
CLI selector coverage and sparse-reference task contract. Adapter-specific tests
must additionally verify pixels, validity, source grouping and real archive
pairing. Record actual commands, output counts and failures before changing an
adapter status to validated. Contour recovery (RIGA/ARIA), polar coordinates
(ONHSD), sparse JPEG lines (AVRDB), stereo geometry (RIM-ONE) and the UoA image72
exception deserve focused checks. A missing external UBMI source is reported as
such and must never be presented as a successful real-dataset validation.

### Decoder sample check executed

An independent read-only check invoked all22 new locally available decoder
functions against actual source ZIPs/directories, stopping after two emitted
records per selector. All44 samples passed image/target shape checks and target
finite/range checks. This covered the eight new segmentation adapters and14
auxiliary adapters; complete UBMI was skipped because its source is unavailable.
`coverage.json` records sample identities, shapes, foreground/support counts and
raw-resource reference counts. The check tested decoder outputs rather than
writing every dataset, so those entries are **implemented and sample checked**,
not automatically marked fully validated. Full conversion and integration-test
results are recorded separately as they complete.

The review also corrected two semantic problems before integration: AVRDB now
uses positive-only vessel support on its sparse manual drawing, and UBMI retains
existing vessel predictions as inference resources rather than promoting them
to verified manual training labels. Missing ARIA vessel resource references were
added; RIM-ONE/UoA disc/cup resource preservation was requested after inspection.

### Integrated preparation check executed

The independent check next called the shared `prepare` API for each of22 new
local selectors with `max_size=64, limit=3`, writing only beneath
`/tmp/deepretina-integration-_4cvceet` (never into ownCloud). All22 calls succeeded.
Written images/masks were reopened; dimensions and validity markers were checked.
Every annotation reference either resolved to an existing copied local file or
explicitly identified its preserved decoded target/valid arrays. Preview
provenance correctly records that these outputs are limited previews, not full
training datasets. The JSON inventory records these bounded validation scopes.

| Selector | Records | Source groups | Dense records | Copied reference uses |
| --- | ---: | ---: | ---: | ---: |
| `aria` | 12 | 3 | 12 | 0 |
| `avrdb` | 6 | 3 | 6 | 0 |
| `diaretdb0` | 3 | 3 | 0 | 3 |
| `diaretdb1` | 3 | 3 | 0 | 12 |
| `eophtha` | 6 | 3 | 0 | 9 |
| `erlangen` | 6 | 3 | 5 | 0 |
| `fire` | 6 | 3 | 0 | 6 |
| `g1020` | 3 | 3 | 3 | 3 |
| `heimed` | 3 | 3 | 0 | 12 |
| `idrid` | 3 | 3 | 0 | 6 |
| `inspireavr` | 3 | 3 | 0 | 3 |
| `messidor` | 3 | 3 | 0 | 3 |
| `onhsd` | 24 | 3 | 24 | 48 |
| `rcrgbma` | 3 | 3 | 0 | 6 |
| `review` | 3 | 3 | 0 | 3 |
| `ridb` | 60 | 3 | 0 | 0 |
| `riga` | 18 | 3 | 18 | 0 |
| `rimone` | 9 | 3 | 9 | 0 |
| `roc` | 3 | 3 | 0 | 3 |
| `stare` | 6 | 3 | 6 | 0 |
| `uoadr` | 6 | 3 | 6 | 0 |
| `vicavr` | 3 | 3 | 0 | 9 |

Record counts can exceed three because the limit applies to source groups,
retaining all included eyes, tasks and annotators. RIDB groups repeated captures
by recognition identity; ONHSD retains both eyes/four observers for each included
identity. A bounded preview cannot establish successful conversion of every
annotation in the full corpus; full-run results remain separate evidence.

Additional exact legacy coverage fixes identified during review: DRIVE test
second-observer masks, CHASE second-observer masks, and IOSTAR optic disc masks.
All30 actual IOSTAR OD-mask TIFFs were inspected: invert the logical mask, label
8-connected components and remove every component touching the image border.
This yields exactly one disc per image, area22667–32648 original pixels. Twenty-nine
masks have two inverse components and one has three, so blindly selecting label2
is brittle. Cup is absent and must have false channel validity. Boolean TIFFs
must be converted consistently to L0/255 before intensity thresholding.

### Final source-label correction

The bundled UoA comparative-analysis paper, section3, identifies the third derived
image as an optic-disc center marker. The historical cup interpretation is incorrect.
The migrated adapter keeps vessels and disc supervision, marks cup invalid, and retains
all center-marker rasters and their original-frame centroids.
