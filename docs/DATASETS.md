# Data preparation and annotation contract

`uv run deepretina prepare --help` documents the command. The Python entry point is
`deepretina.prepare.prepare(data_root, output, datasets, max_size=1024, seed=42)`.
Data remain outside Git. ZIP members are decoded directly: no MATLAB, HDF5
conversion, large intermediate extraction, pretraining database or network access
is required. Dataset selectors are lowercase. All 31 historical selectors are implemented; use
`uv run deepretina datasets` for the packaged coverage inventory.

## Choose the preparation contract

This page's `max_size`, complete-frame and heuristic-FOV descriptions apply to
**`deepretina prepare`**, the basic converter used by the pilot trainers. They do
not describe canonical data required by `train-unified` or `train-legacy`.

For complete native/unified retraining, start from RAW with
[`scripts/prepare_training_data.py`](UNIFIED_QUICKSTART.md#prepare-all-21-annotated-training-sources-from-raw).
That workflow jointly prepares all 21 training sources on the explicit
`legacy25_raw` grid: source-specific camera angle, FOV/crop geometry and
25 pixels/degree. Targets and validity receive the same geometry. See
[canonical data](UNIFIED_DATA.md) and the
[V2 nominal-angle qualifications](CORPUS_V2_PREPARATION.md).
Do not substitute `deepretina prepare --max-size 1024` for this step.

## Implemented adapters

| Selector | ZIP | Tasks | Source images | Split policy |
|---|---|---|---:|---|
| `drive` | DRIVE.zip | vessels | 40 | Official 20 test; 16 train / 4 validation |
| `rite` | RITE.zip | vessels + artery/vein (one AV record) | 40 | Same images and grouping as DRIVE |
| `hrf` | HRF.zip | vessels; AV when new annotations ZIP exists | 45 | Group numeric image prefix, deterministic 60/20/20 groups |
| `drhagis` | DRHAGIS.zip | vessels; AV when new annotations ZIP exists | 40 | Deterministic 60/20/20 image groups |
| `refuge` | REFUGE.zip | disc and cup | 1200 | Official 400/400/400 train/validation/test |
| `drishti` | Drishti-GS.zip | disc and cup | 101 | Official 51 test; 40 train / 10 validation |
| `chasedb1` | CHASEDB1.zip | vessels | 28 | Both eyes of each of 14 children stay together; 60/20/20 groups |
| `iostar` | IOSTAR.zip | vessels, AV and disc (cup unannotated) | 30 | Deterministic 60/20/20 image groups |

Counts describe the local archive inventory, not a promise that other upstream
archive layouts are identical. Fractional split sizes are rounded at group level.
HRF numeric grouping is conservative: matching numbers across diagnostic categories
stay together, without assuming the archive provides verified patient identities.
IOSTAR/DRHAGIS only expose image-level identifiers here; patient-level independence
cannot be asserted. The manifest preserves `source_id` to make these decisions
reviewable. A/V and binary records of the same image share one source group.
DRIVE and RITE refer to the same photographs and should not be reported as
independent datasets; their train/validation assignments are explicitly shared.

HRF/DRHAGIS new AV labels come from the optional
`drive-download-20261019T122612Z-1-001.zip`, under
`HRF/clasified/<id>.png` and `DRHAGIS/clasified/<id>.png`.
The old grayscale half-resolution labels in the original archives are deliberately
not substituted. If that ZIP is absent, vessel preparation still works and produces
no HRF/DRHAGIS AV records. Check manifest task counts before launching AV training.
The newer annotation set contains 45 HRF and 16 DRHAGIS labels.

Archives contain duplicate copies (full Google Drive prefix and relative paths).
The loader indexes by path suffix and selects the shortest matching path. It does
not count duplicate exports as new images. Filenames are paired explicitly;
missing expected masks and unexpected dimensions raise an error.

## Labels, geometry, and validity

Each output RGB PNG preserves the complete original image frame and aspect ratio,
with longest side capped at `max_size`; images smaller than this are not enlarged.
Lanczos resamples photographs; nearest neighbour resamples all label/validity masks.
There is no assumed field-of-view angle, image rotation or label interpolation.
This replaces the old MATLAB conversion rather than claiming numerical equivalence
to MATLAB `imresize` and the database-specific `souradnice` heuristics. The recorded
`original_size` is `[height, width]` before resize.

Every NPZ contains:

- `target`: float32 `[channels, height, width]`, binary 0/1.
- `valid`: bool with the same shape. Losses and metrics must exclude false entries.
- `fov`: bool `[height, width]`. Official FOV masks are used for DRIVE, HRF,
  DRHAGIS and IOSTAR. Otherwise `max(R,G,B) > 10` estimates visible image pixels;
  this is a heuristic and may exclude very dark pixels.

Channel order is `[vessel]`, `[vessel, artery, vein]`, or `[disc, cup]` according to
`task` (`vessels`, `av`, `disc_cup`). Disc includes cup. REFUGE values `<150` mark
disc and `<50` mark cup. Drishti soft consensus maps are thresholded at `>127`,
then cup pixels are included in disc; soft annotation uncertainty is not modelled.

A/V RGB labels use red for artery, blue for vein. Pure green, white, black inside
binary vessels and all other ambiguous colours are excluded from A/V supervision.
The original binary vessel annotations provide the independent vessel channel,
so unknown A/V pixels still train vessel segmentation. A/V losses and metrics are
restricted to confidently classed vessel pixels; background is handled by the
separate vessel channel and masks final class predictions. RGB detection requires
dominant channel `>180` and other channels `<100`, tolerating IOSTAR compression
artifacts. This threshold is an explicit implementation choice, not a measured
accuracy improvement. Artery/vein crossovers are ignored rather than assigned a
single incorrect class.

The manifest is a JSON **list**. Paths are relative to the manifest's directory:

```json
{
  "id": "rite_21_training_av",
  "dataset": "rite",
  "split": "train",
  "image": "images/rite_21_training.png",
  "mask": "masks/rite_21_training_av.npz",
  "task": "av",
  "source_id": "drive:21",
  "original_size": [584, 565],
  "preparation_max_size": 1024
}
```

Original decoded RGB dimensions and pixels define `photo_id`, independently of
filenames, annotation task, FOV mask and resize cap. Patient/source IDs and exact
photo aliases are joined globally before SHA256 seed ranking within source families.
`split_group_id` records that connected component. Distinct photographs of one
patient keep separate image resources while sharing a partition. Known heldout
aliases protect the whole component: test takes precedence over validation, and a
train-only source aliasing heldout data is rejected. See
[PHOTO_SPLIT_GROUPING.md](PHOTO_SPLIT_GROUPING.md) for identity details and checks.
Existing REFUGE validation remains protected from training. Preparation requires a **new or empty output directory** and refuses to modify
any nonempty destination. Use a new path for a new dataset list, resolution or seed.
This prevents a failed conversion from corrupting files referenced by a previous
experiment manifest. A failed initial preparation can leave partial files; inspect
or remove that failed output before retrying.

`preparation_max_size` records the configured cap per image, including photographs
that were already smaller. It allows inference to apply the same scale policy.
`preparation.json` records dataset selectors, seed, cap, split policy and archive
identities: absolute path, size, modification time and ZIP member names, sizes and
CRC32 values. These support provenance checks but are not cryptographic hashes.

## Completed legacy adapter coverage

Every historical dataset loader now has a Python selector: 35 MATLAB `load_*` paths
and 14 Python loader paths across the archived branches map to these 31 datasets.
The packaged `adapters/coverage.json` inventory and coverage regression test check
this mapping. Generic ZIP/TAR/directory reading replaces transport/extraction helpers;
large-corpus autoencoder pretraining remains intentionally excluded as requested.

| Additional selectors | Resource type and meaning |
| --- | --- |
| `aria`, `avrdb`, `g1020`, `onhsd`, `riga`, `rimone`, `stare`, `uoadr` | Dense or explicitly sparse vessel/disc annotations; individual annotators preserved |
| `erlangen` | Corrected vessel masks when present, other images retained without invented masks |
| `diaretdb0`, `diaretdb1`, `eophtha`, `heimed`, `rcrgbma`, `roc` | Original images with lesion resources/marks retained as auxiliary annotations |
| `fire` | Registration pairs and control points; not vessel segmentation ground truth |
| `idrid` | Legacy localization images and disc/fovea center coordinates; center points are not contours |
| `inspireavr`, `vicavr`, `review` | Observer AVR, sparse class/caliber points or vessel-width edges; no fabricated dense labels |
| `messidor` | Images and XLS disease grades; nested archive exports are not double-counted |
| `ridb` | Recognition identity groups; repeated photographs stay in one split |
| `ubmi` | External private inference layout and existing prediction resources; not verified manual training labels |

Detailed semantics, actual counts and regression evidence are in
[ADAPTERS_SEGMENTATION.md](ADAPTERS_SEGMENTATION.md),
[ADAPTERS_AUXILIARY.md](ADAPTERS_AUXILIARY.md) and
[ADAPTER_MIGRATION_AUDIT.md](ADAPTER_MIGRATION_AUDIT.md).
UBMI's complete private data are not available here: its adapter is implemented and
synthetically tested, with an explicit missing-source error and `--source` override.
This is a data-availability limitation, not an unimplemented loader.

The original core adapters now also retain DRIVE's second observer on 20 test images,
CHASEDB1's second observer on all 28 images and all 30 IOSTAR disc masks. These
annotations share their original source groups across tasks/observers; they cannot
create independent train/test samples. Counts of annotation records can therefore
exceed counts of source photographs. The original model-pilot manifests are preserved
unchanged in `reports/2026-11-10`; reproducing those exact scores requires that snapshot,
not a newly expanded multi-observer manifest.

### Annotation validity and portability

New rows include `usable_for_training` and per-output `supervised_channels` flags.
Incomplete/invalid original annotations are retained with quality metadata and raw
resources; all-invalid rows are excluded by the training loader. A missing cup annotation
has a false cup-validity channel, never a background target. Checkpoints record channels
that received supervision; inference masks for unsupported channels are zero placeholders
and explicitly labeled as unsupported in JSON/NPZ output. Their raw probabilities are
untrained and must not be interpreted.

Image-only records use `task="images"`, omit `mask`, and contain a prepared FOV path.
Losses do not consume them. Their metadata retains original coordinates and annotation
resources: small tables and auxiliary masks are copied under `annotations/`; dense raster
references identify their decoded target/valid arrays instead of duplicating every full-size
marked photograph. `preserve=true` keeps auxiliary rasters such as UoA optic-disc-center
markers even when the same row also has a dense disc target.

**UoA-DR label correction:** the bundled dataset paper describes `.3.jpg` as the optic
**disc center**, not the cup. Both historical branches mislabeled it. The migrated adapter
preserves center markers/coordinates and leaves cup validity false. See the supervised
adapter notes for the primary-source evidence and the image 72 annotation swap.

AVRDB draws sparse vessel/class lines on white. Validity is limited to supplied reference
lines; unmarked pixels cannot become negatives. Precision on positive-only vessel support
cannot measure background specificity. Report per-dataset counts/scores and do not compare
a pooled complete-plus-sparse Dice with complete-FOV benchmarks. RIGA includes MESSIDOR
source photographs; source names do not resolve every possible cross-archive duplicate.
Do not report the two as independent cohorts without a verified cross-source identity map.

`--limit N` prepares a preview of at most N source groups per requested dataset, retaining
all included annotators/eyes/tasks. Both rows and preparation provenance identify previews.
Use a fresh directory without `--limit` for complete preparation. Root locations can be
overridden explicitly, for example `--source ubmi=/path/to/private/data`.

## Executed adapter checks (2026-11-10)

A real conversion at longest side 256 decoded and wrote **269 records** from
DRIVE, RITE, Drishti, CHASEDB1 and IOSTAR. Counts matched the table above,
including paired DRIVE/RITE split IDs and two task records for each IOSTAR image.
A separate conversion at longest side 128 decoded **146 records** from HRF and
DRHAGIS: HRF 45 vessel + 45 AV records; DRHAGIS 40 vessel + 16 AV records.
A full REFUGE conversion at longest side 128 decoded **1200 records** with the
official 400/400/400 split (323 seconds on this workstation). Across all **1615**
prepared records, array shapes/dtypes/ranges, cup containment and shared-source
split separation passed integrity checks. No valid target was entirely empty;
REFUGE retained a nonempty cup even at this deliberately small smoke resolution.
These are adapter/integrity checks, not model accuracy measurements.

The data regression tests cover A/V uncertain-pixel validity, cup containment in
REFUGE disc, deterministic shared-source splits, duplicate ZIP prefix elimination
with unchanged native coordinates, and refusing to overwrite prior experiment data.
