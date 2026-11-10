# Data preparation and annotation contract

`uv run deepretina prepare --help` documents the command. The Python entry point is
`deepretina.prepare.prepare(data_root, output, datasets, max_size=1024, seed=42)`.
Data remain outside Git. ZIP members are decoded directly: no MATLAB, HDF5
conversion, large intermediate extraction, pretraining database or network access
is required. Names of datasets passed to Python are lowercase as below.

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
| `iostar` | IOSTAR.zip | vessels and AV | 30 | Deterministic 60/20/20 image groups |

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

Splits are determined by SHA256 of seed plus source group and ranked within source
family. Official test sets never become training or validation data. Existing
REFUGE validation stays intact. Preparation requires a **new or empty output directory** and refuses to modify
any nonempty destination. Use a new path for a new dataset list, resolution or seed.
This prevents a failed conversion from corrupting files referenced by a previous
experiment manifest. A failed initial preparation can leave partial files; inspect
or remove that failed output before retrying.

`preparation_max_size` records the configured cap per image, including photographs
that were already smaller. It allows inference to apply the same scale policy.
`preparation.json` records dataset selectors, seed, cap, split policy and archive
identities: absolute path, size, modification time and ZIP member names, sizes and
CRC32 values. These support provenance checks but are not cryptographic hashes.

## Legacy MATLAB adapter disposition

The historical branches remain available through Git. The supported Python adapters
above replace the training-relevant paths `load_drive`, `load_drive_VA`, `load_hrf`,
`load_hrf_VA`, `load_drhagis`, `load_drhagis_VA`, `load_drishtigs`, `load_refuge`,
`load_chasedb1`, `load_iostar`, and `load_iostar_VA`.
The new generic ZIP pairing, RGB decoding and manifest writer replace the runtime
need for `MAIN`, `MAIN_VA`, `image_adjustment`, `imwrite_single*`,
`folder_creation`, `subdir`, `average_size`, `souradnice`, and `hysthresh`.
These replacements are functionally targeted and do not claim pixel-for-pixel
parity with every legacy heuristic.

The following historical adapters are **not ported** and are not accepted by the
new `prepare` command: ARIA, AVRDB, DiaRetDB0, DiaRetDB1, E-Ophtha, Erlangen, FIRE,
G1020, HEI-MED, IDRiD, INSPIRE-AVR, MESSIDOR, ONHSD, RC-RGB-MA, REVIEW, RIDB,
RIGA, RIM-ONE, ROC, STARE, UoA-DR, VICAVR, and internal UBMI. Some supply only
unlabelled photographs, point measurements or lesion labels; AVRDB has inconsistent
filenames/line annotations, and RIGA requires recovering outlines from annotated
photographs. They require dedicated validation before use. The large image-only
pretraining path (`resave_pretrainig`) is intentionally outside this pipeline.
Legacy exploratory `tmp` code is not a supported data conversion interface.

See the repository experiment notes for executed preparation checks and model
results. Supported data loading does not itself establish segmentation accuracy.

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
