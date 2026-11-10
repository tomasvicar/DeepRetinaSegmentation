# Data for the unified five-output model

`scripts/prepare_unified_data.py` reads the original archives without extracting or
changing them. It writes RGB images, annotation variants and provenance for one
network with vessel, artery, vein, disc and cup outputs. Missing annotations are
unknown; they never become negative labels.

```bash
uv sync --frozen
uv run python scripts/prepare_unified_data.py \
  --data-root /var/home/tom/ownCloud/fundus_databases \
  --output data/unified25
```

Use a new output directory. Preparation refuses an existing nonempty directory.
An interrupted directory is incomplete until `_SUCCESS` exists and its manifest
SHA256 matches `manifest.json`. Source preparation runs on the CPU; no model is
trained by this command.

For a functionality check, add `--limit 3`. This limits source groups per dataset,
retains their annotation variants, and labels every result
`preparation_preview=true`. The unified trainer rejects previews when the run
purpose is `full_data`. A successful preview is not evidence of completed data
preparation or model accuracy.

## Geometry and intensity

The explicit profile is `legacy25_raw`: dataset field angle times 25 pixels per
degree, original field-of-view crop, linear antialiased RGB resampling and
nearest-neighbour labels. Scaling uses the original full frame before cropping,
as in the original single-network Python loader. The model receives
`uint8_RGB / 255 - 0.5`. There is no 1024-pixel cap, illumination correction or
CLAHE in this profile. The verified implementation and numeric source comparisons
are described in [LEGACY_PREPROCESSING.md](LEGACY_PREPROCESSING.md).

Official FOV masks are used for DRIVE, HRF, DRHAGIS and IOSTAR. RITE shares the
DRIVE photograph and official FOV; when prepared alone, its FOV is read from
`DRIVE.zip`. Other original datasets use the original dataset-specific automatic
FOV procedure. Additional annotation FOV restrictions affect label validity,
without silently replacing the original image crop. UoA timestamp suppression
matches the original preprocessing.

RIM-ONE and Erlangen use the 20-degree declaration from the MATLAB loaders.
RIM-ONE release 3 retains the full stereo frame for the scale calculation, crops
to the left annotated eye, and ignores the right eye. Its angular calibration is
explicitly experimental: it was not established by the original 14-dataset
Python pipeline. Do not present its physical scale as independently verified.

## Sources and annotation meaning

The default preparation includes all 17 registry selectors with suitable dense
supervision: DRIVE, RITE, HRF, DRHAGIS, REFUGE, Drishti-GS, CHASEDB1, IOSTAR,
ARIA, AVRDB, G1020, ONHSD, RIGA, RIM-ONE, STARE, UoA-DR and Erlangen.
The remaining registered sources provide image-only, lesion, width or point
annotations rather than the requested five dense outputs. They are not used as
unlabelled training negatives. No unavailable large pretraining database is used.

Annotation variants remain separate manifest rows and are indexed together in
`photos.json`. Model sampling must avoid treating annotator copies as independent
photographs. The unified dataset combines compatible task annotations on a
photograph and selects an annotation variant explicitly.

* A/V targets keep vessel, artery and vein as one unit. Unknown classes and
  crossings have false class validity.
* AVRDB line annotations supervise marked vessel positives only. Unmarked
  background is unknown.
* Disc-only datasets have an entirely false cup-validity channel.
* UoA `.3.jpg` marks the optic-disc center, not the cup. Its center remains
  metadata; it supplies no cup supervision. Image 72 uses the vessel/disc swap
  present in the MATLAB loader and missing from the original Python loader.
* Invalid ONHSD radii, ambiguous G1020 contours and unrecoverable RIGA contours
  are listed in `excluded.json`; they supply no invented dense labels.
* STARE photographs without vessel ground truth are excluded from supervised
  training.

Detailed decoder evidence and full adapter validation counts are in
[ADAPTERS_SEGMENTATION.md](ADAPTERS_SEGMENTATION.md). Those earlier migration
counts use their own output geometry and must not be substituted for this run's
`composition.json` counts.

## Identity, partitions and portability

`photo_id` hashes the original decoded RGB shape and pixels, before geometry or
timestamp changes. Exact aliases such as DRIVE/RITE share the first canonical
photograph frame. `source_id` retains patient/source grouping; it is not assumed
to identify a single photograph, because paired eyes may share a patient.
The split operation unions both constraints before assigning any partition.

Official test partitions remain test. Official validation remains validation;
official training derives a deterministic 20% validation subset unless that
dataset already has official validation. Sources without official partitions
use deterministic grouped 60/20/20 train/validation/test partitions. Conflicting
official aliases choose the most restrictive partition, test then validation
then training, and are reported. A source group and an exact photograph can
never cross partitions. All annotation variants follow their photograph.

These partitions prevent new-model leakage. They do not prove independence from
historical checkpoints: old image counters came from unsorted file enumeration,
and exact training-image mappings are unavailable for many legacy datasets.
Original-model scores must retain the benchmark helper's exposure qualification.

The output is portable: image, FOV and NPZ paths are relative. Hardlinked aliases
contain identical image bytes and remain usable if copied without preserving
hardlinks. `photos.json` records original source paths, original geometry, exact
RGB hashes and annotation variants. Raw auxiliary annotations are referenced in
metadata; the original 33 GB archive collection is not copied into this output.
`preparation.json` records source archive information and preparation parameters.
`composition.json` records actual retained counts and exclusions, not estimates.

The all-source `--limit 3` check retained 65 physical photographs and 146 dense
annotation variants; one unlabelled Erlangen record was excluded. Shape, binary
labels, profile, photograph identity and source-group partitions were checked on
all 146 variants. This is a functionality result only. Full Ada preparation is
written separately to `data/unified25`; inspect its completion marker and
composition before starting a full-data experiment.

After completion, check every retained mask and obtain counts per source,
partition and output channel:

```bash
uv run python scripts/prepare_unified_data.py \
  --output data/unified25 --audit-existing
```

This creates `integrity.json` with the exact manifest hash, checked annotation
and photograph counts, supported/positive photograph counts, and valid/positive
pixel totals. Pixel totals include annotation variants; photograph totals count
an eye once per source, partition and channel. The audit checks binary labels,
RGB/mask geometry, false validity outside FOV, paired conditional A/V labels,
cup containment, identical RGB aliases and patient/photograph partition integrity.

## Checking unexpectedly weak historical predictions

The original-network preview showed weak UoA vessels and IOSTAR discs despite
coherent source overlays. The annotation mappings were checked against the
original Python loaders, independently of model predictions. For the actual
preview IOSTAR `STAR 01_OSC` image, both original and current decoders yield
exactly the same 22,667-pixel disc. Across all 30 IOSTAR source masks, 29 are
identical; `STAR 38_ODC` differs by 87 pixels because the enclosed-background
connectivity differs (mask Dice 0.998085). This does not explain the weak preview
prediction on `STAR 01_OSC`.

UoA preview photograph 100 uses `.1.jpg` vessels and `.2.jpg` disc in both
loaders. The original threshold is red >128; the adapter uses red >127, changing
48 vessel pixels on that photograph and zero disc pixels. Photograph 100 is
unaffected by the genuine image-72 source swap. No evidence of a new
vessel/disc label swap was found. RIM-ONE and Erlangen were outside the original
14-dataset Python loaders. Low scores on these examples remain diagnostic model
results, rather than proof of a data-adapter error or of independent quality.
The numeric source evidence is saved in
`reports/legacy/iostar-uoa-source-annotations.json`.
