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

# Explicit repair/provenance step for preparations made before the patient audit:
uv run python scripts/prepare_unified_data.py \
  --data-root /var/home/tom/ownCloud/fundus_databases \
  --output data/unified25 --regroup-patients --train-only-datasets riga uoadr

uv run python scripts/prepare_unified_data.py \
  --output data/unified25 --audit-existing
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
Raw unified inference on these additional sources requires an explicit
`--field-of-view-degrees 20`. RIM-ONE additionally requires `--fov` pointing to an
original-frame mask that selects the left half; automatic detection of both
stereo views would produce a different crop. Erlangen uses an explicit full-frame
FOV in preparation. Original official FOV masks should also be supplied for the
datasets that provide them.

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
G1020 uses the explicit `G1020.csv` crosswalk: 1,020 images from 432 patient IDs.
Repeated photographs of the same eye stay with that patient. ARIA conservatively
unions its cohort/number base across parenthesized transport counters and view
tokens; this is a grouping rule rather than independent patient proof.

Official test partitions remain test. Official validation remains validation;
official training derives a deterministic 20% validation subset unless that
dataset already has official validation. Sources without official partitions
use deterministic grouped 60/20/20 train/validation/test partitions, subject to
the training-only source exceptions below. Conflicting
official aliases choose the most restrictive partition, test then validation
then training, and are reported. A source group and an exact photograph can
never cross partitions. All annotation variants follow their photograph.

Patient separation is limited by the source metadata. The bundled UoA paper
explicitly says the images were de-identified and no patient information is
available; its spreadsheets contain diagnosis and landmark coordinates, not a
patient crosswalk. UoA image numbers therefore identify source photographs,
without proving that different numbers belong to different patients. The split
does not guess missing patient identities or certify patient independence where
those identities are unavailable. Consequently **all RIGA and UoA-DR records are
training-only**. RIGA's source thesis documents paired right/left eyes, but no
defensible crosswalk was recovered from its seven folders. These datasets have
no new validation/test partition and contribute no independent held-out metrics.
An alias of known official held-out data causes a failure instead of being forced
into training.

`--regroup-patients` explicitly repairs earlier output metadata, then reruns the
group split. It does not regenerate images or masks. It saves the original
metadata, original source/image identities, CSV SHA256, before/after manifest and
split hashes, per-source counts and moved photographs in
`patient-regrouping.json` and `patient-regrouping-history/`. Repeating the same
repair is idempotent. Auditing alone does not mutate patient identities or splits;
it rejects old G1020/ARIA identities and missing training-only policies.

These partitions prevent leakage of known groups and exact photographs. They do
not prove independence from
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
all 146 variants. This is a functionality result only. The completed full Ada
preparation and its independent integrity pass are recorded below.

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

Fingerprint schema 2 hashes the manifest, photograph index and geometry, every
referenced RGB/NPZ file, canonical photograph RGB and standalone FOV images.
Hardlink identity is only a read cache: file paths and bytes determine the hash.
The all-17-source preview was checked locally and after copying to Ada without
preserving hardlinks. Its earlier fingerprint was superseded by the explicit
patient regrouping; use the current output's `integrity.json`.

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

## Completed full-source preparation

The full Ada pass completed on 2026-11-10. It reads all 17 selected sources,
without a preview limit, and retains **7,365 usable annotation variants on 3,820
distinct decoded photographs**. The explicit patient repair completed before
the final audit: G1020 retains all 432 patient IDs across 1,019 usable labelled
images; ARIA counters are normalized; RIGA and UoA are training-only. The repair
moved 892 photographs relative to the earlier image-group partitions, without
regenerating RGB, FOV or NPZ arrays. Original metadata and the CSV SHA256 remain
in the patient-regrouping report and metadata backups.

| Partition | Distinct photographs | Annotation variants |
|---|---:|---:|
| Training | 2,221 | 5,281 |
| Validation | 770 | 993 |
| Test | 829 | 1,091 |

| Source | Photographs | Annotation variants | Training / validation / test photographs |
|---|---:|---:|---|
| ARIA | 143 | 406 | 87 / 28 / 28 |
| AVRDB | 99 | 200 | 59 / 20 / 20 |
| CHASEDB1 | 28 | 56 | 16 / 6 / 6 |
| DRHAGIS | 39 | 56 | 23 / 8 / 8 |
| Drishti-GS | 101 | 101 | 40 / 10 / 51 |
| DRIVE | 40 | 60 | 16 / 4 / 20 |
| Erlangen | 106 | 106 | 63 / 22 / 21 |
| G1020 | 1,019 | 1,019 | 609 / 204 / 206 |
| HRF | 45 | 90 | 27 / 9 / 9 |
| IOSTAR | 30 | 90 | 18 / 6 / 6 |
| ONHSD | 88 | 343 | 53 / 17 / 18 |
| REFUGE | 1,200 | 1,200 | 400 / 400 / 400 |
| RIGA | 510 | 2,681 | 510 / 0 / 0 |
| RIM-ONE | 159 | 477 | 95 / 32 / 32 |
| RITE | 40 | 40 | 16 / 4 / 20 |
| STARE | 20 | 40 | 12 / 4 / 4 |
| UoA-DR | 193 | 400 | 193 / 0 / 0 |

Exact photograph aliases are counted once globally, including DRIVE/RITE and
duplicates within some sources. Annotation variants retain their separate masks.
The per-source photograph column therefore must not be summed to obtain the
global photograph total.

Valid output-channel support, counted in annotation variants:

| Output | Training | Validation | Test |
|---|---:|---:|---:|
| Vessel | 768 | 187 | 239 |
| Artery | 130 | 42 | 59 |
| Vein | 130 | 42 | 59 |
| Disc | 4,513 | 806 | 851 |
| Cup | 3,867 | 676 | 709 |

These channel counts include observers and parallel vessel/A-V labels; they are
not independent photograph counts. Per-source distinct supported photographs,
positive photographs and valid/positive pixel totals are in the
[complete integrity report](../reports/data/unified25/integrity.json).

Excluded records comprise 1,813 unrecoverable RIGA outlines, 53 invalid ONHSD
radial annotations, one ambiguous G1020 JSON, 377 unlabelled STARE photographs and
40 unlabelled Erlangen photographs. The first 1,867 are quarantined dense labels;
the remaining 417 provide no requested dense supervision.

The final audit checked every retained NPZ and passed all binary-label, geometry,
conditional-class, validity, photograph/patient grouping, official holdout and
source-policy checks. Its frozen identifiers are:

* Manifest SHA256: `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`
* Prepared-content SHA256, schema 2: `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`
* Finalizer/auditor source commit: `87cf045ab89425c4994df577fd510d210a1a0eab`

Prepared data live on Ada at
`~/code/DeepRetinaSegmentation_for_Leipzig/data/unified25`. The repository contains
the [compact summary](../reports/data/unified25/summary.json),
[composition](../reports/data/unified25/composition.json),
[patient repair](../reports/data/unified25/patient-regrouping.json) and
[compact source provenance](../reports/data/unified25/source-provenance.json).
The complete archive-member metadata remain in the prepared output; the compact
report records their SHA256 and per-archive member counts.
This establishes a completed, checked training input. Model-quality conclusions
require the separate training and evaluation experiments.
