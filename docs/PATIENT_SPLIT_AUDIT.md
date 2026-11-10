# Patient split audit before full unified training

Audit date:2026-11-10. This records defects in the **pre-correction** partitioning,
not a certification of the eventual training manifest. Preparation, policy and
post-correction integrity results belong in [UNIFIED_DATA.md](UNIFIED_DATA.md).
The compact evidence inventory is
[patient-audit-before.json](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/d83f17a83e873f840751493b1e0457b4c386a07d/reports/data/patient-audit-before.json).
No raw clinical records, full patient crosswalk or photographs are copied here.

## G1020: recoverable patient mapping, confirmed partition leakage

The supplied `G1020.zip` contains `G1020/G1020.csv`, with `patientID`, `imageID`
and `Left/Right`. All1020 images have a patient ID:432 unique patients,
with up to12 images each. CSV SHA256:
`faf663e197880b38ed085f7fb361b3acbc8f1592693aa7163de26e114402ac5f`
(347206bytes). This identifier mapping is authoritative supplied metadata.

The counts below use all1020CSVimages before annotation-validity filtering; they are not counts from the final retained manifest. The original adapter grouped by `g1020:image_<number>`. Replaying the original
`safe_splits(seed=42)` over the actual CSV image IDs gives612train/204val/204test:
228patients span multiple partitions,119occur in both train and validation,
and111in both train and test. These overlap counts are not additive. A concrete
same-eye example is `image_0.jpg` in train and `image_1.jpg` in test: both map to
the same CSV patient and right eye. Exact-image deduplication cannot replace
patient grouping for different views or acquisitions.

The dataset creators independently report1020images from432patients,1–12images
per patient, and patient-level cross-validation to avoid this problem.
See [Bajwa et al., G1020, §III and §IV-B, PDFpp3–4](https://www-live.dfki.de/fileadmin/user_upload/import/10840_G1020_A_Benchmark_Retinal_Fundus_Image_Dataset_for_Computer-Aided_Glaucoma_Detection.pdf).

Approved correction: use `g1020:patient:<CSVpatientID>` as the split group,
retain image ID/laterality and crosswalk resource/hash in metadata, then rerun
all patient/photo unions and split assignment. Existing RGB/target arrays need
no regeneration. A new manifest hash and integrity audit are required before
training. The portable audit should check supplied patient IDs independently of
`source_id`, so a mistaken source grouping cannot silently pass again.

## ARIA: filename counters break existing shared-base grouping

ARIA normally uses the `aria_<cohort>_<base>` prefix as its source group. Eight
raw filenames have leading parenthesized counters. The existing
`stem.split('_')[:3]` keeps those counters, incorrectly separating these shared
base identities. The following are actual pre-correction seed42 results:

| Shared base | First file | Split | Second file | Split |
|---|---|---|---|---|
| `aria_d_26` | `(0001)aria_d_26.tif` | test | `(0018)aria_d_26.tif` | train |
| `aria_d_27` | `(0001)aria_d_27.tif` | test | `(0026)aria_d_27.tif` | train |
| `aria_d_28` | `(0004)aria_d_28.tif` | val | `(0007)aria_d_28.tif` | train |
| `aria_d_29` | `(0003)aria_d_29.tif` | train | `aria_d_29.tif` | val |

All eight decoded RGB hashes differ, so exact-photo union does not repair these
splits. This is a concrete inconsistency with the adapter's own existing base
identity rule. It is **not** independent clinical proof of patient identity.
Conservatively normalize a regex match `aria_([acd])_(\d+)`, ignoring leading
counters and trailing view tokens, before grouping. Do not label the ARIA
filename convention a verified clinical crosswalk without further evidence.

## RIGA: paired eyes, no recoverable patient crosswalk

The creator's primary thesis explicitly describes750fundus photographs as
including paired right/left eyes: MESSIDOR460, BinRushed195 and Magrabi95.
See [Almazroa,2016, §3.2, printedp54/PDFp71](https://uwspace.uwaterloo.ca/bitstreams/29ac8bdf-6c5a-4b1f-ae6e-95d8e60dd510/download).

The available tree has5550image resources and no README, patient table or other
sidecar. All seven constituent folders use `image<number>prime` names; numbers
reset by folder. They do not contain patient or L/R suffixes. Sample JPEG headers
have no EXIF identity; inspected TIFF identity tags are empty apart from editing
timestamps. Neither odd/even neighboring image numbers nor the four BinRushed
folders provide an evidenced patient crosswalk. Magrabi male/female folders are
sex groups, not patient identifiers.

The [official MESSIDOR description](https://www.adcis.net/en/third-party/messidor/)
provides per-image diagnoses, states identifying information was discarded, and
lists13duplicate image pairs in its errata. These facts do not recover a patient
mapping for RIGA's renamed MESSIDOR subset.

Approved policy: retain all suitable RIGA annotations as **training-only across
all seven folders**; generate no new RIGA validation/test quality metrics until
a verified patient mapping exists. Preserve exact-photo/known-reuse union and
never move an official heldout alias into training to satisfy this policy.

## UoA-DR: conservative training-only policy

A patient crosswalk was not recovered from the supplied spreadsheets/papers.
The data owner has separately audited those resources. There is no affirmative
one-image-per-patient proof available to this audit. Approved policy is therefore
whole-source training-only, with no new internal validation/test quality claims.
This preserves suitable training supervision without inventing patient IDs.

## All17sources: evidence and limits

| Source | Available identity/grouping evidence | Interpretation/action |
|---|---|---|
| DRIVE | Official train/test; image numbers | Preserve official protocol and all observer variants. |
| RITE | Reannotations of the same DRIVE photographs | Union under the corresponding `drive:<number>` across tasks/datasets. |
| HRF | Numeric base plus diagnosis suffix | Current shared numeric group is conservative across categories; not a recovered patient table. No concrete leakage counterexample found. |
| DRHAGIS | Numeric image IDs; creators identify duplicate24/32 | Exact decoded-photo union must protect that known reuse. No additional patient table recovered. |
| REFUGE | Official400/400/400 partitions; per-image spreadsheets | Preserve official partitions; call these official-protocol holdouts, without claiming this audit recovered a patient crosswalk. |
| Drishti-GS | Official train/test; image diagnosis spreadsheet | Preserve official partition; no independent patient crosswalk recovered. |
| CHASEDB1 | `Image_<subject>L/R` | Strip eye suffix; retain both eyes and observers together. |
| IOSTAR | Unique STARnumber in each of30files; OS/OD and view suffix | No repeated STARnumber or explicit patient table recovered. Image naming alone is not affirmative clinical patient proof. |
| ARIA | Repeated cohort/base prefixes and counter-prefixed exceptions | Apply conservative base normalization described above; convention is not a verified clinical crosswalk. |
| AVRDB | Opaque IMnumbers, one folder per photo | No explicit patient table or concrete repeated-patient counterexample recovered. Do not promote image IDs to proven patient IDs. |
| G1020 | Complete supplied CSVpatientID→imageID/laterality | Mandatory patient regroup, independently audited. |
| ONHSD | `D<identifier>-L/R` | Current final-dash removal groups paired eyes. Raw examples include `D01187-L/R`. |
| RIGA | Creator confirms paired eyes, raw names have no patient mapping | Whole-source training-only until verified mapping. |
| RIM-ONEr3 | G/Nbase IDs with eye suffix | Strip L/R and visit suffixes conservatively; preserve observer/stereo variants. |
| STARE | Opaque imnumbers; no patient table in supplied archive | Full-source patient identity remains unproven. The standard20vessel photographs must not be conflated with all available disc-labeled photos. |
| UoA-DR | No recovered crosswalk or affirmative uniqueness proof | Whole-source training-only. |
| Erlangen | Study+patient+visit+eye filenames |146raw photos=73Study_02patient prefixes, each exactly one L/R pair at one visit. Current Study+patient grouping retains those pairs. |

Unknown identity is distinguished from a demonstrated leak. No concrete
additional patient-leak counterexample was recovered for IOSTAR, AVRDB, full
STARE or HRF; that does not certify their image IDs as clinical patient IDs.
Any strict patient-disjoint claim must name its supporting identity evidence
per source rather than assert a universal guarantee from filename hashing.

DRHAGIS duplicate evidence is from its creators'
[2017dataset paper, Materials](https://pmc.ncbi.nlm.nih.gov/articles/PMC5299858/).
REFUGE's original [challenge report, §3.1](https://arxiv.org/html/1910.03667)
describes the fixed official partitions and anonymization, but does not provide
an independently recovered patient linkage here.

## G1020 versus Erlangen: no evidenced shared cohort

G1020's primary paper locates acquisition in a private ophthalmology practice
in Kaiserslautern during2005–2017. The separate Erlangen archive contains image,
average and mask directories without a cohort README or crosscollection linkage.
No shared-cohort or patient mapping evidence between these two collections was
found. Their German origin does not justify assuming shared patients. Retain
known per-source grouping plus evidenced exact-photo reuse; do not invent an
additional crosscollection exclusion from geographical coincidence.
