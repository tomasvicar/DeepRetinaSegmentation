# Vascular4 TRAIN annotation coverage — read-only diagnosis

Actual metadata at **2026-11-11 10:18:28 UTC** already shows epoch **53**, not the
earlier epoch 50 snapshot described in the request. All **5,700 / 5,700** eligible
TRAIN annotation records have a strictly positive contribution counter. Missing
IDs: **none**. All **2,529** photos were seen; minimum photo visits is **55** and
actual delivered samples are **434,176 = 53 × 8,192**, with 108,544 optimizer steps.
This is an in-progress observation, not full60 completion or a prediction that a
future run will meet its gate. The strict terminal all-record coverage check is
unchanged.

The original frozen 0b and 72 source bytes were inspected without importing
PyTorch. Their photo-sampler iterator, observer `sample`, crop/augmentation/trace
`__getitem__`, and `sampling_weights` method ASTs are identical. Actual Git and
file/method hashes are in `source-pins.json`. Manifest/run/coverage/CSV hashes and
the checked same-generation metadata are in `observed-state.json`. Only TRAIN
row metadata and contribution counters were inspected: no GT arrays, images,
weights, GPU calls, sampling simulation or live-source/run changes.

## Why photo coverage is not annotation coverage

The coverage-first sampler randomly permutes every photo once, then draws
weighted replacement extras to reach 8,192. Its unit is a decoded photograph,
not a record or annotator. For a sampled photo, `sample` chooses one available
vascular task bundle (A/V or binary vessel) and one annotator within that task,
plus one disc/cup annotator pair. Annotators are drawn independently with
replacement; the code keeps no per-photo observer-cycle counter. A repeated
photo in different source aliases can therefore have 12 disc/cup records rather
than the usual six.

The training loop updates `annotation_visits` from active trace IDs **after a
successful optimizer update**. An active record has retained validity support
for its task in the delivered crop; a positive visit counter does not itself
mean that every foreground pixel or the entire mask was seen. Global fallback
ensures some supervision, not coverage for every selected task/variant.
Consequently photo coverage and weighted task mass cannot imply that every
observer has already contributed.

Five records currently have only **one** contribution. All are RIGA disc/cup
records in 12-option photo groups:

| Record | Actual photo visits | Own contributions |
|---|---:|---:|
| `riga_BinRushed1-Corrected_image28_observer6_disc_cup` | 64 | 1 |
| `riga_BinRushed2_image44_observer4_disc_cup` | 71 | 1 |
| `riga_BinRushed2_image7_observer3_disc_cup` | 77 | 1 |
| `riga_BinRushed3_image10_observer3_disc_cup` | 72 | 1 |
| `riga_MagrabiFemale_image22_observer6_disc_cup` | 63 | 1 |

For each of these photos, the sum of all 12 annotation counters equals its photo
visit count: one actual disc bundle contributed on every recorded photo visit.
This observed metadata is consistent with random replacement among aliases and
observers, rather than a permanently unusable bundle or a photo-sampling failure.
It does not retrospectively identify the sole missing record at epoch 50: the
current cumulative file has replaced the old generation and contains no first-
contribution epoch. Without a preserved epoch 50 metadata snapshot, naming one of
these five as *the* missing historical record would be an unsupported inference.

For intuition only, an ideal independent uniform draw among 12 options gives
`P(a specified observer never selected in n visits) = (11/12)^n`. For the observed
63–77 visits this is approximately 0.42%–0.12% per specified option. These are
conditional analytical selection probabilities, **not measured missed-crop
probabilities**, an explanation proven for one historic record, or a guarantee.
Many photos/options make an occasional late observer unsurprising. Actual
contribution is additionally conditioned on task choice and crop validity.

## Future experiment, not a live correction

Keep the existing photo indices, coverage-first pass, source/task weights and
vascular-task choice unchanged. Within each selected `(photo_id, task)` group,
use a main-process occurrence/counter schedule that cycles all observer bundles
before repetition. A domain-separated private RNG can shuffle cycles without
adding draws to the existing augmentation generator. Store cycle state and
successful contribution accounting in checkpoints; worker-local counters alone
would duplicate/skip entries across workers and resumes. This is a new recipe,
source hash and seed lineage, never a retag of current random receipts.

A cycle can guarantee **selection** within the number of available task
opportunities, while exact active annotation contribution additionally needs
valid task support after cropping. If an owed variant misses support, a future
explicit task-aware fallback/retry policy is needed; silently counting a selected
but unsupported record is invalid. Such a policy changes crop exposure and must
be measured as a separate quality experiment. It need not change the photo/source/
task mass, but cannot promise unchanged global crop RNG or bitwise output when
annotation choice changes foreground-dependent branches. The independent existing
augmentation RNG can avoid extra draws; this alone is not whole-pipeline parity.

No current sampler, proposal contract, budget or strict gate is changed. If a
terminal run still lacks even one eligible record, it fails full-annotation
completion; finite metrics and photo coverage do not waive that requirement.
