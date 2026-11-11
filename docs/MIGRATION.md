# Migration decisions and validation protocol

This branch consolidates three required tasks: binary vessel segmentation,
vessel segmentation with artery/vein classification, and optic disc/cup
segmentation. Python and `uv` are the supported implementation and installation
path. The unavailable large pretraining dataset is not a dependency.

The historical map is in [REPOSITORY_MAP.md](REPOSITORY_MAP.md). The executable
commands and currently supported adapters are documented in the main README and
[DATASETS.md](DATASETS.md); those are authoritative for the implemented scope.
This document separates reasons for changing the code from evidence that a model
has improved.

## Current model families

The original migration also introduced single-dataset pilots. Their architecture
and normalization below describe those pilots, not every model now in the package.

| Family | Input and purpose | Authoritative guide |
| --- | --- | --- |
| Original trained models | Original architecture and model-specific preprocessing; A/V also consumes a vessel mask | [Original models](LEGACY_MODELS.md), [preprocessing](LEGACY_PREPROCESSING.md), [native retraining](NATIVE_LEGACY_RETRAINING.md) |
| Single-dataset pilots | RGB normalized to `[-1,1]`; task-specific scratch experiments | [Pilot experiments](EXPERIMENTS.md) |
| Unified corpus models | Five outputs, partial supervision, `RGB/255 - 0.5`; current V2/TPE models use raw25 geometry | [Unified workflow](UNIFIED.md), [completed TPE preview](UNIFIED_V3_TPE_PREVIEW.md) |

Use the selected checkpoint's configuration and documented preparation workflow.
Do not transfer a pilot's normalization or resize policy to an original or
unified model. The [current status](STATUS.md) distinguishes delivered previews
from the unfinished optimization and final TEST evaluation.

## What to keep from the original work

Keep source images, annotation provenance, official benchmark splits, distinction
between binary vessel segmentation and A/V labels, original image geometry, and
independent disc/cup outputs. Weighted overlapping patch inference is useful for
high-resolution images. Store every decision needed to reproduce an experiment.

The old branches remain in git. Copying all old experimental scripts into the
new package would also copy absolute paths, incompatible HDF5 layouts,
interactive plots, obsolete environments and saved-object dependencies. Removing
those files from the working tree is appropriate once their replacements and
limitations are documented. Use `git show <ref>:<path>` to inspect old behavior;
do not delete historical branches, rewrite history or delete external data.

## Replacement boundaries

| Historical component | Implemented Python behavior |
| --- | --- |
| MATLAB `MAIN.m`, `load_*.m` | Explicit dataset adapters with checked filename pairing and label semantics; selective ZIP reads avoid extracting unrelated data |
| `souradnice.m`, `image_adjustment.m` | Explicit FOV policy and shared image/mask transforms; nearest interpolation for categorical labels; record resized geometry |
| `imwrite_single*.m`, MATLAB HDF5 transpose convention | Portable image files and NumPy mask arrays with documented channel/validity axes; manifest records source, task and split |
| Pretraining resaving and `dataset_pretrain.hdf5` access | Omit from supervised training; start weights randomly unless a future experiment explicitly selects another initialization |
| `Config.py` / `config.py`, editable author paths | Explicit CLI arguments and checkpoint metadata; portable relative manifest paths |
| Unseeded/random validation crops | Fixed image partition, complete deterministic validation tiling and all available labels |
| Scalar A/V encoding | Vessel, artery and vein outputs with annotation validity; unknown class is not an artery/vein negative |
| Four-channel legacy A/V classifier | New RGB model can learn vessel segmentation plus A/V together; old checkpoints require a separate compatibility path |
| Whole-object `torch.save(model)` | Model state dictionary and primitive configuration; no import-time dependency on historical model/config classes |
| Exploratory inference scripts | One prediction implementation shared by evaluation and CLI, stable shapes, probability files and readable overlays |
| PBS/Conda/batch files | `pyproject.toml`, `uv.lock` and documented CPU/GPU invocation |

The initial pilot path is a functional rewrite, not bitwise reproduction of MATLAB.
Its RGB U-Net starts from random weights, uses geometric channel growth, GroupNorm/SiLU
blocks and bilinear decoder upsampling. The old custom network used linear
channel growth, BatchNorm/ReLU and transposed convolution; master also supported
EfficientNet-B2. The historical A/V network consumed RGB plus a supplied vessel
mask. Those saved weights cannot be loaded directly into the pilot architecture;
the package now preserves their original architectures through a separate legacy
path, with numerical checks described in [LEGACY_MODELS.md](LEGACY_MODELS.md).

Pilot RGB normalization is `[-1,1]`, with image dimensions capped during preparation.
This differs from camera-angle resampling, MATLAB FOV heuristics, historical
contrast normalization and old `[-0.5,0.5]` inputs. Manifests record
`preparation_max_size`; training checks one matching policy across train and
validation and stores `inference_max_size` in run metadata and checkpoints.
Prediction applies that cap with Lanczos and restores probabilities bilinearly
to the original image frame before thresholding and hierarchy constraints.
A custom manifest without this field assumes its images use original scale;
its author must supply the field when a resize cap was applied.

The prepared output directory must be new or empty. Preparation rejects an
existing nonempty output before writing, so a failed rerun cannot overwrite
files behind an existing manifest. An initial failed preparation may leave
partial files; use a fresh directory for retry. Manifest reads also reject
source identities assigned to multiple splits across tasks. These checks
resolve the integration findings recorded in [REVIEW.md](REVIEW.md).

All historical dataset adapters now have Python implementations, including contour
recovery, stereo RIM-ONE and sparse/auxiliary collections. The coverage inventory
maps every historical loader path and records actual validation scope. UBMI requires
external private data for a complete real-source test; its external-data interface is
implemented and tested with representative synthetic layouts. Generic pretraining and
video pipelines remain outside the requested three segmentation tasks.

## Correctness checks

1. Inspect archive members and several actual masks before choosing thresholds.
   Validate image/mask dimensions, allowed labels, finite arrays and channel order.
   Check cup containment in disc; count invalid or empty targets explicitly.
2. Assign all annotations/tasks for one original image to the same partition.
   Keep DRIVE/RITE identities together. Preserve official test sets; derive
   validation only from training when none exists. Sort records before seeded
   random operations. Record manifest and seed in each experiment.
3. Test small and odd image sizes, images smaller than the tile, overlap coverage,
   and output restoration. Every pixel must receive finite predictions.
4. Test ignored A/V pixels: changing predictions there must not change loss or
   A/V score. Unlabeled tasks must not generate negative supervision.
5. Run each task on real images through preparation, optimization, checkpoint
   reload, evaluation and exported predictions. Verify masks and overlays by
   inspection; a successful synthetic unit test alone is insufficient.
6. Record exact commands, software versions, hardware, source git revision,
   manifest hash, task/channel names and run artifacts in repository notes.

Checkpoint selection uses validation only. A final test evaluation must load the
selected best checkpoint. Report class support; an empty or unlabeled target
must not silently produce a perfect score. Report per-image and per-source
results in addition to pooled scores when comparing heterogeneous datasets.

## How to establish an actual improvement

A repair that eliminates an exception or incorrect split is a verified
engineering improvement once its regression check passes. A loss decrease on a
few training batches is an optimization smoke check. Neither demonstrates better
held-out segmentation.

To compare architecture, loss, augmentation, resolution, contrast normalization
or postprocessing, fix source images, partitions, train budget, initialization
seeds, checkpoint selection, threshold and evaluation implementation. Change one
factor at a time. Save baseline and candidate runs with their manifests and
commands. Choose hyperparameters and thresholds on validation data; evaluate the
held-out test set after selection. Repeated seeds and paired image-level
confidence intervals help establish whether a small apparent gain is robust.

For vessels report Dice, IoU, precision and recall within the annotated FOV.
For A/V the current artery/vein Dice, IoU, precision and recall are restricted
to confidently labeled reference vessel pixels. They exclude background false
positives and must be interpreted as class-mask scores on that support, rather
than full-FOV artery/vein segmentation scores. Unknown/crossing pixels are
ignored. `av_accuracy` requires correct final thresholded, vessel-gated classes;
`av_conditional_accuracy` compares artery/vein probabilities on labeled reference
vessels without gating or thresholding. Retain vessel Dice/recall and support
counts so conditional classification cannot hide missed vessel detections.
The JSON output records these definitions with `metric_support`, accuracy
descriptions and `by_dataset` aggregates under the same validity masks. For disc/cup report each class separately; their sizes differ greatly.
If binary exports enforce cup-within-disc or A/V-within-vessels, report that
policy and apply the same policy to baseline and candidate. Preserve raw
probabilities so other policies can be compared without rerunning the network.

Legacy text scores alone cannot establish a new model's improvement: source lists,
preprocessed files, original architecture and validation logic differ, and the
historical code contains the defects recorded in `REPOSITORY_MAP.md`. Use actual
matched evaluation reports and the documented legacy numerical checks rather
than substituting historical text scores for a new measurement.
The controlled single-seed pilot recorded in EXPERIMENTS.md compares losses only.
It does not establish superiority over historical architectures or external cohorts.
Subsequent corpus-wide measured comparisons are linked from [STATUS.md](STATUS.md)
and the individual model cards; their explicit support and limitations govern
those results.

## Historical pilot follow-up priorities

These priorities record the initial pilot stage. Current corpus-wide training,
patch/photometry studies and HPO progress are tracked in [STATUS.md](STATUS.md)
and on the separate `experiments/leipzig-hpo` branch.

- Extend the completed 30-epoch scratch pilots with additional seeds, datasets and
  budgets; inspect validation predictions and the recorded unstable disc/cup curves.
- Compare field-of-view/scale preprocessing and illumination normalization under
  fixed partitions; vessel thickness is sensitive to resizing.
- Repeat the measured BCE/BCE+Dice comparison across seeds. The pilot supports
  BCE+Dice for vessels/disc_cup but BCE for AV. Compare foreground-focused sampling
  and context for disc/cup using validation; bright-boundary false positives remain.
- Compare postprocessing against raw predictions rather than assuming
  morphological cleanup improves every source dataset.
- Add further adapters only with real-data label and geometry checks; treat
  annotation quality and cross-dataset leakage as first-class concerns.
