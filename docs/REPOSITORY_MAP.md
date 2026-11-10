# Repository map and research notes

Audit date: 2026-11-10. Source: `https://github.com/tomasvicar/DeepRetinaSegmentation`.
This document describes the historical code at the refs below, even when those files
have been removed from the cleaned `for_Leipzig` working tree. Retrieve a file with
`git show origin/single_net:Dataset.py`, for example. These are code observations,
not claims that the old experiments have been reproduced.

## Branch inventory

| Remote branch | Commit | Purpose and principal files |
| --- | --- | --- |
| `master` | `06ae8f5c027a47b1ecfac8d56b4c33fbe6664449` | 2021 vessel segmentation, optional denoising pretraining, MATLAB preprocessing; `config.py`, `dataset.py`, `split_data.py`, `train.py`, `test_fcn_vessels.py` |
| `artery_vein_class` | `7bb0539b90dfce2ed6eaeef1de812072be1155d0` | 2022 A/V classification: `Ves_Class/`, MATLAB `MAIN_VA.m`, nnUNet v1 interchange and postprocessing |
| `seg_videa` | `c6bbd8f9db0793591c066a9bb42fc19585505cc6` | 2023 vessel/video experiments, `evaluate_videos.py`, `FFFMPEGvideo_read_write.py`, MATLAB video conversion, contrast normalization |
| `segment_sada01` | `cb2f3ef3719b6131746dca15d97055b1fb37efcd` | 2023 vessel inference on own image collection; `evaluate_data.py`, `resave_files.py`, normalization/optimization scripts |
| `segment_sada02` | `8ed4937afc337a16d79fb1f577a2301c01a51a54` | Related own-image segmentation; same general structure, latest commit says segmented |
| `single_net` | `2566c501f52d238c9176c027f60017c57424486c` | 2023 Python dataset loaders; vessels/disc/cup multitask U-Net; nnUNet v2 preparation and evaluation; original-resolution restoration |

`for_Leipzig` was created from `master`; useful functionality must be reconciled
from the other branches rather than assuming master contains all tasks.
The original README contains only the project name. There is no original unified
installation or portable training/prediction CLI.

## Master: preprocessing, training, prediction

- `matlab_preprocessing/MAIN.m` calls database-specific `load_*.m` scripts from a
  hard-coded Windows shared-drive root. It currently selects 12 pixels/degree;
  25 pixels/degree is a commented alternative.
- `souradnice.m` estimates the field of view (FOV), centre and radius using
  dataset-dependent histogram thresholds, morphology and fallback heuristics.
  `image_adjustment.m` resizes by requested pixels/degree and camera FOV angle.
  Its apparent crop computations are unused; the crop code is commented out.
- `imwrite_single.m` thresholds masks and writes MATLAB HDF5 datasets. Its
  independent hard-coded output directory must match `MAIN.m`.
  Python transposes axes when reading this MATLAB layout.
- `config.py` defines class-level settings, detects a relative HDF5 directory at
  import time and raises `no data` if absent. Default input is green channel,
  patch 256, CLAHE, EfficientNet-B2 U-Net, batch 32, CUDA device 0.
- `dataset.py` reads `dataset.hdf5`; supervised masks become binary. Training
  samples random patches and flips/rotates them. Many advertised augmentation
  settings are commented out. Validation also samples random patches.
- `split_data.py` splits each source database and keeps explicit test entries;
  secondary annotations are attached after splitting. It **unconditionally** opens
  `dataset_pretrain.hdf5`, including for supervised training. This blocks use
  without the unavailable pretraining dataset. It also uses removed `np.int`.
- `train.py` supports scratch, ImageNet or saved-object initialization, AdamW,
  BCE/Dice and an LR schedule. It uses interactive plotting every epoch, saves
  entire model objects with `torch.save(model)`, chooses minimum validation loss,
  and recursively deletes the configured temporary model directory afterward.
- `unet.py` is a custom convolutional U-Net with linear channel growth, batch
  normalization and optional dropout. It pads to a power-of-two boundary, moves
  input to the model device internally, and returns original spatial dimensions.
- `main_vessels*.py` orchestrate separate/universal/retrained experiments;
  `main_drive_opt.py` and `read_opt.py` are experiment/tuning scripts.
  `get_results.py` reads a hard-coded result JSON and prints a table.
- `test_fcn_vessels.py` performs overlapping weighted-patch prediction, reads FOV
  masks and computes vessel metrics. It hard-codes CUDA and depends on model
  attributes and old HDF5 paths. Tiling assumes images are at least patch-sized.
- `utils/` contains logging, loss/Dice and GPU-memory helpers. Root `test_*.py`
  and `tmp_reader_test.py` are exploratory executables rather than an isolated
  automated regression suite. PBS/Singularity scripts encode old cluster setup.

## Artery/vein classification branch

`matlab_preprocessing/MAIN_VA.m` combines AVRDB, IOSTAR, DRIVE/RITE, HRF and
DRHAGIS annotations and prepares nnUNet v1 `Task535_Ophtalmo`. `load_ubmi_VA.m`
prepares own UBMI images for prediction without A/V reference masks.

`Ves_Class/Network.py` and `Network_v9.py` define classification networks;
`Loaders.py` handles preprocessed DICOM, `Main_Training.py` trains,
`Main_Predict.py` predicts and restores original dimensions, and
`Utilities.py` contains augmentation, class postprocessing and overlays.
`OptimBayess_Net.py`, `BO_example.py` and `Validation_predict.py` are experiments;
`postproc_nnUNet_predictions.py` handles nnUNet output; `enviSegm.yml` is the old
environment description. Nine model `.pt` files and two history `.pkl` files are
tracked on that branch.

Important distinction: the saved A/V classifier consumes **RGB plus an existing
binary vessel segmentation**, four input channels at 25 pixels/degree. It does
not independently segment vessels. It predicts background/artery/vein; inference
compares artery and vein outputs and removes pixels outside the supplied vessel
mask. Training uses cross entropy, Adam, crops 450 and geometric augmentation.
The checked-in training script fine-tunes v8 into v9 using absolute author paths.
Its train `data_list[1:b+1]` and validation `data_list[b:]` slices overlap at index
`b`, so reported validation results need independent verification.

## Single-network branch

- `datasets_loaders/main.py` runs 14 Python adapters: ARIA, AVRDB, CHASEDB1,
  DRHAGIS, Drishti-GS, DRIVE, G1020, HRF, IOSTAR, ONHSD, REFUGE, RIGA, STARE and
  UoA_DR. Thus substantial MATLAB work already has Python counterparts here.
- `LoaderGeneric.py` detects/crops FOV, rescales to 25 pixels/degree, optionally
  normalizes contrast, and writes one HDF5 group per image. Stored attributes
  include source filename, crop box, original size and scale factor. Re-running
  the top-level loader deletes the existing output HDF5 first.
- `utils/local_contrast_and_clahe.py` fills background by spatial remapping,
  divides by a Gaussian illumination estimate and applies CLAHE in HSV value.
  Gaussian division has no explicit zero guard. This normalization differs from
  master's per-patch CLAHE and from a plain RGB pipeline.
- `DataSpliter.py` identifies masks and explicit splits, assigns unassigned
  samples per dataset, and attempts to derive validation. Its
  `add_valid_if_not_exists_random` condition is inverted: no existing validation
  causes an early `continue`; existing validation causes training repartitioning.
  Iteration over Python `set` objects makes RNG assignment depend on hash order;
  `if self.seed` also fails to seed when seed is zero.
- `Dataset.py` samples task, image, annotator and crop independently of `idx`,
  including validation. Dataset length is nominally ten times image count.
  A/V is encoded as one scalar `0.3*artery + 0.6*vein + 0.9*unknown`, with a
  `# fix this` comment; this is not a sound multiclass target for sigmoid Dice.
- `Config.py` currently enables disc only; commented alternatives enable
  vessels or vessels/disc/cup. A/V is commented out. `Unet.py` and task-selective
  losses implement multiple output channels with incomplete task annotations.
- `train.py` trains from scratch using task-selective Dice, AdamW and an LR
  schedule; defaults are 145 epochs, patch 256, filters 32, depth 4. It selects a
  best checkpoint but calls final evaluation with `model_name`, the last-epoch
  path, rather than the copied best checkpoint.
- `evaluate.py` performs full-image weighted tiling via `predict_by_parts.py`.
  It uses CUDA, no outer inference/no-grad context, and whole-object model load.
  Its annotation loop repeatedly reads `file[name][mask_type_use]`, so secondary
  annotator entries can duplicate the primary mask instead of evaluating theirs.
  `predict_by_parts.py` assumes large-enough images and uses `W == None`, which
  is ambiguous when callers supply an array.
- `optimization.py` tunes models; `evaluate_single_vs_multi.py` compares tasks;
  `evaluate_jiri.py` compares nnUNet predictions. Historical test-driven tuning
  must not be repeated for a clean held-out benchmark.
- `prepare_data_nnUNetv2.py`, `prepare_ubmi_data_nnUNetv2.py`,
  `generate_dataset_json.py`, `file_and_folder_operations.py` and
  `convert_segmentations_to_original.py` provide nnUNet interchange. They are not
  evidence that nnUNet model weights are present.
- Windows environment batch files and `requirements.txt` are platform-specific;
  tracked `__pycache__` files should be removed from the clean working tree.

## MATLAB loader inventory

The master `load_*.m` set covers ARIA, AVRDB, CHASEDB1, DiaRetDB0/1, DRHAGIS,
Drishti-GS, DRIVE, E-Ophtha, Erlangen, FIRE, G1020, HEI-MED, HRF, IDRiD,
INSPIRE-AVR, IOSTAR, MESSIDOR, ONHSD, RC-RGB-MA, REFUGE, REVIEW, RIDB, RIGA,
RIM-ONE, ROC, STARE, UoA_DR and VICAVR. Not every loader supplies dense target
masks: some provide only images for pretraining, lesion annotations, point
coordinates, or scalar AVR measurements. Number of loaders is not the number of
usable supervised datasets.

Disc/cup examples: REFUGE decodes dark labels (`disc <150`, `cup <50` in MATLAB);
Drishti reads soft maps, later thresholded by output writing; RIGA infers filled
regions from differences between original and six annotated images; RIM-ONE r3
reads cup masks for stereo images. RIGA contour extraction is heuristic and needs
visual verification; it must not be silently advertised as a faithful simple
label conversion. Other MATLAB files are HDF5/image output helpers, directory
creation/traversal, pretraining resaving and exploratory scripts.

## Local data and model provenance

Data root: `/var/home/tom/ownCloud/fundus_databases`. Most databases are ZIPs,
with long shared-drive prefixes inside them. See `DATASETS.md` for adapters
actually supported by the clean implementation. The local `NOTES.md`, dated
2026-10-19, is a prior inventory; provenance claims from it are identified below.
Its statements that code folders are full clones could not be relied on in this
session: direct git access there failed, so a fresh upstream clone was made.

A/V label encodings:

| Source | Encoding / caveat |
| --- | --- |
| Old HRF/DRHAGIS labeler output | 0 background, 50 unknown vessel, 100 artery, 150 vein; often half resolution and dilated |
| New HRF/DRHAGIS `clasified/*.png` | RGB: red artery, blue vein, green unknown vessel; the supplementary `drive-download-...zip` contains these |
| RITE | Red artery, blue vein, green crossing/uncertain, white unknown; shares source images with DRIVE |
| IOSTAR | Red artery, blue vein; TIFF masks contain compression artifacts requiring careful thresholds |
| AVRDB | Separate colored-line JPEGs on white; inconsistent artery/vein filename spelling |
| VICAVR / INSPIRE-AVR | Points / scalar AVR references; not dense A/V masks |

The prior local inventory explains that new HRF/DRHAGIS RGB labels were filled
from the half-resolution annotations, upscaled and clipped by original vessel
segmentation. Their geometry should therefore agree with that original mask.
Unknown and crossing A/V pixels must be ignored for artery/vein loss and scoring,
while their available vessel segmentation remains valid. DRIVE and RITE must
share image-level partitions to prevent leakage.

Two saved-model families exist under the external `models/` directory:

1. `Ves_Class_AV_UNet/net_v0_0_1.pt` through `net_v0_0_9.pt` (~8 MB each),
   plus v8/v9 pickle histories. They are whole Python model pickles requiring
   the historical `Network` module. The prior inventory reports v8/v9 validation
   accuracy 0.900/0.906, but exact independent image splits are not saved and the
   training code has overlap described above. These are historical reports only.
2. `single_net_retina_jednotna_sit/`: best checkpoints for multitask, vessels-only
   and disc-only runs; `test_best`, `test_vessels_seg_best`; nnUNet prediction ZIP;
   commit and uncommitted-diff provenance files. The provenance file identifies
   `7a152c31c2947ddf760716dfa8a6dfc4d2b881c9` (2023-11-13).
   According to the prior inventory, these were copied from the MUNI share in
   2026; large preprocessed HDF5 files and most per-epoch checkpoints were not.

The actual historical text `vysledky_s_neoptimalnim_diskem.txt` records Dice:
multitask vessels 0.631237, disc 0.654517, cup 0.285497; vessels-only 0.680068;
disc-only 0.681068 / 0.667593. These figures lack a reproducible matching split
and preprocessing artifact here and **are not a baseline measured in this audit**.
The prior inventory says A/V was disabled in single-net checkpoints; A/V there
was handled by nnUNet. The supplied `nnUNet_raw_predict.zip` contains predictions,
not trained nnUNet weights. Do not claim those weights have been recovered.

Large pretraining data is unavailable by user instruction. None is required for
the cleaned supervised workflow. The extensionless `ziajP7YB` is documented in
the prior inventory as a broken G1020 duplicate; prefer valid `G1020.zip`.
Remote host ada was mentioned by the user, but its availability/hardware and
synchronized paths are not assumed from this local audit.
