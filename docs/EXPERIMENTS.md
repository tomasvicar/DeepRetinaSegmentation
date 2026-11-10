# Experiments and validation

## Environment

All commands use `uv sync --frozen --dev`, Python 3.12 and PyTorch 2.6.0+cu124.
Pascal: GTX 1070 8 GB (GT 1030 also present). Ada: RTX 4070 Ti SUPER 16 GB.
old: GTX 1060 6 GB. CUDA device order can differ from nvidia-smi; use `deepretina doctor`.
The initial fresh-session libvirt check reported a shut-off Windows domain, but a process
check found the original VM still running in a different Flatpak session. After connecting
to its original socket, it was shut down gracefully with `virsh shutdown windows --mode
agent,acpi`; QEMU exited and available RAM rose to 53 GiB. Training pilots had sufficient
memory and completed while the VM was still running. See WORK_LOG.md for the exact command.

## Scope

Tests cover label decoding, source-group split isolation, nearest-neighbour labels, tiling
coverage, geometry, ignored-label gradients, safe checkpoints and train/evaluate/predict for
all three tasks. Real ZIP preparation is checked separately from synthetic tests.

Runs and measured results will be recorded below as they complete. Pilot runs are not a
replacement for a multi-seed external-dataset study. Historical scores in REPOSITORY_MAP.md
come from external archived notes, not a re-evaluation of those weights here.

## Executed software checks

- `uv sync --frozen --dev` and `deepretina doctor` succeeded on all three GPUs/machines.
- `uv run pytest -q`: **18 passed**, including CLI setting precedence, all three task
  round trips, ignored-label gradients, safe loading, split checks, resize restoration,
  data decoding and refusing to overwrite a prepared dataset.
- Ruff check and format checks passed. Wheel and source archive built successfully.
  Inspected archive members: no images, masks, weights or virtual environment included.
- All eight adapters were exercised against the actual ZIP archives. The data agent
  checked 1,615 generated records (including all 1,200 REFUGE records) for array shape,
  dtype, range, nonempty targets, cup containment and source-group split isolation.
- Training dataset `data/leipzig` contains 181 records, maximum image side 1024:
  DRIVE 16/4/20, RITE 16/4/20, Drishti 40/10/51 (train/val/test).
  DRIVE and RITE share the same photographs, so these are task-specific experiments,
  not independent patient cohorts.

## Controlled loss comparison

Six runs start with random weights, seed 42, 30 epochs, batch 4, patch 256,
base channels 16, depth 3, 8 samples/image/epoch, AdamW lr 0.001, 2 data workers,
RGB normalization [-1,1], tile 512 / overlap 64, threshold 0.5.
Each task's BCE and BCE+Dice runs used the same GPU, data manifest and budget.
Vessels/AV each took 960 optimizer steps; disc/cup took 2,400. Validation selected
both the best epoch within a run and the winning loss. Tests were evaluated only
for the three selected winners, after all loss comparisons had finished.

Manifest SHA256: `9d2d0b349a3b1e37eb159a7b26272e6931b141f8b6ba078b68997d3115f29993`.

| Task / data | Machine | Best val mean Dice, BCE | Best val mean Dice, BCE+Dice | Candidate delta | Selected |
| --- | --- | ---: | ---: | ---: | --- |
| Vessels / DRIVE | old, GTX1060 | 0.787102 | 0.798357 | +0.011254 | BCE+Dice |
| Vessel+A/V / RITE | Pascal, GTX1070 | 0.718278 | 0.709250 | -0.009028 | BCE |
| Disc+cup / Drishti | Ada, RTX4070 Ti SUPER | 0.771472 | 0.792756 | +0.021284 | BCE+Dice |

These are single-seed validation improvements for this budget. They do not establish
statistical significance, superiority over legacy/published architectures, or transfer
to other datasets. BCE+Dice was explicitly **not promoted for A/V** after it scored worse.
All three TOML presets reproduce the measured configuration and validation-selected loss.
The engine's generic default is BCE; use the task preset to reproduce these pilots.

Reproduce each pair using a new output directory (example for vessels):

```bash
uv run deepretina train --manifest data/leipzig/manifest.json \
  --config configs/vessels.toml --loss bce --output runs/reproduction/drive_bce
uv run deepretina train --manifest data/leipzig/manifest.json \
  --config configs/vessels.toml --loss bce_dice --output runs/reproduction/drive_bce_dice
uv run python scripts/compare_runs.py \
  runs/reproduction/drive_bce runs/reproduction/drive_bce_dice
```

Use `configs/av.toml` or `configs/disc_cup.toml` for the other pairs. Thread settings used
on all machines: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. A dedicated device was used
for each task. Exact Python patch versions and GPU capabilities are in the environment
JSONs; PyTorch was 2.6.0+cu124 throughout. Seeds and deterministic-algorithm requests are
recorded, but unsupported deterministic operators warn, so bitwise equivalence across
hardware/software versions is not promised.

## Held-out test results of validation-selected models

| Model | Test images | Channel Dice | Additional A/V scores |
| --- | ---: | --- | --- |
| DRIVE, BCE+Dice, best epoch 30 | 20 | vessel **0.798755** | — |
| RITE, BCE, best epoch 30 | 20 | vessel **0.807128**, artery **0.635681**, vein **0.682808** | conditional class accuracy **0.721889**; gated end-to-end accuracy **0.585488** |
| Drishti, BCE+Dice, best epoch 28 | 51 | disc **0.879785**, cup **0.764842** | — |

The exact per-channel values and per-image counts in JSON are authoritative (the table
is rounded). A/V artery/vein scores use **known reference vessel pixels**, excluding
reference background and uncertain/crossing labels. Conditional accuracy ignores vessel
detection; gated accuracy includes missed detections/rejected classes as errors. Always
read these alongside the independent vessel scores. These are pooled confusion-count
metrics at **prepared resolution**, not original-resolution published benchmark numbers.

Predictions were exported and visually inspected on DRIVE test 01 and Drishti test 001,
including probabilities, named masks and overlays. The raw Drishti image was 1751×2049,
inferred at 875×1024, then restored to 1751×2049; output geometry was checked explicitly.
The disc/cup pilot has visible false positives on a bright image boundary, and validation
curves are volatile (including a poor late epoch). These are concrete model limitations;
they were recorded rather than tuning postprocessing against the inspected test image.
A future comparison should examine sampling/context/normalization on validation, multiple
seeds and an untouched external cohort. The current networks are working pilot models.

## Saved evidence and weights

[`reports/2026-11-10`](../reports/2026-11-10) retains manifest snapshot, environments,
all six training curves/configs/validation scores, three loss comparisons, final test
reports and selected model hashes. It contains no photographs, masks or checkpoint weights.
Early baseline validation JSONs predate the addition of explicit support-description
fields; their score semantics match the later test reports and definitions above.

Local weights remain in `runs/{drive_bce_dice,rite_bce,drishti_bce_dice}/best.pt`.
A temporary ownCloud backup of the three selected weights and metadata was removed
after the user clarified the storage constraint. It occupied 6,005,545 bytes (5.73 MiB).
All three local checkpoints were verified byte-identical before removing that new backup
directory. Keep subsequent outputs in this repository working directory, outside ownCloud.

Example outputs are under `predictions/{vessels,av,disc_cup}`. Existing archive/model
folders were not modified. Prepared images, outputs and weights are ignored by Git.
