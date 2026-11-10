# Isolated Leipzig hyperparameter experiments

This branch, `experiments/leipzig-hpo`, is based on stable commit `320e084`.
The migration and the ready-to-send Leipzig models remain in their own branch.
Search outputs, data links and environment files stay in this worktree; stable
model directories and ownCloud databases are read-only inputs.

The runner `scripts/hpo.py` performs a bounded learning-rate search on the original
181-record pilot manifest. It neither prepares new annotations nor changes the
original train/validation/test partition. The required manifest SHA-256 is
recorded per search and checked against the baseline checkpoint and run metadata.
The original pilot manifest used in these runs has SHA-256
`9d2d0b349a3b1e37eb159a7b26272e6931b141f8b6ba078b68997d3115f29993`.

## Search policy

- Learning rates: `0.0003`, `0.001`, `0.003`.
- Short screening stage: 8 epochs per learning rate, initialized from scratch.
- Final stage: train the top two **nonbaseline** learning rates for 30 epochs
  from scratch, ranked by their best validation Dice during screening. With the
  default grid and a `0.001` baseline these are `0.0003` and `0.003`.
- The original, configuration-matched 30-epoch baseline is always eligible in
  final selection, even when its short-stage trial performs poorly. Its model
  directory is never overwritten, retrained or copied into an active trial.
- Fixed seed 42, U-Net base width 16/depth 3, patch size 256, batch size 4,
  8 patches per training image per epoch, 2 data workers, threshold 0.5,
  validation tile size 512 and overlap 64. Disc/cup and vessels use BCE+Dice;
  A/V uses BCE to match its original selected baseline.
- Candidate scores and winner selection use **validation only**. The winner can
  remain the baseline; a search does not force an improvement claim.

Every candidate starts from the same fixed seed. The best epoch within each
trial is selected by full-image validation. This is a single-seed exploratory
comparison on an existing pilot split. It does not establish improved accuracy
across datasets or independent replications.

## Installation and launch

Run commands from this experiment worktree. Link or copy the existing prepared
pilot data; do not regenerate it using later migration adapters.

```bash
uv sync --frozen
mkdir -p data
ln -s /absolute/path/to/stable/data/leipzig data/leipzig
uv run python scripts/hpo.py \
  --task disc_cup \
  --manifest data/leipzig/manifest.json \
  --output runs/hpo/disc_cup \
  --baseline-run /absolute/path/to/stable/runs/drishti_bce_dice \
  --device cuda --cpu-threads 4
```

Use task `vessels` and baseline `runs/drive_bce_dice` for vessels. Use task `av`,
`--loss bce` and baseline `runs/rite_bce` for artery/vein classification. The
baseline must match every model/data/training setting other than device placement
and the candidate learning rate. The baseline's full epoch budget, manifest hash,
validation score and checkpoint metadata are checked before any training begins.

For machines with several GPUs, set `CUDA_VISIBLE_DEVICES` explicitly or pass
`--device cuda:N`. Keep separate task output directories and one experiment per
GPU. A persistent launch can redirect all output to a task-specific log:

```bash
mkdir -p runs/hpo
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python scripts/hpo.py \
  --task disc_cup --manifest data/leipzig/manifest.json \
  --output runs/hpo/disc_cup \
  --baseline-run /absolute/path/to/stable/runs/drishti_bce_dice \
  --device cuda --cpu-threads 4 \
  >runs/hpo/disc_cup.log 2>&1 </dev/null &
```

Resume an interrupted search with the same arguments and `--resume`.
Configuration, paths and manifest identity must match exactly. Finished trials
are reused; an incomplete trial is retained under `interrupted/` and restarted
from scratch. Optimizer-state resumption is not claimed. Do not launch a second
process against an output directory while its first process remains active.

## Artifacts and held-out evaluation

Each search writes:

| Artifact | Meaning |
| --- | --- |
| `search.json` | Immutable search configuration, input paths and manifest hash |
| `runtime.json` | Host, PyTorch version, CPU thread count and start time |
| `configs/*.json` | Exact per-trial `TrainConfig` settings |
| `stage_lr_*/`, `final_lr_*/` | Independent checkpoints, CSV history and validation metrics |
| `candidates.csv` | Completed candidate scores, best epochs and model paths |
| `progress.json` | Last completed trial and completion time |
| `selection.json` | Frozen validation-only winner, baseline and measured delta |

Search never reads old `test.json` files and never calls test evaluation. Only
after `selection.json` exists may the selected checkpoint be evaluated once:

```bash
uv run python scripts/hpo.py \
  --manifest data/leipzig/manifest.json --output runs/hpo/disc_cup \
  --evaluate-selected-test --device cuda
```

This writes `selected_test.json`, references the selection-file hash, and refuses
to overwrite an existing selected test result. Selection is never updated after
observing the test result. The same held-out split was already reported in the
original pilot; this is clearly labeled in the result and is not a pristine,
previously unseen benchmark.

## Launch inventory

Disc/cup runs on Ada in `~/code/DeepRetinaSegmentation_hpo`, using its independent
`uv` environment, the original prepared-data link and read-only baseline
`~/code/DeepRetinaSegmentation_clean/runs/drishti_bce_dice`. Persistent log:
`runs/hpo/disc_cup.log`. A/V and vessel jobs use the same runner on separate
machines/GPUs; their launch details and eventual results belong to this experiment
branch. None of these jobs is required to finish before sharing the stable
Leipzig package and its already validated model weights.
