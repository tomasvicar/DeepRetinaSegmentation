# Vessel and artery/vein learning-rate search

Work is isolated on `experiments/leipzig-hpo` in
`/var/home/tom/code/DeepRetinaSegmentation_hpo`. The stable `for_Leipzig`
implementation and its model files are not changed by these experiments.

The fixed 181-record manifest is shared read-only from the original pilot data.
Its SHA256 is `9d2d0b349a3b1e37eb159a7b26272e6931b141f8b6ba078b68997d3115f29993`.
Vessels run on old's GTX 1060 6GB; artery/vein runs on Pascal's GTX 1070 8GB,
selected by GPU UUID rather than assuming device order. Both use separate HPO
source directories and `uv sync --frozen --dev` environments. Environment details,
fixed configurations and baseline identities are in
[`vessels_av_launch.json`](../reports/hpo/vessels_av_launch.json).

The search screens learning rates 0.0003, 0.001 and 0.003 for 8 epochs using seed 42,
base 16/depth 3 U-Net, patch 256, batch 4, 8 samples per image, 2 workers and
OMP/MKL thread counts 4. Vessel loss is BCE+Dice; A/V loss is BCE. The top two
nonbaseline candidates are trained from scratch for 30 epochs; the matching
original 30-epoch baseline remains eligible. Selection uses validation only.
The final chosen checkpoint is evaluated on test after selection. Search-stage
validation results must not be presented as test improvements.

The reused baselines are `runs/drive_bce_dice` (validation mean Dice 0.7983568)
and `runs/rite_bce` (validation mean Dice 0.7182782) in the stable checkout.
Reusing them requires matching manifest and training configuration; the search
runner checks compatibility. Baselines are evaluated under the same validation
selection metric as candidates. A/V mean Dice combines vessel and class masks
on their documented support, so retain per-channel scores and labeled-pixel
counts when interpreting any change.

Both searches launched persistently on 2026-11-10; the runner accepted
the original baseline metadata and identical manifest. Logs are
`runs/hpo/av.log` on Pascal and `runs/hpo/vessels.log` on old beneath the HPO
source directories. Initial search metadata and candidate CSVs were created.
The initial LR search is now complete; frozen results are recorded below.

Exact command arguments, GPU selection, baseline checkpoint SHA256s, runner
SHA256 and launch PIDs are recorded in the launch JSON. Commands are:

```bash
# Pascal; run from DeepRetinaSegmentation_hpo
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=GPU-3fcefb45-3c90-4606-2e83-eab875df3cc4 uv run python scripts/hpo.py --task av --manifest data/leipzig/manifest.json --output runs/hpo/av --baseline-run /var/home/tom/code/DeepRetinaSegmentation_clean/runs/rite_bce --loss bce --device cuda --cpu-threads 4
# old; run from its separate DeepRetinaSegmentation_hpo directory
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 uv run python scripts/hpo.py --task vessels --manifest data/leipzig/manifest.json --output runs/hpo/vessels --baseline-run /var/home/tom/code/DeepRetinaSegmentation_clean/runs/drive_bce_dice --loss bce_dice --device cuda --cpu-threads 4
```

The jobs were launched with detached/persistent process handling and redirected
logs. Their CLI uses the three default learning rates, 8/30 epoch budgets and
`top_k=2`; fresh final trials exclude the baseline learning rate. After all short
trials, both nonbaseline learning rates therefore receive 30-epoch trials, and
the existing full baseline remains eligible. No test command is part of these
launches. Experiment outputs belong only to this worktree and its corresponding
directory on old.

The baseline learning-rate screens provided an additional reproducibility check:
on both hosts, all eight screening rows exactly matched the corresponding first
eight original baseline rows in both training loss and validation mean Dice.
Epoch 8 validation mean Dice was 0.7532754438643475 for vessels and
0.44455362414451666 for A/V. This is a reproducibility check, not an improvement.

## Extended quality and complexity search

The next bounded experiment uses the backward-compatible experimental core on
this branch. Seven candidates receive 12 screening epochs each: mild photometric
augmentation plus foreground-biased crops; the same with cosine LR; widths 8 and
24; larger 384-pixel patches; 128-pixel vessel patches or whole-image 512-pixel
A/V context; and BatchNorm. Exact overrides are recorded in
[`extended_plans`](../reports/hpo/extended_plans). Candidate learning rate comes
from the frozen preceding LR selection, not from test results.

The top two screening candidates train from scratch for 60 epochs. A matching
60-epoch control keeps the original architecture, augmentation, sampler, constant
LR and original learning rate. The original 30-epoch model and preceding LR
winner remain eligible. This control separates the benefit of a longer budget
from a new configuration; comparisons among multifactor candidates are not
individual augmentation/architecture ablations. Screening a cosine schedule over
12 epochs differs from its 60-epoch schedule, so screening rank is provisional.

Selection remains validation-only. After freezing it, every eligible finalist is
benchmarked on the same prepared validation photograph, tile 512/overlap 64 and
GPU: two warmups, five repeats, CUDA synchronization and median elapsed time.
The benchmark excludes checkpoint loading and export but includes complete
probability inference, CPU blending and required geometry restoration. Report
parameter count, checkpoint size and peak allocated GPU memory alongside quality.
Whole-image candidates use their actual shared inference dispatch, not the patch
benchmark by accident. No stable model is promoted without root review.

Runner: `scripts/hpo_extended.py`; plans and resolved configurations are portable
JSON evidence. `--dry-run` checks manifest/baseline/config compatibility without
training; `--resume` reuses only complete matching trials, archives incomplete
ones and preserves the original baselines. Explicit test evaluation is a separate
operation requiring a persisted validation-only selection. Runtime metadata logs
GPU environment, software versions, CPU threads and runner SHA256.

## Completed initial LR comparison

All candidates below received 30 epochs from random initialization, except that
the original matching baseline was reused. Models were chosen on validation
before the selected-model pilot test evaluation. Trial histories, primitive
configs, validation reports, final selections and selected test reports are saved
under [`reports/hpo`](../reports/hpo).

| Task | Learning rate | Validation mean Dice | Best epoch | Selected |
| --- | ---: | ---: | ---: | --- |
| vessels | 0.001 | 0.798357 | 30 |  |
| vessels | 0.0003 | 0.803842 | 30 | yes |
| vessels | 0.003 | 0.772621 | 27 |  |
| av | 0.001 | 0.718278 | 30 | yes |
| av | 0.0003 | 0.701156 | 30 |  |
| av | 0.003 | 0.588398 | 27 |  |

Vessels selected LR 0.0003: validation improved from 0.798357 to 0.803842;
the subsequent DRIVE pilot test improved from 0.798755 to 0.802079 (delta
+0.003324). Paired image-bootstrap of pooled Dice difference, 10,000 resamples
of the same 20 images, gave a percentile 95% interval [+0.000049, +0.007045].
This describes image variability for this one trained pair, not training-seed
variability or external-data generalization. See the
[paired comparison](../reports/hpo/vessels_lr_test_comparison.json).

A/V retained its original LR 0.001 model: validation mean Dice 0.718278,
pilot test mean Dice 0.708539. No LR-only gain was found for A/V. Its selected
checkpoint exactly reproduces the original reported pilot test scores.

These frozen LR results remain experimental. They do not modify the stable
`for_Leipzig` package or its model registry. The extended search uses only their
validation-selected learning rates and keeps both original and selected models
eligible; the LR pilot-test outcomes do not determine candidate configurations.
