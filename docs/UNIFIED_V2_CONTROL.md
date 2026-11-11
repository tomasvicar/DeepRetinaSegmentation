# Unified V2 control: fixed terminal E60

`unified-v2-control` is a usable five-output RGB **control baseline**, trained from
scratch on 21 suitable TRAIN sources. It is not an optimized/final model or an
improvement on every task. Existing original models, V1 baseline, pilots and
aliases remain available. No heldout TEST result is claimed.

```bash
uv run deepretina predict --model unified-v2-control \
  --input /path/to/fundus.jpg --dataset drive \
  --output predictions/unified-v2 --device cpu
```

Select the actual dataset/camera; do not label every input as DRIVE. The checkpoint
carries the portable V2 camera policy, including explicitly nominal assumptions
for some new sources. For a known DRIONS photograph, use `--dataset drions`; its
30° scale is a research convention, not a measured camera setting. Output masks,
probabilities, overlay and metadata retain the original photograph geometry.

## Why E60, rather than the VAL-best snapshot

Published bytes are the immutable **terminal `last.pt` from epoch 60**, 14,385,264 bytes,
SHA256 `10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2`.
They are copied unchanged, including saved optimizer/RNG state. The registry role
is `fixed_terminal_epoch_complete_annotation_coverage`; `selected_epoch=60` does
not mean best validation epoch.

The completed run has 2,529 actual TRAIN photos and all 5,700 eligible annotation
variants contributing through successful optimizer updates, across 21 sources.
Minimum actual visits: 66 per photo and 2 per annotation. It completed 122,880 steps
and 491,520 delivered samples over 60 epochs with a 60-epoch scheduler. Missing
cup/A/V labels remain invalid, and alternate observer targets remain paired.

The run's VAL-best checkpoint is E52, but that snapshot has only 5,699 annotation
contributors. It is not the model distributed under this all-annotation control
name. E60 training-selection macro was 0.7887626316; independently produced fresh
E60 VAL macro is 0.7887627334. These separately recorded values are not silently
substituted for one another.

## Fresh validation tradeoffs

On the same 770 VAL photos / 612 reviewed split groups, with threshold 0.5:

| Metric | Frozen original reference | V2 terminal E60 |
| --- | ---: | ---: |
| Dataset/head macro Dice | 0.667030 | 0.788763 |
| Pooled vessel Dice | 0.770123 | 0.750472 |
| Pooled artery Dice | 0.689522 | 0.677109 |
| Pooled vein Dice | 0.697286 | 0.754314 |
| Pooled disc Dice | 0.894808 | 0.934122 |
| Pooled cup Dice | 0.435399 | 0.872808 |
| Conditional A/V accuracy | 0.748566 | 0.788037 |

**Vessel regresses:** candidate−original pooled Dice −0.01965, paired 95% interval
[−0.02392,−0.01534]. Artery is lower in point estimate, with an interval crossing
zero. Vein, disc/cup and conditional A/V are higher on this VAL. The overall macro
increase cannot justify an all-task superiority claim. Retain the original vessel
and A/V models when those tradeoffs matter.

Intervals use 2,000 source-stratified split-group resamples, seed 42, conditional
on these checkpoints. Known patient/alias groups are preserved, but opaque group
IDs are not proven independent patients. This is exploratory training-validation
evidence, not training-seed uncertainty, clinical evidence or a heldout test.
Historical original training exposure and preprocessing/backend differences remain
qualifications. Decoding/postprocessing policies are explicit; no threshold tuning
or new TEST evaluation occurred for this control release.

## Reproduce training and inspect provenance

Follow [RAW-to-training](UNIFIED_QUICKSTART.md) to build your own audited 21-source
manifest, then train a fresh run with `configs/unified.toml`. The released control
used immutable source `6b3dfad`, manifest `17a2277c`, canonical content `00d9ef19` and geometry
policy `2591d846`. These complete hashes, terminal coverage, 60-row training CSV,
fresh validation summaries and comparison qualifications are retained in
[release evidence](../reports/release/v2-control-preview/README.md). A fresh RAW
build is not promised to reproduce the historical artifact or model bytes.

The four newly added disc/cup sources contribute TRAIN data; the fresh VAL keeps
the original fixed 770-photo domain. It is not an independent heldout assessment
of all four new domains.

The raw CLI was additionally exercised on a verified TRAIN DRIONS photo with no
GT scoring. All five masks/probabilities kept 400×600 geometry, probabilities were
finite, A/V exclusive and inside vessel, and cup inside disc. This is a release
functionality check, not an accuracy or speed benchmark.

A separate one-photo CPU timing on DRIVE TRAIN image 21 measured medians of
60.58 s for the four-network original pipeline and 5.44 s for this control
(Ryzen 1600, one CPU thread, one warmup and three repeats). This does not establish
general throughput or better quality. [Measured timing and protocol](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/benchmark/strong-pipeline-cpu-drive21-v2-terminal60).

Further vascular sampling/loss experiments and model optimization remain separate
from this control. A final candidate/protocol must be frozen and reviewed before
the heldout TEST is evaluated.
