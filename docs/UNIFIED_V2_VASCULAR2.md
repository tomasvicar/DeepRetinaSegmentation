# V2 vascular sampling preview

`unified-v2-vascular2` is an **experimental five-output RGB model**, trained from
scratch on 21 suitable TRAIN sources. Its run completed 60 epochs; the published
checkpoint is the validation-selected **E59**, which already used all 2,529 TRAIN
photographs and all 5,700 eligible annotation variants. It is not a final optimized
model, an all-task winner or TEST-qualified. Original models, V1 and V2 control
remain available with unchanged aliases.

```bash
uv run deepretina predict --model unified-v2-vascular2 \
  --input /path/to/fundus.jpg --dataset drive \
  --output predictions/vascular2-preview --device cpu
```

Choose the photograph's actual source/camera; `drive` is only an example. For a
DRIONS photograph use `--dataset drions`. The checkpoint carries the portable
V2 camera policy, whose DRIONS 30° scale is a documented research assumption.
All five masks, probabilities, overlay and metadata use the original geometry.

## What changed and what was selected

Training used `vascular_sampling_weight=2` with source/task-balanced sampling:
existing vessel and A/V contributions receive twice the sampling weight of each
disc/cup contribution. This changes sampling, not the A/V loss weight or inference
threshold. Seed 42, patch 384, batch 4, base 12 / depth 4 / BatchNorm and mild augmentation
are recorded in the immutable checkpoint. Missing labels remain invalid, and A/V
labels stay paired with their vessel annotation. No pretraining or teacher model
was used.

The unchanged `best.pt` is 5,453,234 bytes, SHA256
`81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b`.
E59 has 120,832 successful optimizer steps, with minimum actual visits of 64 per
photo and 2 per annotation. The complete run reached 122,880 steps at E60.
The E59 training-selection macro is 0.8038980927; independently produced fresh
VAL macro is 0.8038980942. The published model is E59, not terminal E60.

## Validation tradeoffs

On the same 770 VAL photos / 612 reviewed split groups, threshold 0.5:

| Metric | Strong original reference | Vascular2 E59 |
| --- | ---: | ---: |
| Dataset/head macro Dice | 0.667030 | 0.803898 |
| Pooled vessel Dice | 0.770123 | 0.756713 |
| Pooled artery Dice | 0.689522 | 0.718048 |
| Pooled vein Dice | 0.697286 | 0.779477 |
| Pooled disc Dice | 0.894808 | 0.933591 |
| Pooled cup Dice | 0.435399 | 0.874871 |
| Conditional A/V accuracy | 0.748566 | 0.810257 |

**Vessel remains lower than the strongest original reference:** pooled Dice delta
−0.01341, paired 95% interval [−0.01895, −0.00800]. The higher macro/A/V results do
not remove that regression. Artery's pooled interval is positive, while its
source-macro interval crosses zero; source-specific tradeoffs remain relevant.
Keep the original vessel route when that tradeoff matters.

Intervals use 2,000 source-stratified split-group resamples with seed 42, conditional
on the selected checkpoints. This is shared training-validation evidence after
multiple experiments, not independent TEST or training-seed uncertainty. Known
photo/source/patient groups are protected, but opaque IDs do not prove biological
patient independence. Original-model historical exposure and preprocessing/backend
differences remain qualifications. This release does not establish a causal
sampling-only gain. HPO continues separately; no final candidate or TEST protocol
is selected here.

## Provenance and release checks

Training source is `0b8762e`, with manifest 17a2277c, canonical 00d9ef19 and geometry
policy 2591d846. See [the compact release evidence](../reports/release/v2-vascular2-preview/README.md)
and [the full paired validation reports](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/1a2ea5f/reports/vascular2-paired-val).
The four newly added disc/cup sources contribute TRAIN data; the fixed VAL domain
is not an independent heldout assessment of all four new domains.

A CPU-only public CLI check used one verified DRIONS TRAIN photograph with no GT
scoring. All five original-frame outputs were finite and respected A/V exclusivity,
A/V inside vessel and cup inside disc. The wheel contains this model plus all ten
previous checkpoints, whose bytes and original aliases remain unchanged. These
are functionality and packaging checks, not quality or general speed evidence.
