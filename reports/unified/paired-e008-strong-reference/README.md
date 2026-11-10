# Paired E8 comparison against strong original disc references

The LR .0003, baseline and strong-augmentation E8 checkpoints are compared with
the calibrated original A/V plus dedicated disc0/raw composite from `84f899b`.
The disc1/raw composite is also reported as the original pooled-disc winner.
The primary disc0/raw reference was selected by source-macro disc Dice. Both
references retain the dedicated vessel model and original universal cup counts;
the supported disc counts come from the specified dedicated disc checkpoint.

All six comparisons use exactly 770 validation photographs and a manifest containing
993 annotation variants, grouped into 612 reviewed split groups. Scoring selects the
same first sorted annotation per task, with A/V vessel supervision taking priority;
all record IDs are checked, while annotator uncertainty is not resampled. Every
photo/record ID, source attribution, valid
pixel count and positive GT count matches. Source-stratified paired bootstrap uses
2000 draws, seed 42, with the same group draws across heads and both models. These
are split groups, not 612 proven independent patients.

The analysis reads saved per-photo counts and the manifest only. No new model
inference, masks or test scores are opened. Candidate count files add provenance
to the unchanged validation_metrics exported from independently verified E8 last.pt
milestones; selected-best epochs are also 8. Input/report/code hashes are recorded.

## Source-macro aggregate

| Candidate E8 | Reference disc | Reference macro | Candidate macro | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---:|
| lr0003 | disc0/raw | 0.654498 | 0.708936 | +0.054438 | [+0.041633, +0.066844] |
| lr0003 | disc1/raw | 0.653996 | 0.708936 | +0.054940 | [+0.042090, +0.068405] |
| baseline | disc0/raw | 0.654498 | 0.679359 | +0.024861 | [+0.010647, +0.037452] |
| baseline | disc1/raw | 0.653996 | 0.679359 | +0.025363 | [+0.010465, +0.039074] |
| strong | disc0/raw | 0.654498 | 0.679427 | +0.024929 | [+0.013049, +0.036837] |
| strong | disc1/raw | 0.653996 | 0.679427 | +0.025431 | [+0.012483, +0.038581] |

## LR .0003 E8 per head, primary disc0/raw reference

| Head | Original pooled Dice | Unified pooled Dice | Difference | Paired 95% interval | Source-macro difference |
|---|---:|---:|---:|---:|---:|
| vessel | 0.770123 | 0.722513 | -0.047611 | [-0.057938, -0.038138] | -0.037989 |
| artery | 0.689529 | 0.561068 | -0.128461 | [-0.143791, -0.112735] | -0.138768 |
| vein | 0.697298 | 0.615372 | -0.081926 | [-0.096000, -0.068332] | -0.083601 |
| disc | 0.894808 | 0.849109 | -0.045699 | [-0.062383, -0.029389] | +0.127190 |
| cup | 0.401057 | 0.791965 | +0.390909 | [+0.349711, +0.424756] | +0.405357 |

The pooled vessel, artery, vein and disc differences are negative, with their
conditional paired intervals below zero. Cup is markedly higher. Disc has a
positive source-macro difference despite a negative pooled difference: source-macro
weights databases equally, while pooled Dice aggregates all pixels. The positive
five-head macro does not establish improvement of every task.

These compare complete prediction policies. The original reference preserves
strict logit>0 and adds no cup containment; unified decoding constrains cup to its
predicted disc. The cup difference cannot be attributed to the learned cup head
alone. Both pipelines retain their own A/V and spatial preprocessing policies.

## Why disc pooling and source-macro differ

| Disc annotation source | Photos | Original disc0/raw Dice | LR .0003 E8 Dice |
|---|---:|---:|---:|
| aria | 22 | 0.738979 | 0.716149 |
| drishti | 10 | 0.960409 | 0.954467 |
| g1020 | 204 | 0.857905 | 0.766553 |
| iostar | 6 | 0.004701 | 0.868411 |
| onhsd | 17 | 0.852989 | 0.895001 |
| refuge | 400 | 0.932061 | 0.898587 |
| rimone | 32 | 0.759667 | 0.897872 |

## Missing conditional-A/V uncertainty and qualifications

Conditional A/V accuracy is .652636 for LR .0003 versus .748573 for the calibrated
reference (difference −.095937). The saved per-photo reports omit ungated conditional
correct/scored counts. Gated artery/vein TP/FP/FN cannot reconstruct those values,
so a paired conditional-accuracy interval is not reported. All requested E8 per-photo
Dice reports exist; no forward is requested to fill the missing metric counts.

Intervals are exploratory validation descriptors conditional on these checkpoints,
known source strata and grouping assumptions. They do not include training-seed
uncertainty, validation-selection bias or unrecognized patient correlations. Both
disc variants were selected/inspected on validation. Historical original-model
training exposure remains unresolved. There is no independent clinical or all-task
quality-improvement claim, and no test evaluation.

The generic comparison helper retains a broad raw-proxy qualification. The actual
reference inputs here explicitly identify `vesclass_matlab_provided_fov_calibrated`
A/V and `legacy25_raw` dedicated disc profiles. Their calibration bounds, inherited
native-FOV uncertainty and historical exposure qualifications remain in the source
reports. No reference was silently replaced with universal-disc predictions.

## Reproduce

Requires the referenced report files from `84f899b`, comparison module from
`fb11844`, the canonical manifest with SHA 529edba7..., and this directory.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 5 uv run python reports/unified/paired-e008-strong-reference/run_saved_e008_comparison.py \
  --reference-directory reports/disc-reference \
  --candidate-directory reports/unified/paired-e008-strong-reference/candidate-counts \
  --manifest data/unified25/manifest.json \
  --comparison-module src/deepretina/comparison.py \
  --output /tmp/paired-e008-reproduction
```

`input-provenance.json` records every component/checkpoint/report hash.
`results/summary.json` records Python/NumPy, code hashes, common GT support and
all six comparisons. References remain in their existing report directory rather
than being duplicated here.
