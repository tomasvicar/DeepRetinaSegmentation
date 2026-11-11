# Current full-data experiments

Snapshot: **2026-11-11, 12:40 UTC**. This is a record of observed results and
prepared next steps, not proof that a process is still running. The final
optimized model and held-out TEST comparison are **not finished**.

## Use the released models

For installation, prediction and original verified models, use
[`for_Leipzig`](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/for_Leipzig).
It includes the original five checkpoints and explicitly identified unified
baselines/previews. Installation uses `uv`; inference needs no MATLAB.
This research branch is not a substitute for the product branch.

The original models remain the quality reference. The original stored-input
arithmetic is numerically checked. The calibrated Python raw A/V profile has
measured MATLAB references and saved-model sensitivity checks, with remaining
photometric differences and unresolved historical training-file provenance.
See the product branch's `docs/MATLAB_AV_PROFILE.md`.

## Data and completed comparisons

The current V2 preparation contains **2,529 TRAIN photos / 5,700 annotation
records from 21 sources**, 770 VAL photos and 856 TEST photos. Partially labeled
records supervise only their supported outputs. Splits retain recovered
patient/alias groups; collections without recoverable patient identity are
TRAIN-only. This does not establish complete patient independence or resolve
the original models' unknown historical exposure.

| Model role | Full training | Fresh VAL pooled vessel Dice | Fresh VAL conditional A/V accuracy |
| --- | --- | ---: | ---: |
| Original reference | Historical training unavailable | 0.77012340 | 0.74856642 |
| V2 control LAST E60 | All 2,529 photos / 5,700 records | 0.75047217 | 0.78803663 |
| Vascular sampling2 LAST E60 | All 2,529 photos / 5,700 records | 0.75622289 | 0.80210746 |
| Vascular sampling2 BEST E59 | Completed E60 run; BEST itself covers all records | 0.75671283 | 0.81025700 |
| Vascular sampling4 LAST E60 | All 2,529 photos / 5,700 records | 0.75318596 | 0.82312537 |
| Vascular sampling4 BEST E55 | Completed E60 run; BEST itself covers all records | 0.76038707 | 0.82288321 |

These roles are separate: `validation.json` beside the original training run is
the BEST sidecar, not a fresh LAST evaluation. LAST-to-LAST is the matched-budget
comparison. Artery/vein discrimination accuracy is conditional on the reference
class-labeled vessel support; it does not by itself measure vessel discovery.

Evidence: [corrected original VAL](../reports/frozen-original-val-nearest),
[control terminal VAL](../reports/v2-control60-terminal-val), and
[vascular2 paired VAL](../reports/vascular2-paired-val). The vascular2 LAST versus
control LAST vessel Dice gain is 0.00575 (paired 95% interval
[0.00226, 0.00934]); it still trails the original vessel model. These intervals
describe the recorded validation groups, not seed variability or independence
of every patient. No all-task improvement is claimed.

Sampling4 also completed training, independent safe checkpoint checks and fresh
LAST/BEST VAL. Its BEST versus sampling2 BEST vessel gain is 0.00367
(paired 95% interval [0.00203, 0.00564]); conditional A/V improves by 0.01263.
Macro gain has an interval crossing zero, while G1020 disc/cup regress.
Sampling4 still trails the original vessel model and does not replace a
released default. All eight matched-role comparisons, including adverse
results, are in [the paired reports](../reports/vascular4-paired-val).

Eight [thresholds](../reports/vascular2-thresholds) and TRAIN-only
[BatchNorm recalibration](../reports/vascular2-bn-recalibration) were actually
evaluated. Neither justified replacing the parent/default threshold0.5.
Rejected changes remain documented. A bounded CPU benchmark of the shipped
vascular2 BEST is in
[the measured comparison](../reports/benchmark/strong-pipeline-cpu-drive21-vascular2-best59):
6.03s versus63.32s resident median on one DRIVE TRAIN photo/one CPU thread,
1.11M versus5.94M parameters. It is not a general throughput claim.

## Current search and frozen source families

| Family | Frozen source | Scientific role |
| --- | --- | --- |
| Vascular sampling2/4 | `0b8762e667c997a2dfd129dc65988c1f26f6b2ad` | FP32, width12/depth4/BatchNorm, mild augmentation; weight4 changes only vascular sampling relative to weight2 |
| V3 adaptive | `72fb321187eca517823083aca80cedace9605570` | Native CUDA BF16 training; FP32 parameters/loss/VAL/inference; width12/16, BatchNorm/GroupNorm, basic/mild/strong augmentations, LR, weight decay and vascular sampling search |
| Native Optuna core | `8777d6cca60f7ced08798ad2af7fabf626421d05` | Typed rung validation, native Optuna5 TPE and staged promotion |

All new full-budget runs retain 60 epochs ×8,192 samples, batch4, patch384,
seed42 and the original 60-epoch cosine horizon. The live V3 study uses epochs
3/9 as diagnostics, first pruning at27 and final60, with protected control.
The source72 configuration fields are not supported by this branch's older
trainer merely because this overview is present: execute from the specified
frozen source and its own dependencies. Do not alter a live checkout to make
an old loader accept a new checkpoint. Product BF16 compatibility has now passed
an actual independent native CUDA synthetic training-step parity check on Ada;
this does not establish full-training quality or general GPU equivalence.

At this snapshot two MetaCentrum allocations remain running: protected control
and trial4, both continuing beyond validated E27 toward E60. Three other
qualified E27 configurations were genuinely pruned; their allocations ended.
Those main-study suggestions were **startup exploration**, not learned TPE.
The own-Ada sampling4 run is complete. Exact access scripts, PBS job handles, allocation
usage and failure records are kept in the private `metacentrum-workflows` repo.

## Next TPE stage and final selection

The main study requires eight COMPLETE/PRUNED startup observations. RUNNING and
failed trials do not satisfy that requirement. Its time budget cannot justify
calling startup suggestions optimized models.

A [separate fixed27 proposal](../reports/fixed27-proposal) was actually made
after verifying Ada was free. It imported all five then-available unique
qualified E27 LAST observations, including promoted controls, into a new native
Optuna5 study, and made one post-startup TPE proposal across the original
six-parameter space with startup4. Parent trial states and the live scientific
protocol stayed unchanged. This proposal has been training on Ada since
10:53 UTC: width16/depth4/BatchNorm, basic augmentation, learning rate
0.000287722894608172, weight decay 0.00000552164819184925 and vascular sampling
weight 2.553934065831788. It passed the pinned partial E27 checks and resumed
the same run toward E60. At E27 only 5,686 annotation variants had contributed;
partial success does not make this a final eligible model.
Four or five observations in six dimensions are weak evidence, not proof of
optimality.

The proposal must receive a fresh complete60 run. A stop/resume at27 must keep
the configuration's `epochs=60`, scheduler horizon and RNG state; a partial27
completion cannot pass the full-model completion gate. Final eligibility needs
genuine terminal60, finite weights/optimizer, all5,700 annotation contributions
in the selected role, independent fresh VAL and correct source/QA bindings.
Freeze the checkpoint, thresholds and inference protocol before TEST. Do not
retag a source-specific completion receipt as another model family or select
on TEST results. The objective remains a better simple unified model alongside
the original verified models, with measured quality, size and speed.

## Controlled patch and photometry study

The current runs fix patch384 and raw25; they do not optimize these choices.
The [next study plan](../reports/patch-photometry-study) fixes the actual TPE
configuration above and batch4, comparing raw patches256/384/512 and, at
patch384, HSV CLAHE and legacy illumination correction plus CLAHE. Each arm
retains60×8192 samples and the same cosine horizon. Different patches still
change pixel/context and compute budgets; these differences must be reported.

The opt-in full-frame Python operators are implemented on a separate branch.
Their immutable TRAIN/VAL RGB cache and shared curated PBS runner are being
verified; no full-corpus cache build or new patch/photometry training arm has
been launched at this snapshot. The same operator must be used before TRAIN
cropping and before VAL/raw inference tiling. A higher physical resolution
requires re-preparation from RAW, not interpolation of the25ppd cache.

A separate bounded inference-only study is actually running on Old, using
sampling4 BEST E55 and all770 VAL photos:512/64 baseline, then512/192,
768/192 and1024/256. It tests inference tiling, not training patch size. Its
original reports and actual exits will remain distinct from the original
512/64 completion evidence. TEST remains unused during tuning.
