# Proposed MetaCentrum paired control/residual screen

**Planning only; root approval and a separately reviewed private wrapper are
required. No allocation or model run was submitted by this change.** The new plan
preserves source 3157, its exact execution hash, canonical/cache identities and both
approved Ada configurations. It does not edit or resume the Ada plan.

Plan SHA256:
`b1fb46320a32f7c39814de18cf5a34beb5c55dc8afe7e2d482854c773cd1f8e2`.
Parent approved Ada plan SHA256:
`6be48b57fc85e8ffba7772058562aa5a8bf855f9cb445c5180484111ffba5778`.

| Fixed input | Pin |
| --- | --- |
| Source commit | `3157bb957fcf6e6f632a5c49d6df2c8a78ea3747` |
| Execution sources | `8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b` |
| Checkpoint training sources | `dd69012a1741000ae5841634094094397c690a2371c9a91aa411344a436f1136` |
| Canonical schema2 content | `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e` |
| Manifest bytes | `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69` |
| Derived cache | `4e39e40def03d0f2859dd147107a3736a62b4fc0249dd440a1280a9803aa8591` |

The two candidate dictionaries are exact value copies of the parent. Their sole
between-candidate configuration difference is:

| Trial ID | Architecture |
| --- | --- |
| `lr_0003_control_cached` | `compact_unet` |
| `lr_0003_av_residual_cached` | `compact_unet_av_residual` |

Both keep seed 42, LR .0003, base 12/depth 4/BatchNorm, mild augmentation, batch 4,
patch 384/view 512, workers 2, 8192 samples/epoch, coverage-first source/task balancing,
loss weights 1 and the 60-epoch cosine schedule. Each fresh run stops at completed
E8 without `--final` or `--resume`. Historical Ada screens are context; no checkpoint
or source-guard migration is allowed.

## One allocation, two sequential fresh processes

The private wrapper must enforce `protocol.execution_order`, not start two jobs.
Both trials use the same single assigned GPU UUID, allocation ID and image digest.
One two-hour allocation requests CPU 4 and GPU 1; memory/scratch/image/staging readiness
are separately reviewed by the private workflow. No arrays or parallel trials.

Common runtime for both: OMP/MKL/OpenBLAS environment 4, explicit Torch intra-op 4,
interop 1, common PYTHONHASHSEED 42. The environment is set before Python startup;
Torch thread settings are applied in each fresh process before application imports
or parallel work. Record the actual values before each trial, plus GPU/driver/
CUDA/cuDNN/Torch/image/CPU metadata. Do not rely on environment fields alone to set
Torch interop threads. No additional model warmup, training steps or data change.

A possible dispatch pattern for the separately reviewed launcher is:

```python
# Environment was set by the common allocation wrapper before this process.
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
assert torch.get_num_threads() == 4
assert torch.get_num_interop_threads() == 1
# Record runtime metadata, then dispatch the pinned source CLI with runpy.
# No app source is patched; each trial starts a fresh Python process.
```

The source per-trial CLI intentionally ignores scheduling/runtime metadata in the
plan. The new private wrapper therefore implements this protocol and freezes its
own hash separately. Passing the plan to an old one-trial wrapper does not establish
that the pair/runtime/budget conditions were met.

## Predeclared control gate and budget

Start one monotonic clock at allocation entry, before staging, imports or setup.
After terminal control E8, verify finite best/last weights and last optimizer,
65536 delivered samples/16384 optimizer updates, all 2221 training photographs and
all 17 sources, exact configuration/source/manifest/canonical/cache/runtime/GPU pins,
cosine T_max 60/last_epoch 8. Preserve an independent checkpoint/metadata milestone.

Only after that gate consider the residual trial. **Skip it when elapsed allocation
walltime through setup/control/QA is greater than 2700 seconds.** Exactly 2700 seconds
is allowed by the declared boundary. Persist the elapsed time and explicit skip
reason; do not move the second trial to another GPU/job or extend the allocation.
The two-hour budget includes both trials and cleanup. Unknown actual GPU timing is
resolved by the first pair rather than asserted from Ada timings.

If control fails, is partial or violates any gate, residual is not started. If
residual is interrupted before E8, retain actual completed-epoch checkpoints/counts
and call the result a partial pair, not a matched 8E result. A control-only result
is preferable to an unreported budget extension. Cleanup/staging failures remain
operational failures even when a model checkpoint is scientifically recoverable.

## Validation evidence and interpretation

Use train/validation only; no test prediction, tuning, model selection or dependency
installation. Canonical integrity may hash referenced test bytes solely to preserve
frozen schema2 identity. The portable manifest is `/data/manifest.json`, the runtime
cache is `/cache`, source `/workspace`, plan `/config/plan.json`, and fresh outputs
are separate children of writable `/runs`. Data/cache/source binds are read-only.
There are no account names or workstation paths in the new plan.

Keep all completed-epoch CSV learning curves. Export full validation dictionaries
from E8 last.pt and selected best.pt for each trial: per-head/source TP/FP/FN/valid
pixels, pooled and source-macro Dice, support and exact photo/record identities.
Source 3157 already emits per-photo `av_conditional_counts`: correct, scored_pixels,
eligible_pixels and end_to_end_correct. Aggregate by the A/V annotation source in
channel_datasets, not an unrelated canonical photograph alias. Verify those sums
against global conditional/end-to-end accuracy; do not add another GPU forward to
recover values that are already saved. Unknown/ambiguous reference labels remain
ignored identically. Pairing must reject unequal GT/eligible-pixel support.

The existing macro_dataset_mean_dice remains the selection metric. Report artery,
vein, vessel, disc and cup separately, conditional/end-to-end A/V accuracy, parameter
count and actual train/validation/setup/total timings. An aggregate gain alone does
not establish improvement of all tasks. Eight epochs are a single-seed architecture
screen, not a final trained model or clinical quality result. Source 3157 preserves
initial backbone/logits and global RNG when adding the zero-initialized residual
head; trained V/D/C quality is not guaranteed to remain unchanged.

The intended single-factor contrast is within this shared allocation. Different
hardware/thread runtime prevents treating Ada as its causal control. Control-first
order may warm caches/GPU clocks and confound operational timings; one seed does
not measure seed robustness. Any later paired uncertainty uses reviewed split
identities, which are not all proven independent patients. Longer matched budgets
and submission require separate root decisions.

`provenance.json` records static pin/config/budget-boundary checks only. No Torch
model, optimizer, inference, package install, model test or submission was performed
for this planning artifact. Stage/probe cleanup readiness is a separate gate; this
plan does not declare the currently reviewed private CPU 11 workflow GPU-ready.
