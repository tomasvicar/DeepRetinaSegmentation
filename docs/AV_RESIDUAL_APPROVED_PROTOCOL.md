# Approved fresh common-source LR0.0003 comparison

Root reviewed the source, full563-test result, actual historical LR0.0003
run.json config and full cached-width16 E1 preservation gate. This approves
preparing the next controlled trial; it does not declare improved quality.

Source `3157bb957fcf6e6f632a5c49d6df2c8a78ea3747` is a new separate immutable
Ada checkout. Execution SHA8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b;
training-source SHA dd69012a1741000ae5841634094094397c690a2371c9a91aa411344a436f1136.
The independently checked runtime is unchanged Torch2.6cu124 in the original
uv-managed environment,20 intra/inter-op threads, no OMP/MKL/OPENBLAS/CUBLAS
overrides inherited from SSH. Source/runtime proof is recorded beside the plan.
Use the same Ada GPU and policy for both fresh trials; record each actual launch.
Wait for current width16 E8 terminal status, independent milestone and QA first.

Approved plan: `reports/unified/plan-av-residual-cached-approved-3157.json`,
SHA256 `6be48b57fc85e8ffba7772058562aa5a8bf855f9cb445c5180484111ffba5778`.
The candidate changes only architecture; the control matches every actual
historical LR0.0003 config value, including cuda:0. Both use canonical61032…,
full verified cache4e39…, all2221 train photos,8192 samples/E, seed42, batch4,
patch384, mild augmentation, coverage-first and source/task sampling. No old
optimizer/RNG checkpoint is imported. No teacher or unavailable pretraining.

Scheduler horizon remains60. First invocation stops atE8 to verify coverage,
finite model+optimizer state, exact source/data/config/plan pins, scheduler
T_max60, checkpoint history and validation evidence. Then resume the SAME fresh
control with --resume --final through60. E8 is not the requested final model.
The residual architecture receives a matched E8 comparison, and a matched longer
budget only after reviewing actual A/V and V/D/C curves and cost. Any claim of
improvement must compare equal completed budgets and disclose single-seed limits.

Use macro_dataset_mean_dice as the predeclared selection score, and report pooled
and source-macro per-head Dice, conditional/end-to-end AV, support, paired group
uncertainty, parameters, timings and memory. Original strong refs include calibrated
AV plus dedicated disc0/raw primary and disc1/raw pooled alternative. These two
disc rankings differ. Aggregate gains must not conceal vascular regressions.
Test data remain untouched. This file records approval/preparation, not launch.
