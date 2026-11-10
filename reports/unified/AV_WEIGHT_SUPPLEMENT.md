# Controlled conditional A/V loss-weight screen

The sole candidate `av_weight_2_lr0003` is copied exactly from the original frozen
`lr_0003` configuration. Only `av_weight` changes from1 to2. It keeps source87,
canonical data6103, seed42, LR.0003, mild augmentation, BatchNorm/base width12,
batch4,8192 samples/epoch, eight-epoch screen and unchanged60-epoch cosine horizon.
It starts from scratch, without the new derived cache or any source migration.

Plan: `plan-av-weight2-lr0003-87cf045.json`, SHA256
`afeb05f71945824ea415e2c9aacdcd33d540305dac18ab770f714db63445aa05`.
Parent frozen plan SHA256
`907713f6c4a94c7281e2a5e226777fb02a5cdf613b0d79c20e0fe14831cf5e14`.
The source87 CLI accepted the plan through real argument parsing, source/content
checks and configuration construction with an asserting training stub. No model,
optimizer or GPU job was created by that acceptance proof. Details are in the JSON
proof beside the plan. Runtime launch still requires terminal E8 verification.

Rationale: E3/E5 validation showed vascular/A/V regression relative to the strong
original reference. The source87 loss averages supported head losses by their
weights; doubling conditional A/V cross-entropy weight increases its relative
contribution while preserving valid-pixel selection and conditional decoding.
This is a measured ablation proposal, not an improvement claim. Other heads may
improve or regress and must be reported separately.

Launch guard: original `lr_0003` must finish E8 with finite weights/optimizer, all2221
photos covered, correct source/data/plan pins, scheduler horizon60, and independent
E8 checkpoint milestone. Report E8 pooled artery/vein Dice and conditional A/V
accuracy to root before launching. If all three already exceed the calibrated
reference, pause and consult root. Reference values are artery.6895286745,
vein.6972977562, conditional accuracy.7485732693, from validation-only calibrated
profile `vesclass_matlab_provided_fov_calibrated`, report SHA256
`fc1cb3c0889f8280932634b979c0b53a873d5fab7ebe7298200ef72917045812`.
Historical training exposure remains unresolved; this comparison does not prove
independent generalization. No test evaluation or changed heldout partition.

After the guard passes, use only the idle old GTX1060 with its explicit UUID and a
fresh output outside immutable source. No overlap, restart or `--final`. Separately,
Pascal may start original-plan `width_8` after its baseline terminal E8 QA/milestone.
