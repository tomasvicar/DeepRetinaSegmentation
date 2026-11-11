# Vascular sampling checkpoint compatibility

`UnifiedConfig.vascular_sampling_weight` is a positive finite scalar with default
`1.0`. It allows the normal safe loader to read completed checkpoints whose
training recipe recorded this field. Old checkpoints without it still receive
the original default. Existing positional configuration fields retain their order.

For `dataset_task_balanced` training, a nondefault scalar multiplies the existing
`vessel` and `av` source/task contributions before photo sampling. Disc/cup
contributions remain unchanged. This is a sampling parameter, not a loss weight or
inference threshold. Nondefault values require that sampling policy; booleans,
nonfinite values and nonpositive values are rejected.

The default branch retains the exact former weight expression and RNG path.
Observer selection, crops, augmentations, architecture, loss, inference and
training functions are unchanged. This compatibility change does not migrate or
resume any frozen experiment and adds no BF16 or coordinator functionality.

Verification is in [the release evidence](../reports/release/vascular-sampling-compat/verification.json):

- 42 CPU tests cover exact default weight bits, sampler draws/RNG state, scalar
  validation and safe loading of both old and scalar-bearing checkpoint configs.
- The existing bundled V2 control produced bitwise identical probabilities and
  masks to its saved CPU release output on one verified DRIONS TRAIN photograph.
- The audit-approved completed vascular2 terminal checkpoint produced bitwise
  identical original-frame probabilities and masks under this code and its
  frozen training source on that same TRAIN photograph. Its immutable SHA256 is
  `1f8f9a942b4c7286819ee84f667e4b37cce7db5432d1702cda50f0701eedb62d`.
- All ten bundled weight files remain byte-identical. No model, alias or preset
  was added, and no GT mask, validation/test score or training step was used here.

This proves bounded functional compatibility, not model quality. Publishing or
selecting the vascular2 model remains a separate validation and release decision.
