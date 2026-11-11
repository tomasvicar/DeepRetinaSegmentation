# Optional native CUDA BF16 training

Unified checkpoints may record `mixed_precision = "none"` (default) or `"bf16"`.
The field is appended after the existing configuration fields. Existing models,
recipes, aliases and default training behavior are unchanged. BF16-trained models
retain float32 parameters and optimizer state; their saved configuration can be
loaded safely for ordinary float32 CPU inference.

To opt in with the existing public CLI and an audited corpus:

```bash
uv run deepretina train-unified \
  --manifest data/unified21/manifest.json \
  --config configs/unified-v2-vascular2-bf16.toml \
  --output runs/vascular2-bf16 --device cuda:0
```

This preset differs from the measured vascular2 recipe only by `mixed_precision`.
Alternatively add that field to your own TOML. CPU training and CUDA devices
without native BF16 support are rejected before opening a manifest or creating
outputs; there is no silent CPU emulation, FP16 mode or fallback to another
precision. Changing precision cannot resume a checkpoint under another recipe.
No existing frozen experiment is migrated by this feature.

During opted-in training, autocast covers the forward and loss call, with logits
converted to float32 for loss arithmetic. Parameters, gradients and optimizer
states remain float32. Validation and inference for BF16-config checkpoints
explicitly disable autocast, including an enclosing caller autocast context.
The default `none` route preserves the prior arithmetic and RNG path.

## Hardware and validation limits

Native BF16 detection is a hardware feature check, **not proof that the installed
PyTorch/CUDA binary can execute kernels on that GPU**. The locked Torch 2.6 / cu124
runtime must also support the selected GPU architecture and driver. Actual
Blackwell sm120 previously passed capability detection but failed kernel execution
with that binary. This feature does not upgrade Torch/CUDA, install a runtime or
promise Blackwell support. Use a compatible runtime/device; the private operational
launcher guard is not imported into this public package.

The source 72 AMP implementation has actual native CUDA training evidence, but
that does **not** count as a native CUDA training-step test of this stable port.
At this commit, a native CUDA BF16 step on the port remains **PENDING** until a
compatible GPU is free. CPU tests cover admission/context behavior, tiny operator
emulation, float32 weights/optimizer states, exact default model/optimizer/RNG
parity against frozen stable, resume rejection and public CLI parsing. CPU
emulation tests bypass only admission and do not authorize CPU BF16 training.

One immutable actual source 72 E3 BF16 checkpoint was safely loaded on CPU and
predicted a verified TRAIN photograph. Original-frame probabilities and masks
matched frozen 72 bitwise; the existing bundled V2 default also matched its saved
CPU output bitwise. This is model handover evidence, not E60 completion, model
quality, GPU throughput or native BF16 validation of the port. No checkpoint was
added to the registry. All eleven existing model bytes and aliases are unchanged.
See [the bounded compatibility evidence](../reports/release/bf16-compatibility/README.md).
