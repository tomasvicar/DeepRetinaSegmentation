# Optional unified inference TTA

Unified checkpoints accept an explicit `--tta` mode. The default `none` keeps the
previous inference arithmetic and report schema. Original single_net/A/V and
pilot models reject non-`none` TTA; their existing prediction routes are unchanged.

```bash
uv run deepretina predict --model unified-v2-vascular2 \
  --input /path/to/fundus.jpg --fov /path/to/fov.png --dataset drive \
  --output predictions/tta-example --device cpu --tta flips2
```

Use a new output directory. As with ordinary prediction, an existing nonempty
prediction directory is preserved and rejected. A custom unified model can use
`--checkpoint /path/to/model.pt` instead of `--model`. The same option is available
for public `evaluate --split val`; TEST remains blocked by that interface.

| Mode | Full prepared-image views |
|---|---|
| `none` | Identity only; existing default |
| `flips2` | Identity and horizontal flip |
| `rot4` | Rotations 0°,90°,180°,270° |
| `dihedral8` | Those four rotations and each followed by a horizontal flip |

Raw image preparation happens once. Each view transforms the whole prepared RGB
image before tiling (or whole-view letterboxing). Outputs are inverse-aligned on
that prepared grid. The implementation averages FP32 vessel, vessel×conditional
artery, vessel×conditional vein, disc and cup probabilities, then recovers the
conditional A/V distribution. Averaging conditional A/V probabilities directly
would give low-vessel-confidence views too much influence. Existing FOV and
original-image restoration happen after this ensemble, followed by the existing
vessel-gated A/V and Cup-inside-Disc decoder. This does not port experimental
CLAHE training or photometry caches into the product.

Views run sequentially. More views increase inference work; use `flips2` for a
smaller increase or measure the cost of `rot4`/`dihedral8` on the target device.
Non-default prediction and VAL metadata include the exact operators, averaging
policy, FP32 precision and experimental scope under `tta`. BF16 training metadata
remains intact: inference does not turn BF16 autocast on. Non-default TTA also
disables an enclosing autocast context. Default `none` retains the existing
precision behavior, including exact arithmetic for ordinary product inference.

## Evidence and limits

The [matched TPE BEST58 VAL study](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/41277cbad992e58132ac26c7b9903075639371e9/reports/tpe72-inference-tta-val/README.md)
measured these modes on one immutable checkpoint and all770 fixed VAL photos.
It reports aggregate gains, adverse source/head results and increased invocation
cost. Vessel Dice still trails the frozen original reference. That development
study does not guarantee improvement for every bundled model, source or image;
it is not TEST evidence or a global product quality claim. Default-none latency
must not be assigned to an ensemble's quality result.

The [product compatibility report](../reports/product-unified-tta/README.md)
checks this port without another quality evaluation: actual BEST58 weights on a
small asymmetric prepared RGB input match the pinned joint-TTA implementation in
all four modes, and default `none` matches the previous product bitwise. CPU tests
cover inverse geometry, joint A/V weighting, zero-vessel fallback, FP32/BF16
metadata, once-only preparation/restoration, CLI forwarding and output guards.
These checks do not constitute a new full-corpus or clinical validation.

TTA changes the inference protocol. An earlier default-none completion,
validation or frozen-selection record does not automatically certify another
mode. This feature changes neither trained tensors nor registry/default models,
and it does not enable held-out TEST execution.
