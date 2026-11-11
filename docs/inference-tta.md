# Experimental inference ensembling

Unified checkpoints support opt-in spatial test-time augmentation (TTA) for
prediction and VAL evaluation. The default `none` calls the existing predictor
with exactly its previous arithmetic and report fields. This option does not
change a checkpoint, training configuration, threshold, or decoder.

```bash
deepretina predict --checkpoint checkpoint.pt --input photo.png --output prediction --tta flips2
deepretina evaluate --checkpoint checkpoint.pt --manifest manifest.json --split val --output val.json --tta rot4
python scripts/unified_validation.py --checkpoint checkpoint.pt --manifest manifest.json --output enriched-val.json --tta dihedral8
```

Modes use equal weights in this fixed order:

| Mode | Prepared-image operators | Forwards per tile |
| --- | --- | ---: |
| `none` | identity | 1 |
| `flips2` | identity, horizontal reflection | 2 |
| `rot4` | counterclockwise rotations 0°, 90°, 180°, 270° | 4 |
| `dihedral8` | those four rotations, then each rotation followed by horizontal reflection | 8 |

Raw-image geometry/FOV preparation and checkpoint photometry happen once. TTA
transforms the entire prepared RGB image before the existing tiled or whole-view
predictor. Each probability map is inverse-aligned to the original prepared
shape, including odd rectangular images. Existing FOV restoration and decoding
then run once on the ensemble result.

The FP32 ensemble averages five **joint** probabilities: vessel,
`vessel × conditional artery`, `vessel × conditional vein`, disc, and cup. The
shared conditional recovery function then normalizes the two averaged AV joint
probabilities. Zero AV support gives 0.5/0.5. Averaging AV conditionals directly
would give a low-vessel-confidence view the wrong influence on the decision.
Vessel gating, AV argmax, cup containment, annotation support and metrics retain
their existing implementation.

Python callers can append the keyword `tta="flips2"` (or another mode) to
`predict_probabilities`, `predict_unified`, or `evaluate_unified_model`.
`produce_validation_evidence` accepts the same keyword. Existing positional
arguments retain their meaning. TTA is not a training configuration field.

Opt-in reports record the operators, transform domain, FP32 joint combination,
and experimental scope. Enriched opt-in reports have kind
`experimental_unified_tta_validation_evidence`; the ordinary standalone
selection gate rejects this evidence. A new matched `none` report is needed for
an experiment. Earlier reports from a different implementation/runtime are not
silently reused as that baseline. These VAL experiments do not establish
held-out quality, authorize TEST evaluation, or promote a default mode.

Synthetic CPU tests cover default arithmetic/RNG parity, inverse alignment,
parameter/buffer preservation, joint weighting and zero support, preprocessing
once, unchanged GT/support traces, CLI propagation and rejection by standalone
selection. Scientific results, when available, must report all five heads and AV
accuracy alongside the aggregate and cost, including adverse changes.
