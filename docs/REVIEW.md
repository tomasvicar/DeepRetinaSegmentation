# Independent implementation review

Review date: 2026-11-10. This records the review of the new Python package and
corrections incorporated during integration. Executed experiments and their
limits are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

## Findings addressed during integration

| Finding | Final behavior |
| --- | --- |
| Original-resolution input could change anatomical scale relative to prepared training images | Manifest records `preparation_max_size`; training requires one matching policy across train/validation and saves it as `inference_max_size`; prediction downsamples RGB with Lanczos, infers at that scale, and restores probabilities bilinearly before thresholding at original geometry |
| Failed preparation rerun could partially overwrite data behind an unchanged manifest | Preparation rejects a nonempty/non-directory output before mutation; use a new or empty output. An initial failed preparation may leave partial files, so retry into a fresh directory |
| A custom manifest could place a shared source in different splits | Every manifest read checks `source_id` across all tasks/splits and rejects overlap; when missing, image path is the fallback identity |
| A/V scoring could be mistaken for full foreground/background artery segmentation | Intended support is confidently labeled reference vessel pixels; independent vessel metrics cover segmentation. Gated class accuracy and conditional artery/vein accuracy are reported separately; `metric_support` and accuracy-description fields explicitly identify their support |

The overwrite defect was demonstrated with temporary synthetic DRIVE data: a
second preparation with smaller maximum size changed the first image before
failing on a missing second mask, while leaving the manifest unchanged. The
new fail-fast output rule prevents this. Source archives were not modified.
The regression suite covers existing-output refusal, scale metadata and original
output geometry, source overlap rejection, ignored-label gradients and checkpoint
round trips for all three tasks.

## A/V score interpretation

Artery/vein targets and losses use known reference vessel pixels only. Unknown,
crossing and unlabeled vessel pixels do not become false class negatives.
Background is supervised by the vessel channel and gates exported class masks.
Consequently, artery/vein Dice and precision exclude false positives on reference
background; they describe class predictions within explicitly labeled vessels,
not a complete artery/vein segmentation benchmark over the FOV.

`av_accuracy` requires the correct final thresholded and vessel-gated class, so
missed vessels and rejected classes count as errors. `av_conditional_accuracy`
compares artery and vein probabilities on the same reference pixels without the
vessel gate or probability threshold. Read both with the independent vessel
Dice/recall and labeled-pixel count. This is an intentional two-part segmentation
and classification contract, not a reason to treat uncertain labels as negatives.
Exports include a per-channel `metric_support` map and explicit descriptions of
both accuracy variants. `by_dataset` metrics aggregate counts under the same
support as the pooled scores.

## Architecture and migration scope

The new model is a randomly initialized RGB U-Net. Its encoder channels grow
geometrically from configurable `base_channels`; blocks use GroupNorm and SiLU,
and decoding uses bilinear upsampling followed by convolutions. Defaults are
base 24 and depth 3. The old custom U-Net used linear channel growth,
BatchNorm/ReLU and transposed convolution; master also supported an
EfficientNet-B2 encoder. The legacy A/V model additionally required a supplied
vessel mask as a fourth input channel. These architectures and weights are not
interchangeable. New runs do not automatically load any historical checkpoints.

RGB is normalized to `[-1,1]`. Preparation caps image dimensions rather than
reproducing camera-angle scaling, MATLAB FOV heuristics or historical contrast
normalization. Python-only execution therefore means a supported functional
rewrite, not bitwise MATLAB conversion or reproduction of historical scores.
All historical dataset adapters are now mapped and implemented; exact annotation
semantics and validation scopes are in [DATASETS.md](DATASETS.md) and the migration
audit. UBMI still requires its private source for real-data verification. Old source
remains retrievable through Git.

## Remaining limits and follow-up opportunities

- DRIVE and RITE share source identities; CHASE paired eyes stay together.
  Hash-ranked grouping is independent of traversal order. Patient independence
  cannot be asserted where archives expose only image identifiers.
- Runtime split checks compare declared identities; they cannot detect duplicate
  photographs saved under unrelated names or missing patient metadata.
- Predictions on original pixels are restored from the prepared inference scale.
  Resizing probabilities before thresholding can differ from scoring the prepared
  masks; use one declared scale/evaluation protocol when comparing experiments.
- Directory prediction reloads its checkpoint per image. This is functionally
  correct; profiling can guide a later batched-loading optimization.
- Random initialization, uniform crop sampling, normalization, architecture and
  overlap choices require controlled quality comparisons. Synthetic round trips
  and successful optimization establish functionality, not superior accuracy.

This reviewer changed documentation only; source corrections were incorporated
by the implementation owners.
