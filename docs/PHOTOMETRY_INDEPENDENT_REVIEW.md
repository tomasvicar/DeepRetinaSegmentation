# Independent photometry review:2476e4c

Read-only review of feature commit2476e4c71e35d5949da50043726e733ef78d8ee1,
based on frozen72. No feature-source edits, new training, model forwards, actual
GT scoring, GPU work or TEST evaluation were performed. New processed RGB cache
implementation is not part of this reviewed snapshot.

## Concrete issue: benchmark helpers bypass opt-in photometry

`scripts/benchmark_strong_pipeline.py:155` creates a tensor directly from
`prepared.image` and calls `predict_probabilities`. That low-level function
intentionally expects already profile-processed RGB01. For a checkpoint whose
saved profile is CLAHE or LC+CLAHE, this helper would therefore run RAW inputs and
omit the preprocessing cost while reporting complete preprocessing/inference.
`scripts/benchmark_latency.py:150` has the same bypass. Existing RAW checkpoint
measurements are unaffected. Reported to the feature author and root before any
nonraw model measurement.

A minimal safe fix is to reject nonraw checkpoints in these helpers until their
complete raw-photo operations deliberately apply the shared profile exactly once
with the prepared FOV and record the descriptor. A mock/operator test should
verify the transformed tensor and one application without a model forward.

## Verified implementation paths

- `UnifiedDataset._image` checks raw aliases before transformation, obtains FOV
  solely from `photos.json`, and processes the entire prepared uint8 image before
  crop/padding/augmentation. Its bounded LRU key includes profile and physical
  scale. The native fallback tuple retains that processed image.
- `evaluate_unified_model` constructs the Dataset with the checkpoint config;
  it does not reapply a profile inside tiling. `predict_unified` prepares raw
  geometry once and applies the same operator once to full prepared RGB/FOV.
  `read_unified_sample` retains an explicit raw default and configurable opt-in.
- Neither CLAHE implementation mutates the source raw array. The legacy helper
  receives a fresh float32 image; the wrapper restores original uint8 values
  outside FOV. Dataset tensor conversion copies the transposed input, including
  readonly raw derived-cache memmaps.
- Opt-in FOV loading rejects missing/empty/mismatched or escaping image-FOV
  references. There is no fallback from labels, valid masks or target geometry.
  Ordinary loss-validity masking remains separate from photometric FOV selection.
- Raw returns before FOV/backend work. Missing new config fields retain raw;
  nonraw checkpoint load/resume requires matching descriptor and actual package
  versions. `photometry.py` is included in the training-source hash; its reused
  legacy implementation was already covered. Training remains25pixels/degree.
- `selection.inference_protocol` records opt-in descriptor provenance. This does
  not add a new completion-family eligibility allowance or authorize final TEST.

## Bounded checks and remaining evidence

Fourteen pure preprocessing/Dataset/cache invariant tests passed in4.05seconds
on one CPU with CUDA hidden. Six model/checkpoint/training-related cases were
deliberately deselected for this independent no-NN review. The selected cases
exercise deterministic operators, outside-FOV restoration, unchanged target/RNG
paths, missing-FOV rejection, canonical FOV fingerprint mutation, source hash
mutation, existing raw-derived/canonical parity and processed native fallback.
The author's188-case result is separate evidence, not rerun here.

No all-corpus photometric cache exists in2476. Operator/LRU tests do not establish
whole-epoch throughput or benefit. Before GPU comparison, a separately reviewed
immutable processed-RGB cache must distinguish already processed entries from
old raw derivatives, bind canonical RGB/FOV bytes plus profile/backend/source,
and demonstrate cached/uncached equality without a second CLAHE application.
That cache and actual model-quality gains remain outside this review.

## Follow-up7ce20ef

Static review confirms7ce20ef6cd56061a0b75228afb8b1c55b21ed4d3 resolves the
silent RAW bypass by adding an explicit raw-only admission guard to both helpers.
The strong operation rejects nonraw before image preparation or unified forward;
the older latency wrapper checks the loaded checkpoint config before its unified
operation. Missing historical fields retain raw. The scripts do not claim to
benchmark the new profiles; those require a separately reviewed complete timing
protocol. This resolves the correctness finding without changing prior RAW
measurements. The author's53 benchmark cases are separate evidence; no model
forward was rerun for this static follow-up.
