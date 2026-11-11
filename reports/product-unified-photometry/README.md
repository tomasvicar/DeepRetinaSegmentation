# Product unified photometry port — review candidate

This isolated feature branch is based on product `d028314d6a26c6790d82bdc97219027184c051df`.
It ports the full-image input recipe and immutable processed-RGB cache from frozen
`fd44cdfa83d1df9ccc42960df6d1d2018db29900`. It is not merged, selected, released or
promoted. All twelve bundled blob objects, aliases, original legacy source and
existing configs retain their original identities.

`UnifiedConfig.photometric_profile` is appended with raw default. Dataset,
independent row reader, checkpoint load/resume, shared evaluation, public raw-photo
prediction and training-cache CLI wiring retain the saved profile. Preparation
and photometry precede crops, augmentation, TTA and tiling. The current product
model architecture, FP32 joint A/V TTA math, inverse alignment, decoder, threshold,
FOV restoration and occupied-output checks remain intact. No legacy profile or
source-specific MATLAB limitation is redefined.

The operator module and its legacy dependency remain byte-identical to the fd44
recipe. Operator fingerprint is
`196da8626c87d6b07bed6508252dd37328ee79846c4040c1403f9a090002708e`.
The reader admits genuine fd44 cache fingerprints without rewriting their index,
original builder/training identity or descriptors. Checkpoint/cache admission
additionally binds canonical JSON value types, so `true` cannot stand in for
version1 or integer0 for a false boolean. This strengthening changes no operator
bytes or valid historical cache pins. Raw checkpoints missing the new config field
retain their old behavior; dropping the field from a nonraw saved recipe is rejected.

The historical primitive benchmark also recognizes saved nonraw input and reports
its preprocessing separately rather than feeding raw RGB to a photometric model.
No benchmark was executed here; no speedup or quality is inferred from this wiring.

## Verification

The combined synthetic operator/cache/TTA suite observed89 passing checks in8.34s.
After adding BF16-metadata/profile/TTA combinations and a lost-profile rejection,
the updated product integration file observed32 passes in5.61s. Those operator/mock components contain102 checks, all passed by the component
runs. After explicit clarification, three additional tiny training/resume/CLI
checks passed under an external60s cap on CPU6/thread1/nice17/CUDAhidden. The
cache/resume check passed in the first bounded run; the final raw parity and CLI
checks passed together in3.50s. Ruff and
`git diff --check` pass. No real-model test is claimed by this preparation packet.

[Cross-source verification](cross-source-verification.json) checks actual clean
fd44 and sparse genuine d028 Git checkouts, importing their real source files:

- Ten original product model/joint/TTA/decoder function ASTs are unchanged.
- Training function AST matches fd44; strict recipe admission is a separately
  named metadata validator. Authorized tiny generated-data checks additionally verify d028 raw-step
  weights/optimizer/RNG parity, cached/uncached training and relocated-cache
  resume state, and actual CLI saved-profile/cache-identity serialization/load.
  This is synthetic functionality, not real-model quality or TRAIN-image parity.
- All three synthetic75×97 RGB profile arrays and descriptors match fd44 exactly;
  outside-FOV RGB and Torch/Python/NumPy RNG state remain exact.
- d028 and the new code produce bitwise-identical pointwise-mock probabilities and
  decoded masks across all four TTA modes; raw Dataset crops/targets/valid/RNG match.
- The genuine fd44 builder creates each three-photo synthetic processed cache;
  the new reader retains its original construction/source/cache pins. Cached,
  uncached, fd44 and product Dataset tensors match exactly.

Other focused tests cover once-only full-image photometry across all TTA views,
FP32 inference with none/BF16 training metadata, explicit joint-vs-conditional AV
counterexample, missing FOV, no GT fallback, alias/shape/profile/backend/type/pin
mismatch, corruption/no fallback, bounded array lifetime, raw-derived cache parity,
CLI forwarding and prediction metadata. Inputs are generated synthetic RGB and
labels; no real corpus images or GT, operational checkpoint weights or GPU are
loaded by these checks. There is no actual real-image or trained-model parity proof.

## Scope exception in an additional inherited CLI test subset

An accidentally selected existing CLI rejection subset exercised its `trained`
fixture before checking inapplicable flags: **13 tiny synthetic CPU training
fixtures**, base channels2/depth1, two epochs×two samples, batch1/patch32. The whole
16-test CLI subset returned0 in7.52s with CUDA hidden on one CPU affinity. This
exceeded the initial mocked-NN scope and was reported to the parent immediately.
It read no real images/GT/checkpoints, used no GPU and changed no frozen run.
Those initial toy fits are not presented as real-model validation, training
parity or quality. The parent then explicitly authorized bounded existing tiny
synthetic train/load fixtures; the three additional checks above use that clarified
authorization. The original13-fixture scope exception remains recorded separately.

The scope exception and newly authorized toy fits are separate from the
cross-source operator proof, which uses no trained NN or optimizer. Real TRAIN-photo
parity, real operational checkpoint forward,
TRAIN/VAL evaluation, full cache build and any GPU use remain unperformed by this
port. They require a separately reviewed bounded packet if pursued.

See [saved recipe and retraining guide](../../docs/UNIFIED_PHOTOMETRY.md) and
[cache identity/relocation guide](../../docs/UNIFIED_PHOTOMETRY_CACHE.md).

Reproduce the cross-source synthetic proof with a clean genuine fd44 source
checkout and a clean genuine d028 reference checkout (a sparse Python-source
checkout suffices), using this branch's locked Python environment:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:tests uv run python reports/product-unified-photometry/verify_cross_source.py \
  --authority-source /sources/fd44 --product-reference /sources/product-d028
```

The helper only builds temporary synthetic fixtures/caches, reads source/Git
metadata, and writes this report. It never accepts a real corpus or model path.
