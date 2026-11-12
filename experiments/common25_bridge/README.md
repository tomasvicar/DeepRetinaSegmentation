# Experimental RAW35 → common25 probability bridge

This helper is outside `src/` and has no CLI, dataset, checkpoint, image, GT or
model loader. It is not imported by production or by the frozen source72/6d.
The original 9b850 RAW35 canary and all existing geometry/selection gates are
unchanged. No real-data or neural-network experiment has been run here.

The interpolation convention is fixed **before any future VAL quality result**:

- Each axis maps target pixel center `j` to `(j+0.5)*source_size/target_size-0.5`.
- Five FP32 planes use bilinear/order1 interpolation, nearest-edge extension
  (`skimage mode="edge"`), `preserve_range=True`, `anti_aliasing=False`, `clip=False`.
  NumPy2.5.4, SciPy1.18.1 and scikit-image0.26.0 are required, not inferred from a
  different installation's lockfile. These versions match the earlier canary.
- Source TTA must already have been inverse-aligned and fused on its prepared35
  frame. Convert conditional A/V to joint `vessel*conditional`; resize joint
  V/A/V/D/C once to exact stored25 dimensions; recover conditional A/V using their
  interpolated joint sum. At zero joint class mass the conditional values are.5.
- **No source35 FOV premasking** is performed. The helper accepts no FOV arrays.
  Supply prepared-grid probabilities before `predict_unified`'s FOV masking or
  RAW restoration, not already masked/restored predictions. Apply the fixed25
  domain only through the unchanged25 scorer and its original GT/valid/FOV;
  never resize GT or categorical predictions.
- Identical25→25 geometry takes an explicit independent-copy short circuit:
  no interpolation or joint/conditional round trip, hence bit-exact FP32 values,
  including signed zeros. Different-grid results are numerical FP32 results,
  not a claim of bit identity with a historical resize operator.

`Grid` requires native photo/group identity, TRAIN/VAL split, canonical owner,
original frame, inclusive crop, camera angle, ppd, prepared dimensions, RAW
profile, geometry-policy SHA and distinct manifest/content SHAs. Both grids must
share photo/group/split/owner/frame/crop/angle/policy; each prepared shape must
agree with its native maximum-dimension scale and inclusive crop. Target must be25.
TEST is rejected. Shapes are integer tuples; callers normalize JSON arrays to
tuples explicitly. Different source35 and measurement25 corpus pins are kept
separate rather than retagged equal.

These metadata checks do not authenticate caller-provided provenance. A future
reviewed producer must additionally bind actual35 producer/source/backend,
manifest/content and original encoded resources, and the actual fixed25 manifest
17a/content00d9/policy2591. It must check exact770 photos/993 annotation records,
612 known split groups, canonical owners, sorted observer/A-V-priority traces,
head-source attribution and GT valid/positive support against unchanged0a5c/6808
references. It must retain the interpolation descriptor/backend, checkpoint role,
training geometry and measurement geometry independently in new experimental
reports. None of this enables final selection or TEST, and the helper cannot
verify that an upstream producer did not already mask its inputs.

## Synthetic checks actually run

The fourteen cases use only small synthetic arrays. Manual expected values cover
half-pixel independent-axis3×4→2×3 ramps, clamped1×2→1×4 edges, constants, a
partial-FOV boundary (.35 without premasking versus erroneous.2), opposing A/V
joint interpolation (.3 versus conditional-average.7), zero vessel, exact-copy
identity, geometry/owner/photo/group/policy mismatches, wrong dtype/shape and TEST
metadata rejection. Input probabilities are not modified.

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 4 nice -n 17 EXISTING_PYTHON -m pytest -q \
  experiments/common25_bridge/test_probability_bridge.py
```

Actual final result: **14 passed in0.45s**, exit0; Ruff check0. The first13-case
run passed0.90s. An intermediate14-case run passed0.44s while Ruff reported two
import-order issues; those were corrected without changing arithmetic and the
final checks above reran. No dependency installation, Torch import, real photo,
mask/model decoding, forward, CUDA, remote query or corpus/cache build occurred.
