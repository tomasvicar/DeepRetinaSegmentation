# Composed original-model prediction

For one raw RGB photograph and a supplied original-frame binary FOV mask:

```bash
uv run deepretina predict-original \
  --input fundus.png --fov fov.png --dataset drive \
  --output predictions/original --device cpu
```

Use a new or empty output directory. The FOV must have zero background and one
positive foreground level. Supply a known camera/dataset or an explicit
`--field-of-view-degrees`; special calibration branches such as RIGA additionally
need an independently established `--fov-diameter-pixels`. Do not guess a camera
angle or substitute a bounding-box diameter. This command accepts one photo.

```python
from deepretina import predict_original

result = predict_original(rgb_uint8, fov=binary_fov, dataset="drive", device="cpu")
# Arrays have original-frame dimensions, with channels vessel, artery, vein, disc, cup.
masks = result.masks                      # bool, (5, H, W)
scores = result.probabilities              # float32, (5, H, W)
# Pass output="new-directory" to export PNG masks, overlay, NPZ and prediction.json.
```

The fixed policy combines the bundled dedicated vessel, disc0, universal cup and
v9 AV models. It prepares raw single-net RGB once at 25 pixels/degree, uses each
checkpoint's native tiling, and thresholds raw logits strictly above zero. Cup
intersects the **same dedicated disc0 forward** in prepared coordinates before
nearest restoration. AV uses the existing calibrated provided-FOV MATLAB-style
operator on the original RGB and predicted binary vessel mask, then the existing
full-frame AV classifier. AV logit0 is unused; ties between artery and vein remain
unclassified. No final FOV clipping or extra original-frame vessel intersection
is added. Calibrated vessel resampling can change support at boundaries.

Masks are the decisions used by the benchmark. Exported vessel/disc/cup scores
are independently interpolated sigmoid scores; the cup score itself is not
clipped by disc. Exported artery/vein scores are interpolated conditional softmax
scores gated by the nearest-restored calibrated **binary** vessel support. They
are not unified-model soft-vessel joint probabilities, and thresholding these
interpolated scores need not reproduce the exported masks. The NPZ also includes
`av_conditional_probability` and `av_vessel_support`. Metadata records all four
loaded component SHAs, input byte/array identities, geometry, native tiling,
calibration, probability semantics and fixed policy ID.

This exposes the `inside-disc` composition in the research
[`benchmark_strong_pipeline.py`](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/33390381b759d2957d16f89bf6bd5165056e10a9/scripts/benchmark_strong_pipeline.py)
`original_operation`, with frozen quality-policy source SHA
`44f1faf4ff41b3dd60b8d6dea747a577bcfd2bb77b1f2366e153bc688fb27a61`.
Historical model training exposure and raw/normalized input provenance remain
qualified; the supplied-FOV calibration and Python raw photometry do not certify
arbitrary-camera MATLAB parity or clinical performance. Benchmark metrics do not
certify a new photograph. This route does not select new weights or alter model
aliases. A real TRAIN DRIVE21 check on 2026-11-12 confirmed exact agreement of all five
binary masks with the frozen benchmark operation (see evidence below).

## Integration verification

The 11 new CPU tests compare actual toy-network forwards against a literal copy
of `original_operation`, including native tiling, calibrated photometry and mask
restoration; they check output metadata, probability/support semantics and
pre-load refusal cases. The broader CLI/output/calibration regression selection
passed 90 tests in 11.76 seconds. Some existing regressions deserialize bundled
weights and forward small synthetic CPU images. No real photographs, GT or GPU
were used by those tests. The separately authorized real TRAIN parity check
below then exercised the complete composition.

On 2026-11-12, a single raw DRIVE21 TRAIN photograph (584×565) with its provided
FOV passed complete FP32/CPU1 comparison against the actual frozen benchmark
`original_operation(cup_policy="inside-disc")`: all five masks were bitwise equal,
all scores finite, four checkpoint identities verified, and CUDA uninitialized.
Child and external observer returned 0 in 43.1s, within 600s/8GiB bounds. Candidate
and reference reused the same verified loaded objects; this is not a latency
comparison. Two earlier pre-forward failures (identity hash convention and
runtime-only tiling argument selection) are preserved alongside the successful
[original evidence](../reports/original-composed-real-parity-20261112/README.md).
This one-photo result establishes neither all-camera parity nor segmentation quality.
