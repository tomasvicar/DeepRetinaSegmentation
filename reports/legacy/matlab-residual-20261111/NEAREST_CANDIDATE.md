# Narrow production candidate: nearest coordinate arithmetic

This separate followup changes ONLY `imresize_nearest_mask` coordinate arithmetic.
The Sobel implementation remains the original production implementation. It uses
MATLAB's observed one-based inverse mapping, consistently with the existing
bicubic path:

```
scale = output_length / input_length
position = u / scale + .5 * (1 - 1 / scale) - 1   # u=1..output_length
nearest = floor(position + .5)
```

The old algebraically equivalent `(u-.5)*input_length/output_length` associates
float64 operations differently. At 1152→1005, zero-based output row837 selects
source959 with the observed MATLAB expression, but960 with the old expression.
The actual retained DiaRet resized FOV differs at(837,127) and(837,1164), both
MATLAB true. No epsilon, bin adjustment, dataset/shape conditional or target
change is added. `test_matlab_nearest_coordinates.py` isolates this using row
labels rather than committing a retinal photo or mask.

`production-nearest-comparison.json` replays ALL24 retained synthetic nearest
fixtures and ALL5 automatic source resized-FOV goldens against the changed
production helper: every mask matches exactly, including DiaRet. No additional
MATLAB process was needed. The complete raw DiaRet replay in
`full-auto-nearest.json` preserves the historical74165/max271 result and measures
nearest-only265/max7; remaining photometric/Sobel differences are NOT fixed or
rebaselined. The3/max1 research-Sobel result is not the production candidate.

Ada CPU0/nice15/CUDAhidden:190 tests passed in2.18s across the new nearest tests,
MATLAB AV/FOV/source-auto/source-CLI and original single-net arithmetic tests.
Ruff check/format pass. This is a review candidate, not activation in stable or
frozen training/evaluation checkouts. No network forward on real data, no GT,
heldout metrics or model-quality claim was produced by this followup.
