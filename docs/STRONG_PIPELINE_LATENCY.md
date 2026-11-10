# Measuring the selected original chain against the unified model

Use `scripts/benchmark_strong_pipeline.py` for the calibrated four-network
reference. The earlier `benchmark_latency.py` measures a different, weaker
universal+AV proxy chain; its measurements must not substantiate speed against
this stronger reference.

The original chain consists of dedicated vessels, calibrated original AV,
dedicated disc0 (primary dataset-macro selection) or disc1 (pooled alternative),
and universal single_net for cup. Pass one of the complete calibrated reference
JSONs from `reports/disc-reference/`; every component's actual checkpoint SHA
must match that report. Original networks retain their own native tiling and
strict decoding. Raw single_net geometry is computed once and shared by three
networks, a reasonable optimization of the original complete pipeline.

Example after selecting and copying a unified checkpoint into an immutable
milestone (replace `/ABS/...` paths; do not benchmark a changing best.pt):

```bash
uv run python scripts/benchmark_strong_pipeline.py \
  --input /ABS/fundus.png --fov /ABS/fov.png --dataset drive \
  --vessel src/deepretina/pretrained/legacy-single-net-vessels.pt \
  --disc src/deepretina/pretrained/legacy-single-net-disc-0.pt \
  --universal src/deepretina/pretrained/legacy-single-net-universal.pt \
  --av src/deepretina/pretrained/legacy-av-v9.pt \
  --unified /ABS/immutable-selected-best.pt \
  --reference-report reports/disc-reference/original-calibrated-av-disc0-raw-val.json \
  --device cuda --warmup 1 --repeats 3 --output /ABS/new-latency-report.json
```

Check the bundled filenames with `uv run deepretina models`; filenames in a
historical checkout may differ. The script verifies checkpoint contents, not
filenames. Supply `--field-of-view-degrees` when needed. Provided FOV calibration
is still limited to its documented generic source branch; use suitable supported
images and report representative geometries separately. Undefined original AV
normalization (empty/full vessel predictions or constant RGB) raises an error
rather than silently timing a fallback network.

Times include CPU preprocessing, model inference and original-frame decoding;
exclude image IO, checkpoint loading and prediction-file export. CUDA is
synchronized, all samples and warmups are retained, and peak Torch allocation
is reported separately from persistent model allocation. This excludes memory
outside Torch. Report CPU thread count and other system load when comparing.
Unified inference uses checkpoint-configured tile/whole geometry, vessel gating
and cup containment. The original chain adds no cup containment. Input/FOV,
selected-reference and checkpoint hashes, model parameters, checkpoint bytes,
config and checkpoint epoch are recorded. Checkpoint/reference changes during
the measurement abort it. Timing reads no labels and makes no accuracy claim.

One image, an E8 screen, or a parameter count does not establish final improvement.
Repeat after final validation selection on representative raw image geometries,
then report timing alongside the independently evaluated quality comparison.
Nine focused CPU tests protect selected reference SHA/profile checks, native
dedicated heads, strict zero threshold, calibrated AV vessel input, nonfinite
failure and unified configured geometry/domain/containment.
