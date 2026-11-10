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
  --original-disc-variant disc0/raw \
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
and cup containment. The original chain defaults to native cup decoding; optional containment must be selected explicitly as described below. Input/FOV,
selected-reference and checkpoint hashes, model parameters, checkpoint bytes,
config and checkpoint epoch are recorded. Checkpoint/reference changes during
the measurement abort it. Timing reads no labels and makes no accuracy claim.

If VAL selected a threshold other than0.5, supply that exact frozen value with
`--unified-threshold`; the measured mask decoding and report retain it. Default
is0.5. This changes the common vessel/disc/cup threshold and A/V vessel gate,
not the original model thresholds or conditional A/V class decision. Benchmarking
does not select or tune a threshold; complete VAL selection first.

One image, an E8 screen, or a parameter count does not establish final improvement.
Repeat after final validation selection on representative raw image geometries,
then report timing alongside the independently evaluated quality comparison.
Eighteen focused CPU tests protect selected reference SHA/profile checks, native
dedicated heads, strict zero threshold, calibrated AV vessel input, nonfinite
failure and unified configured geometry/domain/containment, including selected
threshold propagation into timed decoding and report output. All18 passed on
Ada CPU1/nice10 with CUDA hidden (0.85s, 2026-11-10); Ruff/format passed.

Input provenance uses one byte snapshot of each RGB/FOV file before decoding;
reported hashes identify those exact timed inputs even if a path changes later.
Paletted single-channel FOVs are converted to luminance before thresholding,
matching public prediction: palette index zero can legitimately represent white
foreground. Color visualizations remain rejected. No tiling, photometric profile,
model choice or measurement scope is changed by these input-reader fixes.


## Explicit original cup policy

`--original-cup-policy native` is the default. `--original-disc-variant disc0/raw|disc1/raw` is required explicitly and must match both the reference variant and actual dedicated disc checkpoint SHA. To time the stronger optional postprocessing reference, add `--original-cup-policy inside-disc` and pass the corresponding complete paired quality report, for example `reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json` from the completed b5804c7 study. Use disc1 and its corresponding report for that separate alternative. A native quality report cannot accompany clipped timed decoding, or vice versa; mismatched policy/context/disc variants are rejected.

The exact optional rule is `(universal_raw_cup_logit > 0) & (same_forward_selected_raw_disc_logit > 0)` in the shared prepared frame, followed by nearest restoration. No probability threshold, morphology, GT clipping or universal disc substitute is added. Every other original head retains its decoding; the original weights/defaults are untouched. The benchmark records `original.cup_policy`, selected disc variant, checkpoint hashes and quality-reference hash, and executes that same policy inside every timed warmup/repetition. It reads no GT and makes no new quality claim.

The completed paired VAL study covered all612 cup-supported photos and reported disc0-reference macro0.654498→0.667032, with80,891 true-positive cup pixels lost and1,450,920 false positives removed. Those changes describe an exploratory postprocessing tradeoff, not uniform improvement: consult source-level counts and the separately logged native numerical drift in the study report. Do not combine clipped quality with a native-only speed claim, or conceal reduced sensitivity. The final selected pipeline policy must be frozen before independent evaluation.

Focused synthetic tests check strict-zero semantics, canonical-frame clipping before restoration, unchanged other heads/logits, both dedicated disc variants, policy/reference rejection, and policy propagation through timed callbacks and report metadata. They do not run actual networks or read GT.
