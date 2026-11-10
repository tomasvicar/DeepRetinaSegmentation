# Width16 cached E1 paired gate

Post-hoc analysis only: this file lives under reports and is not execution code
for training/search fingerprints. No new training, model inference, image reads
or GPU use is performed by the tool. Run only after both completed E1 checkpoints
and their CSV rows are available. Source87 and its plan/checkpoints remain intact;
new cached training starts fresh under its separate immutable source/plan.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run python reports/unified/cache-width16/compare_checkpoint_states.py \
  --old /ABS/ORIGINAL/last.pt --new /ABS/FRESH_CACHED/last.pt \
  --old-csv /ABS/ORIGINAL/metrics.csv --new-csv /ABS/FRESH_CACHED/metrics.csv \
  --epoch 1 --output /ABS/NEW_REPORT.json
```

The output path must be new; early guard plus exclusive creation preserve existing
reports. Checkpoints are read once into hashed snapshots and loaded on CPU using
weights_only=True. Every model parameter/buffer, optimizer/scheduler/RNG/coverage
leaf, config except device, epoch/steps/canonical data identity, and other shared
checkpoint metadata compares strictly. Missing keys and nonfinite values fail
preservation and remain explicit in the report. Unequal tensors retain dtype,
shape, unequal-element count and maximum finite absolute difference, together
with nonfinite counts. TensorEqual is a value/dtype/shape comparison, not a claim
about identical serialized files or universal CUDA determinism.

Source fingerprint and cache identity are deliberately different and retained
verbatim as provenance; cache canonical identity must match the checkpoint's
canonical data pin. New cached checkpoints require a pinned cache with fallback
false. Validation comes from each checkpoint's embedded metrics for that epoch,
not a potentially newer best-model sidecar. Header fields compare strictly;
per-photo matching uses photo IDs and tp/fp/fn/valid_pixels for each channel. The
new extra per-photo av_conditional_counts and derived per-photo ratios are
explicitly excluded from this old/new comparison.

CSV epoch1 training loss/steps compare strictly, while train/validation seconds
are reported separately as measurements. CSV timings exclude startup, support
scan and cache verification; they are not end-to-end timings. Optional paired
--old-wall-seconds and --new-wall-seconds record separately measured external
wall time and its speedup. One passing pair does not prove universal bitwise
preservation; CUDA warn_only permits some nondeterministic kernels. A mismatch
is reported rather than relabeled as equivalence, and is not alone proof that
storage caused it.

Synthetic CPU tests: tests/test_cache_checkpoint_comparison.py. Actual paired E1
gate is pending fresh cached E1 completion; no real model/data result is claimed.
