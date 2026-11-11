# Research-only native CUDA BF16 training

This branch starts from vascular source0b8762e. It prepares a future experiment;
no active Ada source, job, environment or current protocol was changed. No GPU
benchmark, speedup, memory saving or model-quality improvement is claimed.

`UnifiedConfig.mixed_precision` accepts exactly `"none"` (default) or `"bf16"`.
The new preset `configs/unified-v2-vascular2-bf16.toml` differs from vascular2 only
by this field. Existing TOML/JSON configuration routes accept it without new flags.
BF16 requires CUDA with native support (`including_emulation=False`) on the actual
selected device, checked before manifest/output processing. CPU or unsupported
CUDA BF16 training fails, even though BF16-trained float32 weights can be inferred
on CPU. No FP16 mode, GradScaler, optimizer, scheduler or sampler change is added.

Autocast covers only the training forward/loss block. Logits are cast back to
float32 for the masked segmentation/conditional-AV loss and reductions; model
parameters and optimizer states stay float32. Backward/update remain outside
that context. This changes numerical training behavior when explicitly enabled.
No gradient-equivalence or quality claim follows from finite loss alone.

Default none uses no autocast and no extra tensor/RNG operations. A functional
CPU run compares the pinned0b8762e original implementation against this default:
all saved parameters, optimizer/scheduler state, photo/annotation coverage and
serialized RNG states agree bitwise after two actual optimizer steps. New config
metadata and executable source hashes necessarily differ; checkpoint file bytes
and old-source resume compatibility are not claimed. Existing source/config resume
checks pin precision exactly; none↔bf16 is rejected. No old gate is relaxed.

For BF16-configured checkpoints, fresh epoch VAL and prediction explicitly disable
outer autocast; original none behavior is preserved. Prediction uses the existing
float32 arithmetic and threshold0.5. BF16 precision metadata is training metadata,
not permission to change frozen evaluation precision. Current fixed6b pool adapter
and source0b standalone contracts remain unchanged; future BF16 full runs need new
externally reviewed source/config/protocol pins and their own completion evidence.
Never rewrite an active run's config or retag its receipt.

## Deferred actual Ada benchmark, only after root confirms GPU is free

Do not launch this bench while protected vascular2 or another Ada training process
is active. It is a bounded TRAIN diagnostic, not a new full run or selection step.
Use the existing own Ada runtime, exact GPU UUID, threads20/20, workers2, batch4,
384 patches, compact UNet base12/depth4, vascular2 sampling and canonical cache18c.
Root must review the final new source/config and verify actual idle admission first.
Do not decode VAL/TEST labels or invoke epoch validation for this benchmark.

1. Record exact source/execution/training/config SHAs, runtime/Torch/CUDA versions,
   actual GPU UUID, math-library/Torch threads and BF16 native-support result.
   Do not change TF32/backend flags between variants. Retain inherited flags in
   evidence, including deterministic-algorithm settings and cuDNN benchmark=False.
2. Materialize 100 TRAIN batches from the same actual cache/manifest using one
   seeded sampler/order/observer/augmentation trace. Keep batch tensors in host
   memory, not a new data export. Pin the ordered photo IDs, annotation IDs and
   batch tensor byte digests; do not infer coverage from counts. This isolates
   compute first; separately time the same cached input pipeline if needed.
3. Construct one float32 initial model and optimizer, then deep-clone all weights,
   BatchNorm buffers, optimizer state and Python/NumPy/Torch CPU/CUDA RNG state.
   Both variants must start from exactly these same bytes and replay the exact
   batches; do not initialize sequentially without resetting RNG. This is a
   benchmark-only starting snapshot, not a supported pretrained lineage feature.
4. Warm up each precision for5–10 steps to initialize kernels/allocators. Restore
   the identical initial snapshot and RNG before measuring, keeping warmed kernels.
   Synchronize CUDA before and after every timed measured step; reset peak-memory
   stats immediately before each measured variant. Run30–100 steps (prefer100),
   repeat in reversed order to expose thermal/cache/order effects. Include forward,
   loss, backward and AdamW update in step time; report input-copy time separately.
5. Retain per-step loss and loss components, CUDA-synchronized seconds, samples/s,
   peak allocated/reserved bytes, parameter dtype, gradient dtype, all-finite
   weights/gradients/optimizer states, and gradient norms/nonzero counts by head.
   Compare first-step gradients on identical weights/batches: relative L2/cosine
   and max absolute differences, preserving zero gradients on unknown heads.
   Thereafter weights may diverge numerically; report rather than conceal that.
   Reject any nonfinite loss/gradient/state, missing supervised-head gradient or
   changed data/order/initial-state pin. BF16 uses no loss scaling.
6. Summarize median/p10/p90 step times, end-to-end time and peak memory for both
   orders. Keep any diagnostic snapshots named functionality/benchmark, outside
   model release and ordinary completed-run directories; do not promote them or
   call them full60 completion. A favorable compute bench only justifies a separately
   reviewed future full-budget recipe. Quality requires that completed run and
   fresh unchanged float32 VAL, with task/source results and full annotation proof.

CPU tests are admission/default-parity/operator/serialization checks, not an Ada
performance proxy. The CPU BF16 operator smoke exercises finite float32 gradients
and unknown-head zero gradients; production CPU BF16 training remains forbidden.

## Reusable benchmark entry point (prepared, not executed)

`scripts/benchmark_unified_bf16.py` implements that core diagnostic. It is dry by
rejection unless `--execute` is explicit. **Root must first use the existing private
exclusive-GPU admission/owned process supervision; this public script does not
replace that guard, reserve the GPU, stop another process or launch a worker.**
No invocation is authorized while the current Ada run is active. Its native-Ada,
single-visible-device, BF16 capability and expected GPU UUID checks happen before
input reading/report creation. Actual clean Git commit and executable/training
hashes are recorded. CPU execution is explicitly forbidden for timing.

After a separately reviewed idle launch, the command shape is:

```sh
CUDA_VISIBLE_DEVICES=GPU-d0675605-e6c2-1c85-9799-08b279aab0ff \
OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 PYTHONPATH=src \
/var/home/tom/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python \
  scripts/benchmark_unified_bf16.py --execute \
  --parent /absolute/immutable-control-e060/last.pt \
  --manifest /absolute/canonical/manifest.json --cache /absolute/cache \
  --output /absolute/new-benchmark/report.json --batches 32 --warmup 5 --max-seconds 900
```

Replace paths only after root reviews them. Parent SHA is hard-pinned to actual
terminal60 `10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2`;
loading uses that same verified byte snapshot with safe CPU torch.load. Architecture
is strict-loaded into the vascular2 model. Model/Adam states must be finite and
float32. **The parent's Adam state, including its terminal LR and moments, is
preserved exactly**, not silently changed to fresh optimizer state or preset LR;
actual parameter groups are included beside the performance configuration. This
is numerical/compute comparison only, not a supported warm-start training recipe.

The script preloads32–64 augmented TRAIN batches once through `VerifiedCache18c`
and the existing train dataset/sampler. Only TRAIN labels are decoded; canonical
verification may read encoded bytes from other splits. Ordered photo IDs,
contributing annotation IDs and exact tensor hashes are retained. Shared batch
tensors are capped at2GiB (dataset scratch/cache and interpreter RAM are additional).
No DataLoader workers run during timed legs. Both Torch thread pools are20.
With coverage-first and at most256 photos, these batches are the shuffled initial
coverage pass, not a representative complete8192-sample weighted epoch. Preload
uses one CPU producer (workers0); it is not a claim to reproduce the live two-worker
augmentation stream or measure its pipeline throughput.

Leg order is FP32/BF16/BF16/FP32. Each leg warms5–10 steps, then restores exact
parent parameters, BN buffers, Adam state and Python/NumPy/Torch CPU/CUDA RNG.
A first diagnostic is performed and undone with another full reset before measured
steps. Same-start state hashes must equal the immutable CPU initial snapshot. Last
diagnostics run outside measured bulk. Loss/gradient/state finiteness and nonzero
global gradients are required; first-gradient differences are reported honestly.
Only one GPU model/optimizer exists; variant models are never saved.

H2D and forward/loss/backward/update durations are CUDA-synchronized separately.
First/last diagnostics, evidence hashes and preload are excluded from core timing.
Peak allocated memory and extra working allocation are reported; reserved memory
also reflects warmup/allocator history and should not be treated as independent
cold-process peaks. Per-leg seconds/losses, throughput and p10/median/p90 are
retained. The900s default /1200s maximum budget is checked between preload batches,
warmup and measured steps; root's existing owned supervisor must enforce an outer
hard timeout for a hung device call. A failed admission or diagnostic produces no
successful report. No final checkpoint, selection, completion QA, VAL or TEST score
is generated, and core timing is not full-epoch throughput.
