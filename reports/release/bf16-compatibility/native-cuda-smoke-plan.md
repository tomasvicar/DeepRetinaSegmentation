# Prepared native CUDA BF16 check — not executed

This harness is ready for an owner-approved idle Ada GPU. **No GPU action was
performed during preparation.** Weight4 training currently owns that GPU; do not
run this command until root independently confirms it has finished and the
physical GPU is free. This check does not submit a Meta job or access any dataset,
GT, TEST labels or existing model checkpoint.

The product source must match commit `903c929` and reference source must match
`72fb321` throughout `src/`. Descendant commits containing only this harness/docs
are accepted; altered product source is rejected. No `AGENTS.md` was present in
this worktree or its filesystem ancestors when preparing this check.

Each source runs in a separate fresh Python process with the same seed 42, CUDA
visibility, one CPU thread, deterministic kernels and TF32 disabled. A small
synthetic batch (2 × 3 × 32 × 32; compact width 8, depth 3) exercises vessel,
conditional artery/vein, disc and cup losses. The real production precision
context must produce BF16 training logits, FP32 loss and finite FP32 gradients,
parameters and AdamW states after one step. At least one parameter must change;
all optimizer step counters must be one. There is no mocked admission or CPU AMP
emulation in this GPU check.

Prediction uses the production helper inside an enclosing CUDA BF16 context;
its actual forwards and returned probabilities must remain FP32. Separate-source
model states (including BatchNorm buffers), optimizer state, loss, FP32 prediction
and final CUDA RNG state must be **bitwise equal**. Any mismatch is failure, with
no tolerance-based success or automatic retry. This is a bounded operator/parity
check, not training quality, model completion or general throughput evidence.

## Deferred command

Stage this committed harness and the unchanged product source on Ada first. Root
verified the detached Ada reference checkout below at 09:49:26 UTC: HEAD 72fb321,
execution 347f78, training 3725ad32, tracked clean. Use the existing environment; do not run
`uv sync` on an active environment. After root's idle-device check, replace the
physical UUID with the independently observed Ada UUID and use a fresh output:

```bash
PORT_ROOT=/var/home/tom/code/DeepRetinaSegmentation_bf16_compat
REFERENCE_ROOT=/var/home/tom/code/DeepRetinaSegmentation_bf16_run_72fb321
CUDA_UUID=GPU_REPLACE_WITH_ACTUAL_ADA_UUID
OUTPUT=/var/home/tom/code/DeepRetinaSegmentation_model_backups/bf16-port-native-cuda-smoke

timeout --signal=TERM --kill-after=10s 90s \
  taskset -c 8 nice -n 15 \
  env CUDA_VISIBLE_DEVICES="$CUDA_UUID" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 \
  /var/home/tom/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python \
  "$PORT_ROOT/scripts/smoke_native_bf16.py" --execute \
      --port-source "$PORT_ROOT/src" --reference-source "$REFERENCE_ROOT/src" \
      --expected-gpu-uuid "$CUDA_UUID" --output "$OUTPUT"
printf 'numeric_exit=%s\n' "$?"
```

The outer 90-second timeout owns this invocation; each fresh worker additionally
has a 35-second limit. Keep the complete stdout/stderr log and numeric exit code;
a timeout or missing final verification is incomplete/failure, never PASS.
No common SSH master, training process or unrelated GPU actor is stopped.

Output is fresh-only: `reference.pt`, `port.pt` are synthetic CPU state snapshots,
not deployable models; each has an adjacent runtime JSON. `verification.json`
records actual physical GPU UUID, source hashes, elapsed wall time and each exact
parity result. Worker reports retain CUDA/PyTorch versions, compiled architectures,
actual capability and measured peak allocated CUDA memory. Publish only the small
verification/runtime metadata, not synthetic tensor artifacts. An unsupported
binary/device may pass native-BF16 feature detection but fail these actual kernels;
that is a failed smoke, not a reason to modify the runtime or borrow another GPU.

Preparation verification: three CPU-only harness tests pass, checking that missing
`--execute` cannot dispatch a worker, optimizer/dtype drift is detected, and empty,
nonfinite or reduced-precision master tensors fail. Ruff passes. Actual native
CUDA training-step validation of this product port remains **PENDING**.
