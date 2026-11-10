# Prepared terminal60 verifier — not executed on real checkpoints

`scripts/verify_control_e060.py` is a separate verifier for the original Ada
control run, not a migrated screen verifier. It pins source `3157bb9`, execution
`8e0481ff…`, training source `dd69012a…`, plan `6be48b57…`, manifest `529edba7…`,
canonical content `61032be2…` and derived cache `4e39e40d…`. It refuses any live
PID 1257093 and requires terminal last epoch60 and `completed_budget=true`.
The verifier has not opened or copied any real checkpoint during preparation.
Neither immutable3157 training nor frozen8c evaluation source was modified.

Checks include exactly contiguous CSV epochs1–60 with finite metrics/timings,
2048 steps per epoch, the planned cosine LR sequence, the first strict maximum
score selecting best.pt, exact plan/config/source/cache identities, finite best
and last weights and validation metrics, and finite resumable last optimizer/
training state. Optimizer step counters must be present and their maximum must
equal 122880, independently of the top-level steps field. The scheduler must
retain `T_max=60,last_epoch=60`. All Torch,
Python, NumPy, sampler, augmentation and CUDA RNG keys must exist; CUDA and core
RNG values must be present. Parent augmentation state may legitimately be null
with two worker-local generators. Best.pt intentionally stores no optimizer.

It requires 122880 optimizer steps, 491520 delivered supervised samples, all2221
actual eligible train photo IDs, minimum cumulative60 visits per photo, exact
source inventories/counts and no manifest patient/source or photo cross-split
leakage. Minimum cumulative60 visits alone is not a reconstruction of per-epoch
visits; the coverage-first implementation is independently pinned by source.
No pixel arrays are decoded, no model is instantiated and no inference occurs.

Canonical schema2 content is rehashed using the pinned3157 helper. Cache scope is
explicitly **metadata identity**, not another8GB cache payload audit: fixed
index SHA256 `b624df6a…`, `_SUCCESS`, canonical identity, and checkpoint/run/plan
cache fields must agree. Existing complete cache byte verification remains
separate prior evidence. The resulting report records `cache_payload_rehashed=false`.

After all checks, last/best and five sidecars plus search provenance, frozen
plan and QA summary are independently copied from the exact checked byte
snapshots into a fresh milestone. Source file hashes are checked again before
copying, and destination byte hashes must agree. Any existing milestone or
report is refused. An interrupted partial milestone is preserved for inspection;
use a separately reviewed fresh destination rather than overwrite it.

## Tiny CPU fixtures

2026-11-10: **19 tests passed in0.67s** on Ada, CPU1/nice10/CUDA hidden, existing
uv-frozen runtime Torch2.6.0+cu124. Synthetic two-photo metadata and one-element
tensor fixtures exercise tied best epochs, missing epochs, wrong LR/steps,
nonfinite best/optimizer state, stale optimizer counters, absent CUDA RNG, altered scheduler/cache/source/
plan, incomplete coverage, split leakage, inconsistent sidecars and refusal to
overwrite milestone copies. No production data/checkpoints were loaded.

## Command for parent review after actual completion

This command is documented only and has **not** been executed. The parent must
first verify actual process termination. It does not submit or resume training.
The Python executable must be the existing uv-frozen runtime and CPU limits must
be established before Torch import.

```bash
SOURCE="$HOME/code/DeepRetinaSegmentation_unified_run_3157bb9"
TRIALS="$HOME/code/DeepRetinaSegmentation_unified_trials_3157bb9"
CACHE="$HOME/code/DeepRetinaSegmentation_derived_storage/full/61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e/v1-train-val"
VERIFIER=/path/to/reviewed/scripts/verify_control_e060.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$SOURCE/src" nice -n10 "$SOURCE/.venv/bin/python" "$VERIFIER" \
  --source "$SOURCE" \
  --run "$TRIALS/runs/lr_0003-control-cached-seed42" \
  --plan /path/to/exact/plan-av-residual-cached-approved-3157.json \
  --manifest "$HOME/code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json" \
  --cache "$CACHE" \
  --milestone "$TRIALS/milestones/lr_0003-control-cached-seed42/epoch-060" \
  --report "$TRIALS/lr0003-control-cached-e060-verified.json"
```

All path arguments are host-local; byte pins, rather than mutable path names,
govern identity. The report does not claim test performance, clinical utility,
historical blindness or hardware attestation.
