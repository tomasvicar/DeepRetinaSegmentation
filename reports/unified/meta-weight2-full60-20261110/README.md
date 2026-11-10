# Proposed standalone Meta weight2 full60 follow-up

**Proposal only. No submission, upload, installation, training or forward.**
Candidate: `av_weight_2_lr0003_cached_workers8`. Source remains immutable3157,
execution8e0481ff, trainingdd69012a, canonical61032be2 and cache4e39e40d.
The runner-compatible JSON contains exactly one candidate,60 planned epochs and
an E1 screening boundary. It inherits no old approval timestamp or authorization.

The matched old/GTX1060 E8 comparison gave weight2 the highest point macro score:
0.714549 versus0.708936. However, primary delta+0.005613 has paired95% interval
[-0.001368,+0.011723]. Vessel pooled Dice decreases by0.008120 with interval
[-0.010449,-0.005621]. These exploratory one-seed validation results justify
considering a full-budget follow-up, not declaring a better all-task model.
Conditional AV gains are point-only because per-photo ungated counts are absent.
All2000-draw bootstrap endpoints and multiplicity caveats are preserved in
[`paired-avweight-e008/result.json`](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/cfcc0ab/reports/unified/paired-avweight-e008/result.json).

## Configuration and confounds

Against the approved Ada control6be48 plan, only training-config fields
`av_weight`1→2 and `workers`2→8 change. Architecture remains compact U-Net,
base12/depth4, patch384, batch4, LR0.0003, mild augmentation, seed42,
8192 samples/epoch, coverage-first,60-epoch cosine horizon and identical losses
apart from AV weighting. This is a **standalone weight2 follow-up**, not a paired
architecture test or a causal AV-weight comparison against the Ada run.

The planned resource request is16CPU/32GB RAM/20GB scratch/one GPU with minimum
10GB VRAM and `gpu_cap=sm_80`, maximum4h, no array. Runtime uses Torch intra8/
interop1 while OMP/MKL/OpenBLAS each remain1 and `PYTHONHASHSEED=42`. Torch thread
limits must be explicitly set before executing the unchanged3157 runner; environment
variables alone do not set intra8. Exactly one scheduler-assigned GPU is visible,
and its observed UUID/driver/runtime must be recorded throughout.

Workers8 improve the opportunity to feed the GPU but change worker assignment,
crop/augmentation streams and scheduling despite the same seed. Source3157 uses
the default non-persistent DataLoader workers, recreated each iterator. Hardware,
floating-point runtime and the planned E1 resume boundary also limit cross-run
comparison; no bitwise-equivalence or causal quality claim is made. The source
and cached data themselves remain unchanged.

## Feasibility and E1 guard

Storage is plausible, not permission to skip staging checks: verified extracted
cache allocation is8.172GB, canonical budget3.5GB, source0.1GB, outputs1GB and
runtime temporary2GB, plus2GB free reserve, totaling16.77GB under a conservative
20GB limit. Stream archives from persistent storage directly into scratch.
Never retain the8.159GB cache tar alongside its extracted payload. Bind the pinned
SIF read-only from persistent storage; no scratch-side image build or install.
Check actual free bytes/inodes and pinned cache/source/canonical identities.

32GB host RAM is proposed for eight worker caches/prefetch and the main process;
actual peak is unmeasured. Prior base12/batch4 runs fit6GB-class GPUs, supporting
the minimum10GB request, but exact-device E1 memory still must be observed. No
runtime or allocation availability is established by this proposal.

Run the unchanged runner to `--stop-after-epoch 1`. After finite checkpoint/state,
all2221 coverage and exact pin/config QA, preserve an E1 milestone. Measure from
GPU-allocation start, including staging and E1 QA. The controller's predeclared
guard is:

```
projected_seconds = elapsed_after_E1_QA
                  + 1.2 * 59 * (actual_E1_train_seconds + actual_E1_validation_seconds)
                  + 300
continue only if projected_seconds <= 14400
```

The20% remaining margin also needs to absorb resume preflight/checkpointing and
later timing variance; record initial/resume overhead separately. The300s reserve
is for terminal QA/export. If too slow, OOM, source/cache/pin drift or QA fails,
export E1 checkpoints/metrics/projection and halt; do not silently reduce batch,
workers or epoch budget, resubmit, or call an incomplete run full60. Otherwise
the reviewed private controller may resume the **same output** `--resume --final`
with unchanged config, source, cache, scheduler and allocated GPU. This proposal
does not itself authorize that submission or controller execution.

The private wrapper/review remains separate. Public plan uses portable `/data`,
`/cache` and `/workspace` mounts and contains no account/storage credentials. No TEST
evaluation or selection freeze is included; completed models still require fresh
VAL evidence and reviewed final selection.
