# Vascular sampling 2: fresh VAL readiness

**The existing frozen producer works without relaxation. No new evaluator is
needed. No VAL/model/GT execution or live-run change was performed for this
readiness check.** Actual evaluation remains conditional on root review and a
successful original terminal plus strict standalone completion verification.

The running experiment changes `vascular_sampling_weight` to `2.0`; vessel loss
and positive BCE weights remain `1.0`. Do not describe it as a vessel-loss-2 run.

## Why the existing CLI is compatible

Frozen `scripts/unified_validation.py` delegates to `produce_validation_evidence`.
The producer takes an original safe checkpoint snapshot, validates independent
scratch/full-data lineage, all five supervised heads, finite float32 model state,
normalization and the planned 60-epoch config, then checks canonical identities.
It evaluates **VAL only** and records actual source/runtime/config/protocol and
per-photo counts with byte/source/data checks before and after. It does not
require a pool receipt, coordinator decision, HPO search marker or pool QA kind.

Ordinary standalone best/last checkpoints retain their genuine training-selection
score, so there is no BN-derived-checkpoint compatibility issue. That original
training score is kept distinct from fresh measured VAL scores. Do not fabricate
or copy scores from a different artifact.

## Immutable identities to preserve

| Artifact | Exact identity |
|---|---|
| Running source commit | `0b8762e667c997a2dfd129dc65988c1f26f6b2ad` |
| Execution code | `3ffe949e787045546a9470750b1aedc27c8ae9bee3603db568f719e4b407f24e` |
| Training source | `85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f` |
| Original reviewed standalone protocol | `f4ca4a1aaa7f2a01ad3f6dc818e5a61087fd299cd9ad436941b938bbfd76ec8c` |
| Strict standalone verifier | `4b568b7e5e66229e378fd8712000af90a05190364563bad6311687e419eb56fe` |
| Frozen VAL CLI | `824bab3412676a8a918d3efc8c863b7b85803e653ef7cf7c957c375fd1f17fbf` |
| Frozen VAL producer | `447c55c2b61e68652c11077b395b7a59ba9deacc747d773c8baef4154e3a1bea` |
| Canonical manifest | `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767` |
| Canonical content | `00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52` |

Private original protocol/profile/process/source-path references already exist
in the operational repository; no private paths/endpoints are copied here.
The protocol fixes 60 epochs, 8192 samples per epoch, 2529 TRAIN photos,
5700 eligible annotation records, 21 sources and 770 VAL photos. Its inference
recipe is threshold 0.5, tile 512, overlap 64.

## Execution sequence after real completion

1. Preserve the actual original supervisor terminal and run artifacts. Verify
   original protocol/terminal/verifier byte pins using the strict standalone
   verifier from `1a50618`; require `kind=standalone_unified_completion_qa`.
   Do not rename or adapt it into `trusted_pool_completion_qa`, a pool receipt or
   a coordinator decision. No retrospective terminal fabrication is acceptable.
2. Make an independent immutable checkpoint/sidecar snapshot with exact hashes
   matched to that original QA. Preserve **both** `best.pt` and `last.pt` hashes,
   their own epochs and coverage. Do not read a mutable live `best.pt` while
   training continues or relabel last as best.
3. Choose the role explicitly. For `best`, use its exact QA hash/epoch and disclose
   `best_checkpoint_coverage` and `selected_best_full_annotations`. Earlier best
   can have fewer than 5700 contributed annotations even when the run completed.
   For `fixed_terminal60`, use the actual last checkpoint hash, epoch 60 and
   `last_checkpoint_coverage`. Completion of the run does not transfer terminal
   coverage to an earlier checkpoint.
4. Under the existing reviewed owned-process-group launcher, run the command
   below once on the original frozen source. CPU 4, affinity `8,10,12,14`, nice 15,
   idle I/O and CUDA hidden; maximum 1200 seconds plus at most 60 seconds cleanup.
   A failed/timed-out evaluator is not silently retried. No new user permission
   flow is introduced; operational root review and actual prerequisites apply.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH="$VASCULAR_SOURCE/src" \
  nice -n 15 ionice -c 3 taskset -c 8,10,12,14 "$VASCULAR_PYTHON" \
  "$VASCULAR_SOURCE/scripts/unified_validation.py" \
  --checkpoint "$VASCULAR_IMMUTABLE_CHECKPOINT" \
  --manifest "$VASCULAR_CANONICAL_MANIFEST" \
  --output "$VASCULAR_NEW_VAL_REPORT" \
  --device cpu --cpu-threads 4 --threshold 0.5 --tile-size 512 --overlap 64
```

These variables identify the exact original source, existing active environment,
reviewed immutable best OR last snapshot, canonical manifest and a new output
outside source/run/corpus. They are placeholders, not commands executed here.
The CLI has no split/test flag. It has no completion-QA input: completion/role
binding belongs in the separate operational evidence, preserving typed QA bytes.
Do not edit the frozen producer to inject a new kind or fake completion history.

## Correct postcheck: metrics are nested

The existing producer emits `kind=unified_validation_evidence`, with
`report['metrics']['photos']`, `report['metrics']['per_photo']` and
`report['checkpoint']`. A top-level `report['photos']` check is incorrect.
`reports/vascular2-val-readiness/postcheck_example.py` demonstrates a metadata-only
postcheck for this exact typed standalone QA and explicit checkpoint role. It is
not a completion verifier, launch authorization, final-selection gate or substitute
for full source/config/data/QA byte checks. Failed postchecks must preserve the
producer report and original wrapper failure, not rerun NN inference or invent an
original terminal. Generic paired comparisons may explicitly unwrap metric views
while retaining full original wrapper hashes.

Existing completion/selection/TEST routes that only accept older typed families
or source 6b are **not** declared compatible by this readiness result. Do not retag
standalone evidence to pass them. Fresh VAL can be produced normally; a later
review can decide the appropriate comparison/selection path without changing
historical artifacts. This task adds no such adapter.

## Bounded validation performed

Seven metadata-only tests pass in 0.04 seconds. The exact byte-pinned frozen CLI
was exercised with all ML imports stubbed: it delegates the selected checkpoint,
CPU/threshold arguments without any QA/decision dependency. Synthetic nested
770-photo reports test best versus terminal role/coverage, wrong hash, pool-kind,
partial and flat-shape rejection. No Torch model, GT array, VAL NN, TEST, training,
source checkpoint load or real completion receipt was used. Ruff/format pass.
