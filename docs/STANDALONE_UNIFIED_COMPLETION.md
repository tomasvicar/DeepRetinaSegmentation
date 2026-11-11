# Standalone full60 unified completion verification

`scripts/verify_unified_completion.py` verifies a completed, root-owned standalone
`train-unified` run on its actual Git source host (the intended deployment is Ada).
It produces `kind=standalone_unified_completion_qa`, compatible with the flat
completion fields consumed by `compare_completed_val.verify_completion`.
It is not a pool receipt, an Optuna decision, a supervisor or a selection artifact.
Existing6b pool adapters and VAL/TEST selection gates are unchanged.

The verifier performs safe CPU checkpoint loading, finite/shape checks and canonical
encoded-byte hashing/metadata checks. It never calls a model forward, GT decoder,
validation metric calculation or TEST evaluator; CUDA initialization is rejected.
Canonical encoded bytes, including heldout files, are hashed for identity without
interpreting their arrays. Cache identity is bound to reviewed protocol/run/
checkpoints; the cache payload is not re-audited here.

## External review and original supervisor evidence

Finalize and freeze the future training source commit first. Only then materialize
and externally review the protocol file, outside that source checkout. It must
contain full commit/execution/training hashes and the exact complete
`asdict(UnifiedConfig(...))` configuration, including defaults and device. Compute
execution bytes using `scripts/unified_hpo.py::code_digest`; training bytes use
`unified._training_source_sha256`. Protocol bytes are pinned externally with SHA256.
No protocol embeds its own hash, and the verifier creates neither protocol nor
terminal metadata. This avoids self-referential source/metadata hashing.

Future protocol schema (field description, not an approved protocol artifact):

| Field | Required value |
| --- | --- |
| schema_version, kind |1, `reviewed_standalone_unified_protocol` |
| source | `{root: ABSOLUTE_GIT_ROOT, commit: FULL40, execution_code_sha256: FULL64, training_source_sha256: FULL64}` |
| run | exact absolute owned run directory |
| config | exact full serialized config; independent full_data,60epochs,8192samples, coverage_first, no teacher/max_steps |
| budget | `{epochs:60,epoch_samples:8192,train_photos:2529,train_annotation_records:5700,dataset_count:21}` |
| data | manifest17a227…5a767, prepared_content00d9…e52, geometry_policy2591…301e3d, explicit derived_cache |
| derived_cache within data | null, or schema1/fingerprint18c3…766e/canonical00d9/allow_canonical_fallback=false |
| inference | threshold0.5, tile_size/overlap exactly matching reviewed config |

After the bounded supervisor has observed the actual child finish, the supervisor
must independently emit its **original** terminal record. Root reviews and pins
those exact bytes. Do not construct a terminal afterwards from checkpoints alone.
Existing pool/waiter terminals cannot be renamed or adapted into this family.
Required future terminal schema:

- `schema_version=1`, `kind=bounded_unified_supervisor_terminal`,
  `status=completed`, integer `returncode=0`, `timed_out=false`,
  `test_evaluation_performed=false`, `completed_epochs=60`.
- `protocol_sha256`: externally reviewed protocol's original file-byte SHA.
- `source`, `run`, `budget` and `data`: same values as the reviewed protocol.
- `config_sha256`: SHA256 of compact sorted JSON of the full config, **no final
  newline**. This is the full config, including device.
- `steps` and `delivered_samples`: actual terminal optimizer/sample counts.
- `artifact_sha256`: exact final original byte hashes for `best.pt`, `last.pt`,
  `metrics.csv`, `training_summary.json`, `run.json`, `coverage.json`,
  `validation.json`. Additional supervisor process/timing provenance may remain.

External SHA review authenticates which supplied bytes are trusted by the operator;
it is not cryptographic proof of process truth or reviewer identity. Preserve the
actual supervisor logs/process evidence alongside the original terminal.

## Execute on the actual source host

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/frozen/new-source/src \
  nice -n 10 taskset -c ALLOWED_CPU python /reviewed/verify_unified_completion.py \
  --source /frozen/new-source --run /owned/completed-run \
  --manifest /canonical/corpus-v2/manifest.json \
  --protocol /reviewed/protocol.json \
  --trusted-protocol-sha256 ORIGINAL_PROTOCOL_BYTES_SHA256 \
  --terminal /supervisor/original-terminal.json \
  --trusted-terminal-sha256 ORIGINAL_TERMINAL_BYTES_SHA256 \
  --output /new/standalone-completion-qa.json
```

Use the reviewed source's existing environment. No installation, SIF/fake-Git
fallback or remote orchestration is implemented. Git HEAD, tracked cleanliness and
actual executable bytes must match; the imported `selection` module must reside
in that source. Torch uses CPU with one intra/inter-op thread. Outputs must be new,
outside source/run/canonical inputs, and existing or dangling-symlink outputs fail.

## Strict completion checks and scope

The implementation reuses `selection.checkpoint_snapshot`, `config_identity`,
`data_identity` and `completion_evidence`. It adds standalone stricter requirements:
all2529 actual TRAIN photo IDs, min60 visits each and491520 total delivered samples;
all5700 eligible TRAIN annotation IDs with positive actual contributions; all21
source inventories; exact full config/data/cache/policy/run/sidecar identities;
lastE60, contiguous60CSV epochs with exact per-epoch steps/LR, first strict best
maximum, finite float32 best/last weights, and finite last optimizer/RNG state.
AdamW parameter IDs/moment shapes/dtypes/step counters must match the safely loaded
model, with completed cosine60/minLR. No partial-coverage tolerance is inherited
from a looser generic helper. Unsupported optimizer layouts fail conservatively.

The output preserves original protocol/terminal text and original byte SHAs, all
artifact hashes, Git/source pins, typed completion and full TRAIN counts. It makes
no quality, full blindness, selection or TEST authorization claim. A real E60
completion is not produced by this code change; positive tests use synthetic
metadata/tensors and deliberately invalid encoded image/NPZ bytes, with model
forward, CUDA and `np.load` forbidden. The stripped actual E27 pool receipt is a
negative fixture only. Evidence is under `reports/standalone-unified-completion/`.

Verification: **96 bounded CPU tests pass** (31 new verifier tests plus existing
selection/flat-comparison tests),15.56seconds; Ruff passes. No full suite was run.
Tests cover real safe checkpoint loading, cached/canonical identity, E27/pool
rejection, source Git/byte pins, original supervisor SHA/artifact binding, missing
annotation IDs, stale optimizer state, moment shape, malformed schedule/config,
nonfinite weights and preservation of existing/dangling-symlink outputs.
