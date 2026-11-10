# Explicit reviewed unified held-out execution

`unified_test.py` is a separate stage2 route. It evaluates only the test partition
of a complete frozen selection checked by the existing stage1 metadata gate.
The general `deepretina evaluate --split test` route remains blocked.

The stage1 artifact and receipt retain `test_evaluation_enabled: false`: they
record metadata verification, not an execution permit. This separate reviewed
script requires the explicit `--execute-reviewed-heldout` request. It does not
silently flip or reinterpret that boundary flag. The output records both the
metadata-only receipt and the separate stage2 request. This request is a command
within the authorized workflow, not an additional human approval mechanism or
an authenticated reviewer identity.

## Freeze before evaluation

The final source checkout must include this dispatcher **before** producing fresh
VAL evidence and creating/reviewing its selection artifact. Adding/changing this
module afterwards changes the evaluation-source identity and invalidates that
evidence. Never retrofit provenance onto thin historical VAL reports.

1. Freeze the final source/runtime and chosen protocol.
2. Generate [fresh VAL evidence](FRESH_VALIDATION.md) for the chosen checkpoint.
3. Create and review a [completed60 selection artifact](HELDOUT_SELECTION.md).
   Retain its emitted SHA separately as the external reviewed digest.
4. Only then explicitly invoke stage2 with those exact inputs.

No actual trained-model selection artifact, real test result or GPU evaluation
was created while implementing this route. Tests use tiny synthetic models and
two synthetic all-five-head test photos, plus fresh synthetic VAL evidence.

```bash
uv run python scripts/unified_test.py \
  --execute-reviewed-heldout \
  --selection review/selection.json --expected-selection-sha256 REVIEWED_SHA \
  --checkpoint RUN/best.pt --run-dir RUN \
  --manifest DATA/manifest.json --validation-evidence reports/fresh-selected-val.json \
  --output reports/final-heldout.json --device cpu --cpu-threads 1
```

`--checkpoint` remains explicit so the selected earlier best snapshot is
unambiguous. `--run-dir` supplies `last.pt`, `metrics.csv` and
`training_summary.json`; their explicit options override those defaults. A fully
explicit call can omit `--run-dir`. Artifact, external digest, canonical manifest,
fresh VAL evidence and output are always explicit. A scheduled CUDA evaluation
may use `--device cuda:0`; no CUDA initialization is allowed before gate success.

There are no threshold, tile-size, overlap or split overrides on this route.
The externally pinned artifact's actual frozen protocol is verified by the gate
and reconstructed exactly for inference, including valid nondefault protocol
choices. Whole/tiled geometry, float32/no-autocast, five-head decoding, FOV and
GT-source policy remain the shared unified evaluator's implementation.

## Boundaries and recorded evidence

Before any held-out GT decoder, model forward or CUDA initialization, the route
rejects an existing output and verifies the actual artifact, checkpoint, complete
training run, fresh VAL evidence, source/runtime and canonical identities. It
loads only `CheckedSelection.checkpoint_snapshot` via BytesIO with
`weights_only=True`; the checkpoint pathname is never reopened for execution.
Replacing that pathname after checking cannot change evaluated weights.

An operation-scoped canonical/source lease begins before the metadata gate and
spans inference, test-support verification and report emission. Full fingerprint
and file metadata checks repeat at its boundaries, detecting changed-then-restored
canonical inputs during this operation. Hashing the frozen corpus is identity
checking; test GT is decoded/scored only after gate success. Canonical inputs
must remain immutable throughout. Existing reports and a concurrent output
creation are preserved; if inputs change during emission, the newly created
report is discarded without removing a concurrently replaced file.

The report binds the reviewed selection SHA, checkpoint SHA/epoch/steps/lineage,
normalized configuration, original decision/completion/VAL receipts, source and
training-source identities, manifest/content/inventory fingerprints, exact frozen
protocol, real UTC times and observed device/runtime. Metrics name split `test`
and include every eligible test photo exactly once with actual source/record IDs,
GT support, per-photo counts, pooled/dataset/macro counts and conditional A/V
counts. It does not create a new selection score or reselect a model from test.
The shared count validator's default remains VAL; its explicit TEST mode checks
real test inventory and labels rather than relabeling a VAL result.

Unsigned records, local time and device observations cannot prove historical
blindness, authenticate a reviewer, attest the hardware, or prevent an operator
with data access from inspecting held-out labels. Cross-hardware bitwise
agreement is not claimed. Those limitations stay explicit in the report.
