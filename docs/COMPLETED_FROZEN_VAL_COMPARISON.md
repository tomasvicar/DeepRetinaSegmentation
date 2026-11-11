# Completed unified VAL versus a fresh frozen original reference

`scripts/compare_frozen_completed_val.py` accepts the flat
`family=frozen_original_reference` emitted by `evaluate_original_frozen.py`.
The older `compare_completed_val.py` continues to require its original composed
cup-clipped family. Neither route relabels one family as the other.

This is **metadata-only VAL analysis**: no images, GT arrays, checkpoints,
forward passes, CUDA initialization or TEST evaluation. It is intended for a
future actually completed60 V2 run; an unfinished rung is rejected. This change
has not produced a real model comparison or new quality result.

Both pipelines must have their own externally reviewed, SHA-pinned JSON context.
Their source/runtime identities may differ; each is checked against its own
context, never against the other pipeline or the current checkout. A context
records reviewed byte identities; hashes do not authenticate the reviewer or
prove historical blinding. Context creation is an explicit review step, not a
command that silently trusts whatever report happens to be present.

The original context has this shape:

```json
{
  "schema_version": 1,
  "kind": "completed_val_original_context",
  "report_sha256": "<SHA256 of complete original validation.json bytes>",
  "reviewed_at_utc": "<actual timezone-aware UTC review timestamp>",
  "rationale": "<reviewed original source, runtime and declared recipe>",
  "bindings": {
    "source": "<exact complete original report source object>",
    "policy_spec": "<exact original policy object>",
    "manifest_sha256": "<exact report value>",
    "prepared_content_sha256": "<exact report value>",
    "eligible_photo_counts": "<exact report object>",
    "inventory_sha256": "<exact report value>",
    "geometry_policy_sha256": "<exact report value>",
    "photos_sha256": "<exact report value: hash of photos.json bytes, NOT photo-ID list>"
  }
}
```

The unified context uses `kind=completed_val_unified_context`, the complete
enriched VAL report SHA, actual review timestamp/rationale, and
`completion_qa_sha256` for the completed-run QA artifact. Its `bindings` contains
exact complete objects `source`, `configuration`, `inference_protocol`, `data`,
and `checkpoint` copied from the *reviewed* enriched evidence. In both examples,
angle-bracket descriptions stand for JSON values/objects, not literal strings.
Keep the externally reviewed context SHA independently of the file supplied to
the command. For V2 pool completion, `adapt_pool_completion_qa.py` is the separate
strict receipt/task/decision adapter; its output remains a separately pinned
input here. No synthetic or imported quality baseline substitutes for actual60.

```bash
uv run python scripts/compare_frozen_completed_val.py \
  --reference original-validation.json \
  --reference-context reviewed-original-context.json \
  --trusted-reference-context-sha256 "$ORIGINAL_CONTEXT_SHA" \
  --candidate unified-validation.json \
  --candidate-context reviewed-unified-context.json \
  --trusted-candidate-context-sha256 "$UNIFIED_CONTEXT_SHA" \
  --completion-qa completed60-qa.json \
  --manifest /path/to/canonical/manifest.json \
  --output completed-val-comparison.json
```

The command requires exactly770 eligible VAL photos, the complete annotation-ID
inventory and deterministic source attribution, shared manifest/content identity,
per-head valid/positive GT support, and exclusive-decoder A/V count consistency.
TRAIN count comes from the supplied manifest (2529 for the current V2 corpus),
not a hardcoded V1 count. Completion requires terminal60, finite best/last and
optimizer evidence, all-photo coverage,60×8192 delivered samples, matching
selected best checkpoint/training source/data, and the fixed full-data60 config.
The original four-model recipe/checkpoint declarations are checked separately.
The new route accepts only the fixed unified threshold0.5; it adds no threshold
selection or bridge to a different corpus. Different manifests require a
separately reviewed route, not rewritten metadata.

All input JSON bytes are snapshotted, parsed with duplicate-key/nonfinite-value
rejection, compared from those snapshots, then checked unchanged. Outputs are
exclusive and existing files/symlinks are rejected before comparison. The result
retains complete input hashes, both source identities, original recipe, selected
unified checkpoint/configuration and inference protocol. Intervals describe
source-stratified paired split groups, not proven independent patients. The
original raw photometry's measured MATLAB residuals and incompletely recoverable
historical training exposure remain qualifications.

Validation: synthetic770-photo fixtures only, plus the existing old completed
VAL and reviewed-report gates. No actual TEST report or data was read.

The isolated branch based on HPO1a87636 passes70 targeted tests in4.72s
(CPU only; OMP/MKL/OpenBLAS1, CUDA hidden), comprising the new adapter and the
unchanged completed-VAL/reviewed-report gates. Ruff check and formatting pass.
Tests include external-context changes, altered source/config/protocol and
completion, exact770 inventories, annotation/support/A/V counterexamples,
input replacement during comparison, existing-output preservation, and a fresh
subprocess confirming that importing this route does not import Torch/models.
