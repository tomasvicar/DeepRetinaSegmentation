# Fresh unified VAL evidence

This optional command evaluates **VAL only** from one immutable checkpoint-byte
snapshot and creates version1 `unified_validation_evidence` accepted by the
[selection metadata gate](HELDOUT_SELECTION.md). It does not select a model,
produce test predictions or enable public `--split test`. No historical thin
report is accepted as input or retroactively enriched.

Run from the pinned source checkout using its uv environment:

```bash
uv run python scripts/unified_validation.py \
  --checkpoint RUN/best.pt --manifest DATA/manifest.json \
  --output reports/fresh-selected-val.json --device cpu --cpu-threads 1
```

For an explicitly scheduled GPU validation, use `--device cuda:0`; the producer
records actual device properties, CUDA build, cuDNN version and observed driver
rows. This change was verified using tiny synthetic CPU snapshots only; it adds
no actual trained-model validation result or quality claim. No GPU job is
launched by installing or importing it.

The checkpoint must be independent scratch lineage, all five supervised heads,
`purpose=full_data`, canonical legacy25/25pixels-per-degree and a planned60-epoch
budget without max_steps. Checkpoints from E1 through E60 can produce fresh VAL
evidence; selection creation separately requires a completed last60 checkpoint,
coverage, CSV and summary. Early evidence does not waive completion requirements.

`--threshold`, `--tile-size` and `--overlap` are optional explicit protocol
choices. Otherwise checkpoint inference defaults apply. Whole-view checkpoints
reject tiling overrides. To freeze/check selection for a nondefault protocol,
pass the same overrides to `unified_selection.py`; another threshold or geometry
cannot reuse this evidence.

## Recorded checks

Before any model forward, the producer rejects an existing output, safely checks
checkpoint metadata with `weights_only=True`, verifies the externally pinned
canonical content/manifest/inventory and records actual source/runtime identity.
It loads the evaluated weights from those same checked bytes via BytesIO, never
reopening the checkpoint path. Changing `best.pt` afterwards therefore cannot
change the evaluated snapshot.

The evaluator receives only split `val`. Every eligible VAL photo and annotation
record must appear exactly once with the source-attribution and A/V count
constraints required by the gate. The same shared mapper verifies actual selected
VAL GT support. Test GT is never decoded or scored. Hashing the whole canonical
artifact for identity is not test-label evaluation; selected-VAL NPZ reads are
separately scoped.

Full canonical content/operation metadata and source/runtime checks repeat across
inference and GT-support verification. The producer fails without a report if
inputs change, including changed-then-restored canonical files. Canonical inputs
must remain immutable throughout this operation. It records an actual completion
UTC timestamp, final-protocol validation score and the original training-selection
score separately. Output is opened exclusively, outside the canonical corpus;
existing reports, including a file created during inference, are preserved.

`metrics.evaluation_runtime` contains extra device/thread/environment observations
for reproducibility. They are not authenticated hardware evidence and are not an
additional hardware-enforcement guarantee of the stage1 gate. Source-byte hashes
and unsigned reports cannot prove historical blindness, protect against an
operator controlling the runtime, or make cross-hardware bitwise promises.

After this fresh evaluation is reviewed, the existing separate create/check
workflow can bind it to a proposed final decision. The public test CLI remains
blocked pending an independently reviewed dispatcher consuming the checked byte
snapshot.
