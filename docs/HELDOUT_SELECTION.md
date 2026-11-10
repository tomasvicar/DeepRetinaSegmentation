# Frozen unified selection metadata — stage1

This is a CPU-only creation/verification workflow for a future held-out evaluation.
**The public unified CLI still rejects every `--split test` request**, even with a
valid selection artifact. No test evaluator integration or actual held-out result
is added here. A separate [fresh VAL producer](FRESH_VALIDATION.md) evaluates validation only.
The final test dispatcher requires separate review. Running inference/prediction
is not part of these metadata scripts.

## What is bound

Version1 `unified_heldout_selection` records contain these mandatory objects:

| Object | Checked contents |
| --- | --- |
| `decision` | Generated UTC recording time, explicit reviewer text and rationale |
| `checkpoint` | Actual immutable byte SHA, epoch/steps, normalized complete configuration and its SHA, architecture, channels, independent lineage, original training-selection score |
| `source` | Checkpoint training-source SHA plus actual evaluation/gate implementation, project metadata/uv.lock and runtime/precision/backend identities |
| `canonical_data` | Manifest-byte SHA, re-read schema2 prepared-content fingerprint, portable split/source/photo/record inventory SHA and eligible split sizes |
| `inference_protocol` | Canonical prepared geometry, RGB/255−0.5, legacy25/25pixels per degree, five channels, tiled/whole geometry, threshold, float32/no-autocast, AV joint fusion/argmax/vessel gating, cup containment, GT/source and aggregation policy |
| `validation_evidence` | Actual enriched VAL-report SHA, evaluation UTC time, selected metric and final-protocol score |
| `training_completion` | Last checkpoint SHA/epoch60, full60-row metric and summary SHAs, best epoch/score, actual optimizer-step/scheduler and coverage consistency |
| `boundaries` | Explicitly false test enablement, prior-blindness proof and authenticated reviewer identity |

Unknown/missing top-level fields, duplicate JSON keys and nonfinite JSON fail.
Device choice is omitted only from the normalized training configuration; the
checkpoint byte SHA still binds its original complete serialized contents.
Threshold/tiling changes cannot reuse a frozen record. The source digest covers **every actual `src/deepretina/**/*.py` file**, sorted
by its relative path, including added optional/residual modules; `.pyc` files are
excluded. `pyproject.toml` and `uv.lock` are also hashed. A source/runtime identity
check is repeated before the checked snapshot is returned. Actual float32 weights,
no active autocast, source/runtime/backend identity and checked inference settings
must agree. Source identity uses the actual files and uv.lock from a pinned source
checkout, independently of its location; an unreviewed package without that source
checkout cannot create this record.

## Completion and validation evidence requirements

Planned training must be60 epochs and `last.pt` must actually declare completed
E60, with an agreeing completed-budget summary, optimizer step history, scheduler,
full60-row CSV history and all eligible training-photo/family coverage. No early
stop exception or `max_steps` functionality run is accepted. The chosen `best.pt`
may come from an earlier epoch, but must agree with the completed run's maximum
training-validation score and recorded steps. Historical/teacher initialization,
unknown exposure lineage, unsupported heads, nonfinite weights/optimizer and
functionality/pilot checkpoints are rejected.

The external validation evidence must be a **freshly generated** version1
`unified_validation_evidence`, produced while evaluating this exact checkpoint
under the frozen final runtime/protocol. Current thin HPO/public-CLI validation
reports lack the required provenance block and are deliberately ineligible.
There is no command that retrospectively fabricates this missing provenance.

Its required fields are `schema_version`, `kind`, `checkpoint`, `configuration`,
`source`, `data`, `inference_protocol`, `evaluated_at_utc`, `metrics`. The three
identity objects and protocol must equal the corresponding live snapshot values;
`checkpoint` includes actual SHA/epoch/steps and original training-selection score.
The raw `metrics` use the existing unified evaluator format, with explicit
`selection_metric`/`selection_score`. They must contain every eligible VAL photo
exactly once, correct annotation-source/record IDs, all five supervised heads and
finite count-derived per-photo/per-dataset/pooled/macro and A/V accuracy headers.
The exclusive A/V decoder also requires per-photo
`tpA + fpV <= positiveGT_A`, `tpV + fpA <= positiveGT_V`, and
`tpA + tpV == end_to_end_correct <= conditional_correct <= nKnown - fpA - fpV`.
A gated class false positive is necessarily a conditional-classification error;
both heads cannot claim the same known vessel pixel as a correct positive.

The checker decodes **only the selected VAL GT NPZs**, through the shared canonical
mapper/FOV rules, to verify actual valid and positive pixel counts. It never
decodes test GT. Prepared-content verification streams bytes of the whole frozen
artifact for identity, including held-out files; hashing is not label interpretation
or evaluation. The full canonical fingerprint is checked before and after selected
VAL GT reads, against the same external checkpoint pin. Referenced-file metadata
(device/inode/size/mtime/ctime/mode, plus manifest/photos/audit/completion files)
must remain stable across the whole operation; this detects replacement, same-GT-
total label swaps and changed-then-restored bytes/mtime during a check. Canonical
inputs must be immutable by contract. These are operation-scoped checks, not a
promise that storage cannot change after the receipt or protection against an
operator controlling the filesystem/runtime. A future evaluator must enforce
its own immutable-input contract and reverify at its evaluation boundary. Training/final-inference validation scores are recorded separately:
a fresh final-protocol validation can differ from the earlier training measurement,
but both are bound explicitly to the selected checkpoint and evidence.

The validation evaluation time must precede the recorded decision. An external
review digest must be supplied to `check`; a matching filename or a hash of an
arbitrary JSON file is insufficient. The entire schema and actual checkpoint,
source, config, data, protocol, GT support and completed run are checked again.

## CPU-only commands

Run from this pinned checkout with its frozen uv environment. No GPU is needed;
the script caps Torch metadata work to two CPU threads (`--cpu-threads 1` is valid).
The separate fresh producer is documented in [FRESH_VALIDATION.md](FRESH_VALIDATION.md).
Its tests use synthetic CPU snapshots; no current trained-model decision record
is minted as part of this change.

```bash
uv run python scripts/unified_selection.py create \
  --checkpoint RUN/best.pt --last-checkpoint RUN/last.pt \
  --manifest DATA/manifest.json --validation-evidence fresh-enriched-val.json \
  --metrics-csv RUN/metrics.csv --training-summary RUN/training_summary.json \
  --reviewer REVIEWER --rationale 'Frozen validation choice and tradeoff' \
  --output review/selection.json
```

Record the emitted selection SHA in the review. Then the same explicit inputs:

```bash
uv run python scripts/unified_selection.py check \
  --checkpoint RUN/best.pt --last-checkpoint RUN/last.pt \
  --manifest DATA/manifest.json --validation-evidence fresh-enriched-val.json \
  --metrics-csv RUN/metrics.csv --training-summary RUN/training_summary.json \
  --selection review/selection.json --expected-selection-sha256 REVIEWED_SHA
```

Both commands remain metadata-only. `test_evaluation_enabled` is always false.
Creation refuses overwrite. Scripts use CPU `weights_only=True` loading through
the public immutable-snapshot loader; no unsafe pickle fallback is introduced.

The runtime identity includes installed distribution versions via
`importlib.metadata` for Torch/NumPy, Pillow, SciPy, scikit-image, both OpenCV
package variants, h5py and pydicom (explicit null if a variant is absent), plus
precision/backend settings. These are installed-version observations, not a
claim that lockfile contents prove which binaries ran. The CPU checker does not
query/attest GPU hardware or driver state. The future fresh validation/test
producer must log actual devices/drivers separately; cross-hardware bitwise
equivalence is not implied. Source-byte hashes do not attest arbitrary runtime
monkeypatching or third-party binary contents.

## Future integration boundary

`check_selection` returns immutable checked checkpoint bytes, protocol and canonical
identity snapshots. A future reviewed test dispatcher must consume **those exact
bytes**, not validate one file and reopen its pathname later. Replacing a same-name
checkpoint after checking cannot change the returned snapshot. Actual canonical
inputs must remain immutable during any future evaluation, and source/protocol
must be verified before the test reader is entered. No such dispatcher is enabled
in stage1; the lower-level research evaluator remains callable Python code.

This is a reproducibility/process gate, **not an access-control or cryptographic
attestation system**. An operator with source/data access can manually inspect
held-out data or forge unsigned evidence. Reviewer text and local UTC time cannot
prove identity, chronology of all past actions, or that test data were never seen.
Hashes detect changes to the reviewed content; they do not prove historical blindness,
actual physical training effort, or optimality among all possible candidates.

CPU tests construct tiny synthetic snapshots and forbid every model forward and
CUDA initialization. They exercise same-name replacement, snapshot retention,
wrong source/config/data/protocol/GT, thin/arbitrary evidence, incorrect completion,
nonfinite/teacher/pilot models, LR/optimizer history, timestamps/duplicate JSON,
Python `-O` and the still-blocked public test route. No actual model selection,
real validation prediction or test prediction is performed by this test suite.

The separate **fresh enriched VAL producer** runs under the selected final
runtime/protocol; see [FRESH_VALIDATION.md](FRESH_VALIDATION.md). Existing thin
historical reports must not be retroactively given invented provenance. Stage1
still enables no test entrypoint and its metadata scripts perform no network
forward.
