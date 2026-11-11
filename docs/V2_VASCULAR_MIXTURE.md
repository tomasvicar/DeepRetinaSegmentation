# Research recipe: vascular sampling weight2

This branch adds one sampling scalar, `vascular_sampling_weight`, to the existing
`UnifiedConfig` and `dataset_task_balanced` photo mixture. It implements proposed
variant A only. There is no new AV selection probability, observer cycle, quota,
architecture, loss, augmentation or scheduler. No quality gain is claimed.

The field defaults to `1.0` and is appended to the dataclass to preserve existing
positional constructor arguments. It accepts finite positive Python numbers and
rejects bool, NaN, infinity, zero and negatives. Any nondefault value requires
`sampling="dataset_task_balanced"`; configuration validation and direct weight
computation reject incompatible policies explicitly.

For each existing `(sampling_owner, task)` key, the replacement contribution is:

```
priority(task) / eligible_photo_population[sampling_owner, task]
priority(vessel) = priority(av) = vascular_sampling_weight
priority(disc) = priority(cup) = 1
```

The same memberships, alias ownership and `sampling_group` conflict guard remain.
This does not double an entire mixed vascular/disc photo weight: its DC terms
retain weight1. The mandatory shuffled coverage pass over all TRAIN photos is
unchanged; only replacement-draw mixture weights change. Default `1.0` executes the
literal frozen6b sum expression, with no added RNG calls. The sampler itself,
`sample`, `__getitem__`, augmentation, loss, model build and training loop are
unchanged source segments. AV observer choice stays0.5 where both AV/vessel rows
exist; each chosen observer keeps its own paired GT bundle.

## Two explicit presets

- `configs/unified-v2-control384.toml`: scalar1, existing control recipe.
- `configs/unified-v2-vascular2.toml`: identical settings except scalar2.

Both declare compact UNet/base12/depth4/BN, patch384/batch4, mild augmentation,
legacy25_raw25px/degree, full60 epochs ×8192 samples, coverage-first, seed42,
AdamW lr0.0003/decay1e−5, cosine/min-ratio0.05, unchanged BCE/Dice/AV/vessel losses,
macro-dataset VAL selection and all heads. Runtime device/profile binding remains
separate. Configs contain no GPU allocation, submission or execution authorization.
The ordinary unified config reader supports this field through `--config`.

The TRAIN-only analytical scenario from baseHPO1d310a7 expects selected vessel
opportunities3426.40→4251.58 and AV905.16→1184.79 per epoch. Disc falls6156.16→5433.83
and cup3871.28→3204.60. These are pre-crop expectations, not successful gradient
contributions or quality evidence. Restoring vascular exposure can hurt DC; cup
macro gain alone cannot establish improvement of all tasks.

## Compatibility evidence, including actual TRAIN flags

[Verification](../reports/v2-vascular-mixture/verification.json) records the new
source/training/execution and preset byte hashes. Controlled fixtures use the exact
frozen6b sampling method and `CoverageSampler` source with provenance. All four
existing policies have bit-identical default float64 weights. Three epochs of
indices and final Torch generator states match exactly for seeds0/42/717,
counts3/8/8192, with coverage on/off. Unknown/missing task flags, mixed memberships,
shared-source ownership, invalid scalars and policy conflicts are covered.

The [actual metadata audit](../reports/v2-vascular-mixture/actual-train-flags-hashseed42.json)
uses all5700 measured TRAIN record task flags and2529 photo groups from the exact
manifest17a227…5a767. Flag file SHA is62a4dd…49c17, retained under data-report commit
`312cd431f2c98b221f45967a7b308290144a610f`. It was derived by the separate pinned
TRAIN cache scan; this sampler check opens only the manifest JSON and that flag
JSON. It performs no RGB/GT-array decode, model forward, validation or TEST work.
Default weight bits and nine8192-draw epoch sequences/RNG states match frozen6b;
priority2 matches an independent key-weight oracle. Total key mass changes35→50.

**132 relevant tests pass** (41 new checks plus existing unified/config/sampler,
HPO, CLI, loss and exposure tests); Ruff passes. The full suite was not run.

The actual check fixes `PYTHONHASHSEED=42`; both compared implementations execute
in the same process/backend. Set summation can differ in final floating bits across
uncontrolled Python hash seeds. This evidence establishes source/default
compatibility under matched state, not reproduction of every asynchronous GPU run.
Reproduce with the report's `audit_actual_train_flags.py`, explicit pinned metadata
paths and a fresh output JSON. No further TRAIN-label scan is required.

## New source contract; no live continuation

This is research-only work based on HPO1d310a7. It changes training source bytes
and adds the scalar to serialized config/plan identity. Before any future run,
freeze its own commit, execution/training hashes, manifest/content/cache/policy and
preset/config hashes. New output and plan identities are required. Existing strict
resume guards must continue to reject switching source/config on an old run.

The current actual6b protocol and its fixed completion adapter remain unchanged.
Do not submit this config to that study, retag its receipts, import synthetic
completion or weaken its gate. A separately reviewed new recipe needs its own
pinned typed completion contract and actual contributions from all eligible TRAIN
annotation IDs, plus fresh paired VAL evidence. Wait for the current protocol's
actual full60 outcomes before deciding whether to allocate another GPU run.
Selection must inspect vessel, conditional/end-to-end AV, disc/cup and per-source
regressions together; TEST stays under the separate reviewed final-selection gate.
No stable cherry-pick or deployment is implied by this research branch.
