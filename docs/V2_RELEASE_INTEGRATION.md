# Verified V2 release integration

Merged into `for_Leipzig` after root review of the scoped runtime diff, complete
CPU test evidence and bundled-file preservation receipt. No training process
or checkpoint bytes changed during the merge.

This integration starts from stable `for_Leipzig` commit
`0b8c74ccf6824057100fffc9b4328e5b6fc9c61e`. It selectively integrates V2 preparation,
loss controls and portable checkpoint geometry. It does not merge the HPO branch.
No new trained model, model-quality improvement, new default, or Meta V2 readiness
is claimed. The active frozen training checkout is not modified by this integration.

## Selected changes

| Source commit | Included behavior/evidence |
| --- | --- |
| `8694bc8`, `84d4464` | Reviewed nominal geometry policy, exact JSON and hash receipt |
| `016f193` | Optional vessel loss/positive BCE weights, defaults1, v1 loss reference tests |
| `b8e1c9f`, `9cd9278` | Archive audit and native DRIONS/GAMMA/IDRiD annotation adapters |
| `925cfa5` | Explicit audited BIOMISA export-role policy; default quarantine retained |
| `399805c`, `527317d` | Independent append producer and completed conversion/reader evidence |
| `6b3dfad` | Geometry policy captured in checkpoints, validated on load, portable prediction |
| `4fc0e8d`, `7ca9705` | Alias/read-only copy hardening and metadata-only artifact-freeze evidence |

All patches applied to the stable base without source-code conflicts. The only
cherry-pick conflict was a preparation report README absent from the stable base;
the audited report version was retained. Its two historical dependencies,
`launch.json` and `bounded_launcher.py`, are copied byte-for-byte from `527317d` so
its evidence links resolve. These historical scripts were not executed.

Runtime edits are confined to `unified.py`, `prepare.py`, adapter coverage, and
three new modules: additional native adapters, explicit BIOMISA policy and portable
geometry policy. Stable original/source A/V modules, all nine bundled model files,
registry aliases, CLI, bundled dispatcher and model module remain
byte-identical to the stable base. Full hashes are in
[the preservation receipt](../reports/v2-release-integration/preservation.json).
The original full-data baseline checkpoint is preserved byte-for-byte. README is
the intentional documentation exception: it links the usable V2 preparation path
and explicitly identifies the unchanged bundled baseline as corpus-v1. All27
other protected files remain byte-identical.

## Data and compatibility boundaries

The append producer supports the verified 21-source corpus: TRAIN2529 photos/5700
annotation records, VAL770 photos/993 records and TEST856 photos/1118 records.
The nominal geometry and BIOMISA role choices remain explicitly qualified research
policies. The original corpus memberships and bytes are not migrated. Preparation
is an explicit offline action into an independent output root; this integration
performs no data conversion or data writes.

Old checkpoints without geometry policy retain their existing behavior. New
policy-bearing checkpoints validate the embedded policy and use it for prediction
without requiring the original preparation directory. Loss weights default to1;
existing bundled checkpoints, original model aliases and default selection remain
unchanged. Future V2 weights require their own verified artifact/provenance and
review before any release/default decision.

## Verification

The verification receipt and full test log are stored in
`reports/v2-release-integration/`. Tests use the existing isolated env87 with this
worktree's `src` first on `PYTHONPATH`, CPU affinity restricted to one core, nice10,
one numerical-library thread, hidden CUDA devices and explicit exceptions on CUDA
initialization. Only temporary synthetic test files are created. No training run,
model TEST scoring, real prepared-data modification or active GPU work is performed.

Full CPU suite: **2517 passed**, 79 dependency/runtime warnings, 51.61seconds.
The ASHA reference-only test introduced by the loss cherry was removed from this
release branch: stable has no optional Optuna package/dependency. Its recorded proof
remains in `reports/v2-loss/asha-reference-proof.json` and the HPO branch.

The added geometry module changes the training-source fingerprint. Resume an old
training run only with its original frozen source/environment and matching plan;
this branch does not bypass source-hash resume checks. Existing weight loading
and inference remain supported, while retraining on21sources starts a fresh run.

`uv build --wheel` succeeded. ZIP inspection verified the new geometry/native
adapter modules, existing source A/V modules, registry and all nine bundled models
are included and byte-identical to the worktree; see
[wheel receipt](../reports/v2-release-integration/wheel.json).
