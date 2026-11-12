# Current product test inventory, not a test execution result

Snapshot: `d280cf504663136861cc38bcd0e3beeb0219b55e`, 12 November 2026.

[docs/TESTING.md](../../docs/TESTING.md) documents every current test module,
reproducible selected/full-suite commands, input requirements, and existing
source-scoped observed verification records. The inventory contains **62 modules
and451 test definitions**, not expanded pytest cases or new PASS counts.

`discover.py` uses Git to enumerate **all172 tracked Python files**, then stdlib
AST to inspect module/class `test*` definitions regardless of filename or class
prefix. No additional nonstandard modules were found in this product snapshot. It never
imports test modules, pytest, Torch, datasets or model checkpoints. `inventory.json`
records exact module SHAs, every test definition/line/decorator, module docstrings
and per-module commands. All current test modules use top-level functions; methods in any class and
nonstandard test-bearing report/script files are also supported.
Reference modules under `tests/reference` were scanned too; they are supporting
oracles, not additional test-bearing modules in this snapshot. Actual pytest import/collection/parameter expansion is a separate
operation described in the guide, not claimed here.

Actual documentation checks observed in this work:

- Regeneration produces an equal JSON value; all62 discovered module paths
  match the table and each module's command is present.
- All451 definition records retain source line/name/decorators; every module
  SHA matches the unchanged file.
- Every local Markdown link in the guide resolves; no linked dataset is opened.
- Torch remains absent from the process module registry.
- Ruff check/format on the discovery utility and `git diff --check` pass.

The original inventory-only operation did no pytest execution. Its later optional
bundled-defaults update links one separately observed34-case mocked metadata-only
module run in docs/TESTING.md; no entire-suite rerun is claimed. No NN forward,
checkpoint tensor load, GT decode,
training, GPU operation, dependency installation or package rebuild was performed.
The guide links historical successes and failures with their own source/role/data
scope; it does not assert the current entire suite is green. The inventory script itself does not change runtime, weights or defaults.
The later optional-defaults feature is separately documented; existing model
defaults and weights remain unchanged.
