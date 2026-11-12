# Current product test inventory, not a test execution result

Snapshot: `e27a7747c711c4714fd0616422b405a3132a34e6`, 12 November 2026.

[docs/TESTING.md](../../docs/TESTING.md) documents every current test module,
reproducible selected/full-suite commands, input requirements, and existing
source-scoped observed verification records. The inventory contains **61 modules
and439 test definitions**, not expanded pytest cases or new PASS counts.

`discover.py` uses stdlib AST to inspect `tests/test_*.py` recursively. It never
imports test modules, pytest, Torch, datasets or model checkpoints. `inventory.json`
records exact module SHAs, every test definition/line/decorator, module docstrings
and per-module commands. All current modules use top-level test functions;
Test-prefixed class methods are supported by the discovery helper as well.
Reference modules under `tests/reference` are supporting oracles, not separate
pytest modules. Actual pytest import/collection/parameter expansion is a separate
operation described in the guide, not claimed here.

Actual documentation checks observed in this work:

- Regeneration produces an equal JSON value; all61 discovered module paths
  match the table and each module's command is present.
- All439 definition records retain source line/name/decorators; every module
  SHA matches the unchanged file.
- Every local Markdown link in the guide resolves; no linked dataset is opened.
- Torch remains absent from the process module registry.
- Ruff check/format on the discovery utility and `git diff --check` pass.

No pytest collection/execution, NN forward, checkpoint tensor load, GT decode,
training, GPU operation, dependency installation or package rebuild was performed.
The guide links historical successes and failures with their own source/role/data
scope; it does not assert the current entire suite is green. Product runtime,
models, defaults and all test source files remain unchanged.
