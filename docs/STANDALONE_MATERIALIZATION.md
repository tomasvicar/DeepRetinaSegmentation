# Relocating original standalone72 completion evidence

Audit baseline: `8961276cdee09fbca9eea3d2a3fd329c0f443d6b`, 12 November 2026.
This is a byte-preserving transport recipe, **not** a completion, selected model,
permission to evaluate TEST, or authorization to run a future continuation.
No actual future width24 E27/E60 evidence existed for this audit.

## Finding: the existing gate already accepts relocated files

`check_freeze_eligibility(family="standalone_72", ...)` accepts explicit paths to
materialized inputs. `_pinned_json` verifies externally supplied hashes of QA,
protocol and original training terminal. `_standalone` then compares the original
protocol/terminal bytes with the exact text embedded in the genuine QA. It checks
source72 commit/execution/training hashes, original config/cache/corpus identities,
observed completed60 terminal and declared budget. It does **not** require the
current caller's file paths to equal path strings embedded in those originals.

Therefore `/workspace` and `/runs/train` can remain in the original protocol,
terminal and QA while files are read from a new host's read-only backup. The
original JSON must never be rewritten to the new host path. No gate dispatch,
family allowlist, schema, source identity or coverage rule needs to change.
Externally pinned genuine QA is essential: a newly composed index or copied
filename alone does not establish that the original verifier actually ran.

The unchanged original standalone verifier `4b568b7e…eb56fe` has a different job:
its `check_contracts` deliberately requires the actual source/run arguments to
match the protocol's source/root and run strings. Its actual Git/source import
checks must also pass. A direct re-run with `--source /new/checkout --run
/new/backup` will fail when the original says `/workspace` and `/runs/train`.
That refusal must remain. If an independent re-audit is later authorized, use the
same read-only container namespace for source/run/canonical data and genuine Git
metadata, with output outside those inputs. Do not edit the original protocol,
mock Git, patch source checks or present a relocated custom QA as original4b QA.
The final freeze gate need not rerun4b: it verifies its externally pinned genuine
original result and independently recomputes checkpoint/history/own coverage.

## Immutable materialization index

Keep a **separate**, externally pinned inventory of where bytes were transported.
It is an operational custody record, not a new completion family and not a new
argument secretly accepted by the standalone gate. A practical shape is:

```json
{
  "schema_version": 1,
  "kind": "original_standalone72_materialization_index",
  "original_namespace": {"source_root": "/workspace", "run_root": "/runs/train"},
  "source": {
    "commit": "<original reviewed source72 commit>",
    "execution_code_sha256": "<original execution pin>",
    "training_source_sha256": "<original training pin>",
    "local_checkout": "<read-only genuine checkout>"
  },
  "contracts": {
    "protocol": {"path": "<copied original>", "sha256": "<external original pin>"},
    "training_terminal": {"path": "<copied original>", "sha256": "<external original pin>"},
    "completion_qa": {"path": "<copied original>", "sha256": "<external original pin>"}
  },
  "run_artifacts": {
    "best.pt": {"path": "<new backup>/best.pt", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "last.pt": {"path": "<new backup>/last.pt", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "metrics.csv": {"path": "<new backup>/metrics.csv", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "training_summary.json": {"path": "<new backup>/training_summary.json", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "run.json": {"path": "<new backup>/run.json", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "coverage.json": {"path": "<new backup>/coverage.json", "bytes": "<actual integer>", "sha256": "<original pin>"},
    "validation.json": {"path": "<new backup>/validation.json", "bytes": "<actual integer>", "sha256": "<original pin>"}
  },
  "external_exit_evidence": {
    "training_child": {"path": "<original observed receipt>", "sha256": "<pin>"},
    "completion_verifier": {"path": "<original actual exit receipt>", "sha256": "<pin>"},
    "allocation_terminal": {"path": "<original externally observed receipt>", "sha256": "<pin>"},
    "copy_verification": {"path": "<original independent snapshot receipt>", "sha256": "<pin>"}
  }
}
```

This is an illustrative schema, not a runnable approval: every placeholder must
come from actual stopped originals. Seven original run files plus original QA
make the eight-file model/QA bundle; protocol, training terminal and independently
observed process/allocation/copy receipts are additional mandatory custody inputs.
Record numeric sizes, actual read-only modes and independent copy verification.
A new host path belongs **only** in this index and command arguments. Preserve
literal original text, paths, line endings and bytes in all originals.

Before invoking eligibility on a future real completion:

1. Preserve original stopped source/run/protocol/terminal/QA and external exits.
   Verify genuine observed training0, actual verifier0 and original overall
   allocation success independently; self-planned0 is not an observed exit.
2. Copy exclusively to a fresh immutable namespace; refuse existing destinations.
   Rehash local copies against original pins, verify encoded sizes and keep
   separate, non-writable model copies. Do not lend a writable hardlink to a live
   run. The original source can be a clean genuine checkout of exactly the same
   source72 commit; a filename or hand-copied module is not Git verification.
3. Require exactly the seven run names above. Every file hash must agree with
   `training_terminal.artifact_sha256`; model hashes must also agree with
   `completion_qa.checkpoint_sha256`, and all five sidecars with
   `completion_qa.sidecar_sha256`. Preserve the QA's original protocol/terminal
   texts and external pins, even when its source/run strings are container paths.
4. Pin the materialization index itself externally after review. Its custody
   checks supplement, rather than replace, the actual eligibility checks below.
   The current standalone gate consumes BEST/LAST, CSV/summary and original QA/
   protocol/terminal; it does not independently read the other three sidecars or
   an allocation scheduler receipt. Preserve and verify those through this
   explicit copy/operations chain; do not claim the gate checked inputs it did not.
5. Use an explicitly frozen source-compatible evaluation checkout and unchanged
   canonical corpus. The evaluation-source lease is a separate identity from the
   original training source's container path. Fresh VAL/freeze must use that actual
   new evaluation source; historical reports are not retagged as new-source VAL.

## Existing CLI, with relocated explicit inputs

After genuine completion/custody review only, the existing CLI accepts this shape
(the placeholder hashes must be reviewed originals; **not run by this audit**):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python scripts/check_freeze_eligibility.py \
  --family standalone_72 --checkpoint-role fixed_terminal60 \
  --checkpoint /new/backup/last.pt \
  --best-checkpoint /new/backup/best.pt --last-checkpoint /new/backup/last.pt \
  --metrics-csv /new/backup/metrics.csv \
  --training-summary /new/backup/training_summary.json \
  --manifest /same-canonical/manifest.json \
  --completion-qa /new/contracts/full60-QA.json --expected-qa-sha256 QA_SHA \
  --reviewed-protocol /new/contracts/original-protocol.json \
  --expected-protocol-sha256 PROTOCOL_SHA \
  --original-terminal /new/contracts/original-training-terminal.json \
  --expected-terminal-sha256 TERMINAL_SHA \
  --output /new/audit/last-eligibility.json
```

For `training_best`, select `best.pt` and change the explicit role; retain actual
LAST60 as the completion witness. Both terminal and selected weights must cover
all actual2,529 TRAIN photos and all5,700 annotation records under their **own**
coverage and budget histories. An incomplete BEST is rejected even if LAST60 is
complete; fixed-terminal selection must use LAST's own saved VAL metric/epoch.
The same source/config/cache/manifest/profile checks remain mandatory. Relocation
never turns a partial27 record, native V3 pool receipt or custom materialization
index into original standalone72 completion QA.

## Narrow synthetic proof and limits

[`tests/test_standalone_materialization.py`](../tests/test_standalone_materialization.py)
exercises the actual existing `_pinned_json` and `_standalone` functions with
synthetic metadata and unchanged original container path text. It shows equal
contract results after read-only copying, rejects changes under stale external
pins and rejects replacing only the protocol's path with the QA left original.
A separate marker-byte test checks the seven-file terminal/QA hash inventory and
BEST/LAST substitution refusal. Marker `.pt` files are deliberately not model
checkpoints and are never deserialized. No full eligibility PASS or real width24
completion is claimed by these narrow tests.

Existing [`test_freeze_eligibility.py`](../tests/test_freeze_eligibility.py) checks
actual own-photo/annotation coverage helpers with synthetic2,529/5,700 IDs and
rejects missing/zero/foreign annotations, insufficient visits and sample-budget
mismatch. Full checkpoint/AdamW/RNG/history/canonical/source verification remains
in the unchanged production gate; these metadata tests do not replace it.

No production code changed. No checkpoint/source archive/corpus hashing, model
load/forward, GT read, GPU call, transport, scheduler operation or TEST was
performed. Operational paths and actual future deployment remain in private notes.

Observed on this audit snapshot: six new namespace fixtures passed in1.59s and
ten existing coverage/family/byte-pin fixtures passed in1.79s on CPU1, CUDA hidden.
A separate synthetic call to byte-exact original4b `check_contracts` returned
normally with `/workspace`+`/runs/train` and rejected each direct relocated source
or run argument; the external probe process returned0. This probe did not invoke
Git verification, model loading or full completion QA. The initial new-fixture
run correctly rejected earlier than its asserted message because the synthetic
source dictionary was shared between protocol and terminal; only fixture
isolation was corrected, not the production gate. See the
[recorded source identities, results and limits](../reports/standalone-materialization-audit-20261112/verification.json).
