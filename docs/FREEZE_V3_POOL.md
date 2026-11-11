# Genuine V3 pool final60 eligibility

This experimental addition extends schema2 with `completion_family=pool_72_v3`.
It consumes the original COMPLETE worker receipt as `completion_qa`; it does not
create a standalone QA or supervisor terminal. No actual full60 V3 pool positive
exists in the implementation evidence. Live source72/core8777/workers are unchanged.

The existing `original_pool_evidence` / `expected_pool_evidence_sha256` inputs are
required for this family in eligibility, create/check and reviewed stage2. The
externally pinned JSON index has this shape (paths remain in private operational
storage and are not copied into the public checked proof):

```json
{
  "schema_version": 1,
  "kind": "original_v3_pool_evidence_index",
  "core_source_commit": "8777d6cca60f7ced08798ad2af7fabf626421d05",
  "files": {
    "task": {"path": "original-task.json", "sha256": "ACTUAL_SHA"},
    "protocol": {"path": "original-protocol.json", "sha256": "ACTUAL_SHA"},
    "envelope": {"path": "original-envelope.json", "sha256": "ACTUAL_SHA"},
    "profile": {"path": "original-profile.json", "sha256": "ACTUAL_SHA"}
  },
  "milestone_directory": "preserved-host-independent-milestone"
}
```

Index paths use the caller's ordinary filesystem resolution. Original protocol,
envelope, task/lease, config, study/source/cache/canonical and profile hashes must
agree. Original `run_result` must show integer returncode0, no TEST, actual physical
GPU/PBS/runtime/worker bindings, and original typed `v3_rung_evidence` must bind raw
BEST/LAST, CSV and the exact stored LAST metrics. Native8777 `validate_protocol`
and `validate_rung` are called directly from vendored byte-exact stdlib modules
and fixed public reference JSON. Their hashes are checked at runtime; there is no
Optuna/SQLite dependency in the evaluation environment. The `adaptive64` study
suffix denotes worker RAM, not64 configurations; the native recipe retains24.

The original host-independent milestone must attest a verified copy without
hardlinks. Every actual copied BEST/LAST/CSV/summary/coverage/validation/run/runner
marker byte hash is rechecked, with read-only files and no shared file hardlinks.
Actual run metadata and marker must match the training recipe. Inputs are checked
again after the entire eligibility operation. Original receipts and artifacts
are never rewritten. These are unsigned process/provenance checks, not hardware
attestation, authenticated authority, or proof of prior blindness.

Both `training_best` and `fixed_terminal60` retain the existing raw checkpoint
role, own epoch/CSV score and own full5700 selected annotation requirements. The
V3 study's LAST60 worst-of-six ratio is recorded separately as
`study_terminal_objective` with role `fixed_terminal60_worst6_ratio`. The macro
BEST epoch/score remains distinct; a macro-BEST snapshot cannot inherit LAST60
coverage or the study objective. Stage2 and the metadata comparator retain this
family distinction and the externally reviewed schema2 pin.

CPU synthetic positive tests use explicitly fabricated final60 training receipts
and dummy milestone bytes, with native8777 validation against the fixed actual
public770 support inventory and2529/5700 TRAIN identity hashes. They do not imply
an actual candidate completed60. Negative tests cover partial runs, failed child,
GPU/lease/role/source/objective/coverage/milestone and retagging errors. The actual
five E27 original receipts pass native partial-rung validation and are all rejected
for final freeze before reading checkpoints. Actual saved E3 is separately
identified as partial. See `reports/freeze-v3-pool72-actual-partial-20261111`.

No TEST decoding/evaluation, model forward, new training, model selection or
release occurs in this change. A real final60 pool candidate still needs its
original final receipt, retained milestone and fresh final-source VAL before
root review; a synthetic fixture or old thin report cannot substitute for them.
