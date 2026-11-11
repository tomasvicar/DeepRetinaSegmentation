# Typed freeze eligibility and schema2 review integration

This experimental branch adds source-compatible loading, typed CPU eligibility
and fail-closed schema2 create/check/reviewed execution interfaces. No actual
model selection or TEST evaluation was performed during implementation. Genuine V3 pool
completion has a separate [native adapter](FREEZE_V3_POOL.md). The earlier compatibility-only commit
is not independently deployable: consume the complete reviewed integration.

`check_freeze_eligibility` / `scripts/check_freeze_eligibility.py` require explicit
family, selected role, BEST/LAST snapshots, actual manifest, original metrics CSV
and summary, and an externally SHA-pinned completion QA. Supported families:

- `pool_6b`: genuine `trusted_pool_completion_qa`, original COMPLETE worker receipt,
  task/plan/config bindings and original finished decision.
- `standalone_0b`, `standalone_72`: genuine `standalone_unified_completion_qa` from
  verifier4b568, separately pinned original reviewed protocol and observed bounded
  supervisor terminal. Raw source/config/data/cache/artifact hashes must agree.

No name/QA retagging or field filtering is permitted. The raw saved training config
is retained; new runtime defaults do not retroactively enter its source-family hash.
Native vascular sampling and BF16 features come from1775a2a/5c1d2d6, with fields
appended to preserve the prior HPO positional API. Default FP32/sampler behavior
passes56 feature-contract tests. BF16 training metadata retains float32 stored
weights and float32 inference; there is no CUDA/BF16 execution in eligibility.

Two distinct roles are supported:

- `training_best`: selected bytes must equal the genuine CSV-selected BEST; its
  own epoch's coverage must include all5700 eligible annotations and2529 photos.
- `fixed_terminal60`: selected bytes must equal LAST60. True BEST remains recorded
  separately; LAST's own CSV score is never replaced with the BEST score.

Actual TRAIN IDs come from the pinned manifest; photo/source/split-group aliases
must not cross partitions. Both terminal and selected weight require full own photo
and positive annotation-ID coverage, exact epoch sample totals, finite stored
state, complete optimizer parameter/moment bindings, scheduler/RNG and raw60CSV
history. Encoded canonical byte/source leases are checked without GT-array decoding
or NN forward. The result retains the selected immutable weight-byte snapshot and
explicit provenance/coverage; it is not a frozen selection or an execution permit.

## Actual first-pass checks

`reports/freeze-eligibility-actual-20261111` records actual CPU-only checks:

- Bundled0b BEST59 `81113e7e…`: **PASS**, genuine original QA/protocol/terminal,
  all5700 selected annotations,2529 photos and actual unchanged canonical bytes.
- Control6b LAST60 `10670cff…`: **PASS**, original pool family, all5700 selected
  annotations; the genuine BEST52 remains separate.
- Control6b BEST52: **REJECT**, selected weights have only5699 annotation visits.
- Real72 LAST3: **REJECT**, actual run has not completed60. Its V3 partial receipt
  was not converted to standalone provenance or a final eligibility claim.

Both accepted checks performed actual corpus-byte hashing (~130seconds each), safe
CPU checkpoint loading and optimizer/coverage audits; no real VAL/TEST labels or
model forward. Ten additional metadata invariant tests reject missing/zero/foreign
annotations, deficient photo/sample budgets, unsupported family/role and wrong
external pins. Real inputs and historical outcomes were not modified.

## Integrated selection and remaining scope

Schema2 create/check and the explicit reviewed stage2 API now rebuild this typed
gate with externally pinned original inputs and the explicit selected role.
Historical schema1 is never rewritten and is rejected for new execution. The
metadata comparator supports both schema2 roles and preserves a historical
schema1 reader. No compatible-source path can fall back to untyped completion.
See [selection](HELDOUT_SELECTION.md) and [reviewed execution](REVIEWED_HELDOUT.md).

The separate [genuine source72 V3 final60 pool adapter](FREEZE_V3_POOL.md) consumes
original native rung/COMPLETE/task/protocol/profile evidence and host-independent
milestones. It supports both roles while retaining LAST worst6 objective and
macro-BEST separately. Actual final60 positive evidence remains pending; all
actual E27/E3 artifacts are partial and cannot qualify.

No source0b/72 live run imports this worktree. Production aliases and model cards
are unchanged. No final decision or TEST authorization is created by these checks.

## Schema2 integration validation

The six targeted test modules contain205 distinct synthetic CPU cases. The main
199-case batch passed in34.04seconds; the six additional malformed-proof,
fixed-terminal, changed-request and explicit-default cases passed in focused
runs. Ruff passed. Toy stage2 tests deliberately isolate the fixed full-corpus
eligibility boundary with an explicit in-process mock; no fixture bypass exists
in production. The typed coverage/profile checker has separate synthetic tests
and actual completed-artifact proofs.

Final integration proofs are saved separately under
`reports/freeze-eligibility-schema2-actual-20261111`; the earlier first-pass proofs
are retained without retagging their source identity. These prove CPU eligibility
only. They are neither enriched final-source VAL nor a model selection, and must
not be substituted for either. A later final source merge changes the evaluation
identity and requires rebuilding its own operation-scoped evidence.
