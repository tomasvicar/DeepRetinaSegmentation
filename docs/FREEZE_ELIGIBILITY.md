# Typed freeze eligibility — experimental first pass

This branch adds loader compatibility and a separate CPU eligibility check. It
**does not create a selection, authorize TEST or implement V3 pool completion**.
Do not deploy this intermediate branch before the fail-closed selection wiring:
the compatible loader also makes older untyped gate code parse the new configs.

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

## Mandatory next integrations

1. Wire typed eligibility into create/check with explicit selected role, frozen
   inference/source leases and a separately versioned selection schema. No new
   source0b/72 path may fall back to the weak untyped historical completion gate.
2. Preserve old schema1 bytes as historical records; do not relabel them. Update
   the explicit reviewed stage2 API and metadata comparator deliberately for the
   new selection/role schema, without running TEST during implementation.
3. Add the **genuine source72 V3 final60 pool adapter**: original typed
   `v3_rung_evidence`, COMPLETE receipt, actual task/lease/plan/runtime/PBS pins,
   native V3 validation and independent terminal milestones. It must support
   selection among actual Meta candidates, not just a future standalone TPE run.
   Partial3/9/27 never qualifies. V3 ranks LAST worst6ratio; macro-BEST is a separate
   saved checkpoint role. No standalone QA can be manufactured without a real
   standalone supervisor terminal.

No source0b/72 live run imports this worktree. Production aliases and model cards
are unchanged. No final decision or TEST authorization is created by these checks.
