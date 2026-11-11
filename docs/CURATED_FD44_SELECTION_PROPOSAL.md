# Curated fd44 final-eligibility adapter proposal

This branch adds a distinct `curated_fd44` completion family for review. It does
not select a model, freeze an actual artifact, evaluate actual VAL/TEST data, or
change a default inference protocol. Existing standalone and native-pool families
keep their original contracts.

The authority is the existing `curated_completion.check_curated_completion`
checker, previously reviewed with original completed-arm evidence. Its two
supported original dependency encodings remain separate: raw dependency-file
SHA256 values and the canonical JSON profile identity. Original QA, receipts,
profile, completion inputs and their bytes are retained, never converted into a
standalone/native envelope.

## Explicit inputs and binding

Eligibility, selection create/check, and the separate reviewed held-out route
accept `original_curated_index` and `expected_curated_index_sha256` (CLI flags
`--original-curated-index` and `--expected-curated-index-sha256`). Curated callers
must supply both. Other families reject these arguments. The external original
QA SHA remains independently required.

The adapter checks that supplied selected, BEST, LAST, CSV, summary and raw QA
paths resolve to their corresponding original index entries, with matching raw
bytes and SHA256. A byte-identical copy at a foreign path is rejected. Explicit
BEST/LAST role selects its own original weight snapshot; partial BEST coverage
cannot borrow LAST coverage. The original checker verifies the full completion
chain, optimizer/history, original source and cache/config/photometry provenance,
and each selected snapshot's full 5,700 annotation / 2,529 TRAIN-photo coverage.
All captured original inputs are checked unchanged across the operation.

The standardized outer `checked_freeze_eligibility` proof is explicitly marked
`explicit_original_curated_fd44_final_adapter` and retains the complete checker
proof under `original_curated_completion`. The checker action flags remain false: checking completion itself creates no
selection or inference permission. Newly generated proofs describe the separate
adapter and fresh-evidence requirements. The nested proof is retained exactly as
returned by the checker; historical proof files remain byte-unchanged. No original
proof file is relabeled or edited.

Only this known original family is admitted:

- Commit: `fd44cdfa83d1df9ccc42960df6d1d2018db29900`.
- Execution SHA256: `fcb186c98b812e9ead0fe19bec9f7e06063d31a7c7c020a785169b6c13c761ee`.
- Training SHA256: `2b7076759c8fab1594e5108bcbf348888007a94e06c4d21d4431dbfb31057747`.

## Fresh inference evidence

The fresh VAL producer accepts the same original-index pair plus an explicit
`checkpoint_role` / `--checkpoint-role`. Curated TTA requires the known training
source and the genuine checked original chain **before inference**. Selected path,
role and bytes must match the original index. The producer retains original input
snapshots and rejects changes before writing evidence.

Typed curated evidence includes the complete `original_curated_completion` header.
Final selection requires a freshly produced header equal in canonical JSON value
and type to its independently checked original chain. Historical reports without
this binding are not eligible for the curated final path. An untyped default-none
producer invocation retains its prior exploratory behavior but cannot satisfy the
curated final gate.

Existing full saved configuration, full-image photometry, backend/cache identity,
FOV, inference protocol and strict TTA/photometry descriptor checks remain in
force. Curated frozen contexts also compare canonical JSON types, including the
retained proof; booleans cannot alias integer fields. Original raw protocols of
existing families are unchanged.

## Validation scope

Tests use constructed tiny weights and synthetic metadata/images only. They cover
both original index schemas, own BEST/LAST role, exact original-proof preservation,
foreign paths/QA/index/CSV/weights/source/config/descriptor rejection, partial BEST,
all four TTA modes, fresh-proof binding, JSON-type aliases, and old-family gates.
The existing synthetic held-out fixtures test wiring without actual held-out data.
No operational checkpoint, actual dataset/GT, GPU, actual selection or TEST run is
part of this proposal. Actual curated adoption still requires independent review,
fresh source-compatible VAL evidence, an explicit model/protocol decision and a
separately authorized held-out stage.
