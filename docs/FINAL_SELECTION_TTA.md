# Explicit TTA in reviewed final selection

This implementation prepares an eligible evaluation route; it performs no actual
model selection, validation forward or held-out TEST. Historical experimental
TTA reports, including source `1f723ad`, remain ineligible. They must not be
relabelled or assigned a newer source identity.

The optional modes are `none`, `flips2`, `rot4` and `dihedral8`. `none` keeps the
existing arithmetic, report kind and protocol schema: no `tta` field is added.
Non-default modes require a freshly produced `unified_tta_validation_evidence`
report, and explicit `tta` in both the protocol and actual metrics. The descriptor
pins ordered full-prepared-image transforms, inverse alignment, FP32 arithmetic,
mean **joint** vessel/artery/vein probabilities followed by recovery of conditional
A/V, and single restoration of original geometry. Threshold, tiling, activation,
A/V decoder, cup containment, canonical geometry and support remain independently
bound by the existing inference protocol. BF16 training metadata never enables
BF16 TTA inference, including under an ambient autocast context.

Only `standalone_72` and `pool_72_v3` can request non-default TTA in the final gate.
They retain their genuine, separate original completion verification paths:

- Standalone: externally pinned original completion QA, reviewed training
  protocol and original supervisor terminal; actual full60/optimizer/scheduler,
  CSV and selected-role coverage checks remain unchanged. For the first completed
  TPE candidate these are original QA `9efabed…21204d`, training protocol `9177511b7ee6895cbb3e9eea5bd75272d7e86e2c2af764f44c1600faa1fc0c65`
  and terminal `158cff…376e6d3`; use the full authoritative original digests.
- Native V3 pool: genuine original COMPLETE receipt, original task/lease,
  protocol/envelope/profile and original input index through native V3 validation;
  no fabricated standalone terminal or QA. Original index bytes stay unchanged.
- Both roles remain explicit: `training_best` and `fixed_terminal60`. The selected
  weights' own photo and 5,700-annotation coverage must pass; another epoch's
  full coverage cannot qualify a partial BEST.

Source0b, curated photometry families and untyped aliases are not implicitly
admitted. TTA changes inference only, not the historical training protocol or QA.
No completion-family allowlist was broadened or verifier changed in this work.

## Fresh evidence and frozen decision

After the complete implementation source is frozen and separately approved, the
VAL-only producer supports, for example:

```bash
uv run python scripts/unified_validation.py \
  --checkpoint /immutable/selected.pt --manifest /canonical/manifest.json \
  --output /new/fresh-rot4-val.json --device cpu \
  --threshold 0.5 --tile-size 512 --overlap 64 --tta rot4
```

This is a future operation, not one executed for this change. The fresh report
binds actual checkpoint bytes/epoch/steps, original training configuration/source,
canonical contents, exact full inference descriptor, current implementation and
installed runtime identity. It is not a substitute for completion QA.

`unified_selection.py create/check` takes the same explicit `--tta` in addition to
all existing role/family/original QA and external digest arguments. A report from
another mode, implementation/runtime, checkpoint, support, threshold or tile policy
fails. The externally reviewed schema2 selection digest pins the descriptor; no
selection artifact is produced automatically by validation.

Stage2 takes **no TTA override**. It reads the exact descriptor from the externally
pinned selection, rebuilds the full original eligibility/source/data lease and
inference protocol, then dispatches that exact mode to the evaluator. Unknown,
partial, differently ordered or experimental descriptors fail before scoring;
there is no fallback to `none`. The metadata-only final comparator also validates
the exact descriptor in the saved metrics. A new implementation/source requires
fresh evidence even for `none`; preserving schema does not retrofit old reports.

Before any real final evaluation, remaining work is: approve/freeze this source,
produce actual fresh VAL evidence under the chosen mode, verify genuine original
completion inputs and selected coverage, explicitly review the mode/quality/cost
tradeoff and freeze the decision. Then a separately authorized stage2 can run
under the exact frozen source/runtime/protocol. TEST remains untouched here.

Hashes/unsigned reviewer records bind bytes and decisions; they cannot prove
previous blindness or authenticate a reviewer. Existing scientific limits of
adaptive VAL selection, original historical exposure and patient identity remain.
