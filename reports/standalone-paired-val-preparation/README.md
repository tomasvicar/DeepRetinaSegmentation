# Reusable standalone paired VAL comparison — preparation only

No vascular4 or future standalone source72 result was available or compared when
this helper was prepared. This branch contains no actual new scientific result,
checkpoint, GT array, GPU evaluation or TEST score. Existing per-run comparisons
and frozen training sources remain unchanged.

`compare_standalone_val.py` consumes one externally pinned candidate and its
**original full standalone QA** from verifier 4b568, original protocol and original
supervisor terminal. It does not accept a pool receipt, a rewritten summary or
another family as completion proof. The default final-comparison route is strict:
both the completed run and the selected checkpoint must have all 2,529 photos and
all 5,700 eligible annotation contributions. A BEST missing even one annotation
fails; inspect that checkpoint separately as a clearly named diagnostic, without
using this final-comparison CLI or weakening its gate.

The supported contracts are exactly vascular4/source0b (protocol 29d3) and a
future source72 BF16/workers8 standalone protocol whose **actual original bytes**
are externally reviewed and pinned. Source commit, execution and training hashes
are fixed in code; source72's future protocol SHA is not invented here. Original
QA must embed the exact protocol/terminal text and matching byte hashes. Role
BEST or fixed_terminal60 binds its own checkpoint SHA, epoch, step count and own
coverage. Fixed terminal means actual epoch 60, never a macro-BEST substitute.
Inference remains scalar threshold 0.5, FP32, tile 512 / overlap 64 and cup&disc.

The helper reuses existing `unwrap`, `check_inventory`,
`validate_source_aggregates`, manifest `inventory`, the six-ratio point calculation
and `deepretina.comparison.compare_reports`. Source/record attribution and GT
positive/valid support remain exact per photo; source integer headers are
reaggregated. All 770 VAL photographs and 612 known split groups are required.
Paired source-stratified bootstrap uses 2,000 draws and seed 42, with the same
source/head/conditional calculations as existing comparisons. Nothing decodes an
image/mask or loads a model. Manifest TEST rows are identity metadata only.

References are a small exact pinned set: original d14, control terminal E60,
vascular2 BEST E59 and vascular2 terminal E60. The requested subset is explicit;
there is no arbitrary report family/alias fallback. LAST60 vs LAST60 gets an
explicit matched-epoch label, which **does not** imply a causal isolated recipe
comparison. BEST vs another BEST or LAST, or a historical original reference,
retains a different-role/original label. Backend/hardware/seed and development-VAL
qualifications remain visible. Every result denies model selection and TEST
authorization. The original six-ratio objective is a point estimate; its minimum
has no bootstrapped CI claim. Per-head/source intervals are descriptive and not
multiplicity-corrected; opaque patient identities remain a qualification.

## Deferred input and command

Copy `inputs-template.json` into a new private evidence directory. It is marked
NONEXECUTABLE because the candidate, QA, protocol and terminal byte hashes do not
exist yet. After actual completion/fresh VAL/root review, create a typed
`reviewed_standalone_paired_val_inputs` with the exact absolute or relative asset
paths and actual SHA256 strings. Set the explicit role and source contract. Bind
the entire JSON to a separately supplied SHA256; do not generate missing receipts
or populate a synthetic result. Available reference hashes in the template are
actual existing artifacts, while candidate placeholders are null.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src nice -n 15 \
  /path/to/existing/venv/bin/python scripts/compare_standalone_val.py \
    --inputs /path/to/root-reviewed-inputs.json \
    --trusted-input-sha256 ACTUAL_EXTERNAL_INPUT_SHA \
    --output /path/to/fresh-comparison-directory
```

Outputs are role-specific `best-vs-...json` or
`fixed_terminal60-vs-...json`. Existing outputs are never overwritten, and input
bytes are rechecked after comparison. Publication should include these small
count-derived reports and their input hashes, without private operational paths
or original private QA contents. Do not use this to choose a final model from
TEST or to imply that a higher macro score proves gains on every vascular head.
