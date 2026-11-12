# Curated saved-TEST comparison admission

This corrects only `scripts/compare_reviewed_test.py`, starting from HPOc0432b9.
It does not change the frozen held-out evaluator, final selection, inference
operator, checkpoint, threshold/TTA, counts, or any running operation. It grants
no permission to evaluate TEST. The helper compares externally pinned, already
saved original and unified TEST reports using their counts and metadata.

The existing selection gate already supports genuine `curated_fd44` with rot4.
The comparison helper still restricted TTA to two source72 families, and then
expected an `original_completion` standalone/native outcome. Genuine curated
eligibility instead has `adapter_kind` and `original_curated_completion`, with
its original curated index, receipt, QA and independent CPU-check proof.

The explicit curated branch now binds that original proof to the frozen role,
checkpoint/epoch/steps/score, full raw configuration, both own TRAIN-coverage
views, source/runtime and original QA/index/receipt/BEST/LAST/CSV/summary byte
identities. It requires its actual separate PBS completion check and rejects a
fabricated native envelope. Existing generic source/config/data/coverage checks,
external whole-selection/report SHA pins, exact reviewed TTA descriptor, split
inventory and original pipeline policy checks remain active. No generic family
fallback or reclassification is introduced.

Actual verification (CUDA hidden, OMP/MKL/OpenBLAS1, existing cached runtime,
`PYTHONPATH=src`):

```
python -m pytest -q tests/test_reviewed_test_comparison.py
60 passed in 5.49s
```

Observed returncode0. Synthetic saved-count fixtures cover curated rot4 for both
training_best and fixed_terminal60, immutable inputs, and rejection of unknown
family, missing eligibility/curated proof/index, native-family substitution,
wrong receipt/source/own coverage and altered TTA protocol. Existing family,
role, external pin, source, canonical inventory and TEST-versus-VAL cases pass.
No real TEST counts, model, GT pixels, inference or GPU were used in these tests.
Ruff format/check and `git diff --check` also passed.

Separately, the embedded-provenance function accepted the **existing frozen
selection** SHA`caef7716a5656a151cf6c0ce1e9b8cd694aa9f13620cef36f14abcd0a6368e80`
with actual exit0. This read one original metadata JSON only, not any TEST report
or checkpoint; `torch` remained absent from `sys.modules`. This proves structural
interoperability with that genuine chain, not completion of held-out evaluation
or a quality result. A future full saved-report comparison still checks all
original inputs and reports under the supplied external pins.
