# Product optional-TTA compatibility checks

This isolated port starts from product8b99b64 and transplants the optional
joint-probability TTA math/API from3548264/1f723ad. It adds `none`, `flips2`, `rot4`
and `dihedral8` to the unified product prediction/VAL CLI. Model weights,
registry aliases, default mode, training configuration, original-model inference,
BF16 compatibility, native AV support fix and prediction-output guard are retained.
No experimental photometry/cache feature is ported.

## Actual fixed-checkpoint prepared-input parity

The genuine TPE72 BEST58 checkpoint SHA
`20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`
was safely loaded under three separately isolated source imports: product8b99b64,
joint reference1f723ad, and this product port. The same deterministic asymmetric
RGB01 input `[3,35,49]` and explicit tiny32/8 tiling were used on CPU1,interop1,
affinity4,nice17,idleI/O,CUDA hidden. No photograph, GT, VAL or TEST arrays were
read. This is functional numeric compatibility, not a quality or latency benchmark.

| Compared output | Bitwise equality | Max absolute error |
|---|---|---:|
| Product port default `none` vs product8b99b64 |true|0|
| Product port `none` vs joint1f723ad |true|0|
| Product port `flips2` vs joint1f723ad |true|0|
| Product port `rot4` vs joint1f723ad |true|0|
| Product port `dihedral8` vs joint1f723ad |true|0|

[prepared-forward-parity.json](prepared-forward-parity.json) binds exact commits,
checkpoint/input/output hashes and the comparison scope. Per-source reports retain
actual source-module fingerprints, shape, dtype, finite output, unchanged
parameters/buffers and measured software. Source imports used distinct Python
processes and explicit `PYTHONPATH`; checkpoint and model tensors remained
unchanged. Generated synthetic probability arrays stay outside Git. Tiny invocation
seconds in the reports must not be treated as image latency or full-size speed.

`compare_prepared_forward.py` reproduces one source's prepared-input checks with
explicit `--checkpoint`, `--source-root`, fresh `--output` and `--modes` arguments.
Select that source's `src` with `PYTHONPATH`, use the locked existing Python runtime
and the exact pinned checkpoint. Repeat against the two pinned reference checkouts
to compare numeric output hashes. The script does not read a corpus or score quality.

## Focused CPU regression scope

148 unique focused tests passed across TTA, unified CLI, mixed precision,
prediction-output admission, native legacy training and existing unified behavior.
This includes inverse transforms of odd asymmetric rectangles; default-none exact
arithmetic/RNG; joint A/V vessel-confidence weighting and zero-vessel fallback;
patch and whole-view geometry; once-only raw preparation/probability restoration;
FP32 inference under BF16 checkpoint metadata/ambient autocast; no parameter or
gradient mutation; unchanged GT/source support in synthetic VAL; CLI forwarding;
rejection for original/pilot families; and occupied output preservation before load.
Ruff passed. No GPU, new quality evaluation or TEST was executed.

The first selected run passed126 cases; nine new fixture namespaces lacked required
CLI fields, and seven existing later tests hit the shared temporary filesystem's
quota. The fixture was corrected and the failing16 passed on dedicated home-backed
scratch. One setup attempt had a missing parent directory and executed no tests;
that setup was corrected. Six newly added namespace/whole-geometry cases passed
separately. These attempts are explicit in the verification notes; they are not
hidden numerical failures or evidence of a source compatibility problem. Only the
confirmed owned synthetic scratch was removed; no other agent's files were touched.

The only deliberate edge-case extension beyond the reference is that non-default
TTA disables an enclosing autocast even for FP32-trained metadata. The default
`none` keeps the previous precision-context behavior. Actual BF16-trained BEST58
outputs still match the reference in all four modes exactly.

See [UNIFIED_TTA.md](../../docs/UNIFIED_TTA.md) for usage and the separately measured
TPE BEST58 VAL study, including source regressions and cost. That existing study
is not a newly rerun product evaluation, a global model improvement claim, or
held-out TEST evidence. No original default-none completion record is retagged
as certification of a different inference protocol.
