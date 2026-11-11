# Bounded Sobel residual investigation,2026-11-11

This branch changes **no production preprocessing code, model weights, aliases,
APIs, golden tolerances or default profile**. It records a research candidate
whose stage-level improvement is measured, together with remaining counterexamples.
Training6b3 and original-evaluation193662c sources/processes/data were untouched.
No model forward, GT evaluation, TEST score, GPU action or VM was run.

Only public MATLAB `imgradientxy`/`imfilter` APIs and project-owned code were used.
No proprietary toolbox source was inspected or copied. The one new MATLAB call
was a small batch of synthetic single matrices, singleCompThread, CUDA hidden,
nice15, affinity CPU0,90-second timeout. Subsequent Python replays used the same
CPU/thread/priority constraints. Existing real TRAIN/image-only reference arrays
were read, without raw ownCloud writes or new real-image arrays in Git.

## New concrete finding

The old discrepancies are spatially structured. All measured old Sobel errors
are in the first/last columns and a terminal row block. HRF's1350 gx and801 gy
errors were entirely border pixels. DRIVE H1222 leaves a6-row tail; DiaRet H1005
leaves13 rows; H128 synthetic inputs leave16 rows. The observed candidate uses
`start=((H-1)//16)*16`; this is an experimentally observed R2026b CPU path model,
not a claim about proprietary implementation internals or another MATLAB release.

A new random oracle batch at H3/15/16/17/31/32/33/64/65 confirms public
imgradientxy and explicit same-kernel imfilter are bitwise identical in all nine
cases. For evidenced frames>=16×16:

- Main body keeps the existing column traversal.
- Left-column gx uses reversed column traversal.
- The terminal gx block follows a separable smooth-then-difference expression.
- Terminal *interior* gy follows difference-then-smooth.

These shape/traversal rules do not depend on image content, dataset identity,
error locations read from the oracle, epsilons or CLAHE histogram snapping.
The research candidate leaves unsupported tiny/pre-vector frames unchanged.
It is not installed in `_sobel_single` or any public runtime route.

## Measured results

On the existing DRIVE/HRF/DiaRet and six synthetic distance arrays, candidate gx
is bitwise exact in all nine. Gy remains imperfect:

| Case | Old gy unequal | Research gy unequal |
|---|---:|---:|
|DRIVE|3938|546|
|HRF|801|498|
|DiaRet|4313|500|

The new synthetic random cases also retain gy boundary errors. H3×W5 is an
explicit unresolved tiny-frame counterexample; H15×W19 remains an unchanged
fallback in the guarded prototype. These are preserved by tests, not rebaselined.
The exact origin/order of gy left/corner accumulation remains unresolved.

One existing DiaRet image-only/TRAIN reference was replayed with the **same
MATLAB-resized RGB and processing FOV**, changing only the process-local Sobel
function object in an isolated copied package:

| Stage/output | Old | Research candidate |
|---|---:|---:|
|gx unequal|6706|0|
|gy unequal|4313|500|
|V-before maximum error|0.0016551432|0.0004087870|
|Normalized RGB maximum error|0.0015219156|0.0000430089|
|Stored12 unequal values|265|3|
|Stored12 maximum error|7|1|

A second pair using Python bicubic RGB with that **same reference processing
FOV/calibration** gives the same stored12 results; resize input maximum error was
7.77e-16. The historical full-automatic pipeline counterexample74165/max271 is
not repeated by this isolated-FOV experiment. It is not overwritten or claimed
resolved. This run does not reevaluate automatic FOV/calibration composition,
the flat-image Gaussian/bin-boundary counterexample, network logits or labels.

`candidate-comparison.json` and `candidate-raw-comparison.json` retain every
stage/error, including continuous residuals. The positive result is a narrower
operator/photometry improvement, not complete MATLAB parity, a production
recommendation, historical training provenance or improved segmentation quality.

## Reproduction

The small new synthetic fixture contains random operator inputs/outputs only.
Its generator, MATLAB version and original MAT-file SHA are preserved under
`tests/reference/matlab_sobel/`. The raw synthetic MAT remains outside Git.

```sh
# Python replay, no MATLAB needed:
uv run python reports/legacy/matlab-residual-20261111/compare_random_probe.py \
  --fixture tests/reference/matlab_sobel/probe.npz --output /new/probe-replay.json
uv run pytest tests/test_matlab_sobel_research.py

# Licensed public MATLAB oracle; execute from a NEW output directory:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0 nice -n 15 timeout 90s /path/to/matlab \
  -nojvm -nodisplay -singleCompThread \
  -batch "run('/repo/reports/legacy/matlab-residual-20261111/generate_operator_probe.m')"
```

`diagnose_sobel.py` and `probe_accumulation.py` consume existing external
`sobel_operator_goldens.mat` and `diaret_photometry_stages.mat`. For one complete
photometry replay, `compare_candidate.py --root /external/oracles --output
/new/report.json` also needs `source_auto_chain.mat`, generated by the already
retained source-auto oracle scripts. Its scope is deliberately the reference
RGB/FOV stages and one known DiaRet case; it does not fabricate a full-auto test.
Use the copied isolated package on PYTHONPATH, never replace a live frozen source.

Nine tiny CPU regressions passed0.09s. Ruff/format checks pass for the new Python
files. Runtime versions and source/artifact hashes are recorded alongside them.

## Next bounded step

Determine gy left/corner accumulation with a small operator-derived grouping
probe (not720 permutations), and test very narrow/small frames independently.
Recheck complete raw automatic preparation and DRIVE/HRF before considering any
runtime activation. Any later activation needs its own reviewed numerical gate;
this research branch deliberately leaves all shipping behavior unchanged.

## Bounded followup: terminal Gy and the full-auto first divergence

The one permitted fresh public-operator batch (`generate_followup_probe.m`,
seed61112, 14 tiny/random shapes) distinguishes direct-small and terminal-vector
behavior. The terminal horizontal filter associates its replicated left edge as
`left + (2*middle + right)` while its interior/right edge uses
`(left + 2*middle) + right`. `candidate_followup.py` applies this only in the
already evidenced terminal block. It exactly matches both gradients at16x16,
but tiny/direct shapes and the vector-body Gy left column remain counterexamples.
See `followup-metrics.json`; no production Sobel change is proposed.

Replaying the **actual complete** retained DiaRet automatic chain reproduced the
historical74165 unequal stored12 values/max271. Original processing FOV and
radius717 matched, but the resized FOV differed at TWO pixels. The previous
isolated photometry comparison had deliberately supplied the reference resized
FOV, so its improvement did not establish complete-chain fidelity.

`nearest_research.py` preserves the one-based inverse coordinate expression
already used by the bicubic resize, then rounds nearest by floor(position+.5).
There is no epsilon, threshold snapping or image-specific branch. Algebraic
reassociation matters at exact half-pixel ties. All24 retained synthetic nearest
fixtures and all5 source-auto resized FOVs match MATLAB with this expression;
the current production expression misses two DiaRet pixels. See
`nearest-comparison.json` (real photos/masks stay outside Git).

`full-auto-nearest.json` records the SAME raw DiaRet input with no labels/model
forward: current74165/max271; research Sobel alone73865/max271;
nearest-only265/max7; nearest plus research Sobel3/max1. The remaining3 stored12
values are unresolved; this is measured improvement on one complete chain,
not a universal parity certificate. The production module is unchanged in this
research commit. Research tests:37 passed0.16s, Ada CPU0 nice15, CUDA hidden.
The original CPU validation and GPU training sources were not modified.
