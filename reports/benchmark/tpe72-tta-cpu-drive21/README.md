# TPE BEST58 TTA: one resident CPU quality/time example

Exactly one authorized CPU run completed with observed child and outer exit0 in
678.46s under a900s owned cap. It measured **16 complete operations** in fixed order:
original four-network pipeline, then TPE `none`, `rot4`, `dihedral8`, each with one
warmup and three measured calls. There were no retries, extra NN calls, GPU use,
GT/TEST arrays or additional photographs. This is one DRIVE TRAIN raw photograph
with its official FOV, not a quality assessment or general throughput benchmark.

| Pipeline, same run | Three measured samples (seconds) | Resident median | Original/median |
| --- | --- | ---: | ---: |
| Original four networks |59.59455,59.80121,59.33062|59.59455|1.0000|
| TPE none |8.80692,8.74876,8.70102|8.74876|6.8118|
| TPE rot4 |33.19967,33.49377,33.01503|33.19967|1.7950|
| TPE dihedral8 |66.34664,64.17571,65.53144|65.53144|.9094|

**Dihedral8 was about10% slower than the original on this photograph.** Rot4 was
about1.80× faster than original, but3.79× slower than none; dihedral8 cost7.49× none
and about1.97× rot4. None's8.75s must not be attached to8-view quality. These are
scoped observations, not a default choice or evidence that rot4 and8 have equal
quality. [Separate exploratory VAL770 TTA results](../../tpe72-inference-tta-val)
contain the quality intervals and adverse RIMONE cup results. That experiment's
observed8-minus4 macro difference is .00079856; no quality-equivalence test is
created by this CPU timing run.

## Unchanged arithmetic and output identity

The executed public benchmark SHA is
`3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae`.
Source`1f723ad92898fad8b04a6ef8b39065f021f52a9c` remained clean and byte-identical.
Only the benchmark module's local `predict_probabilities` binding was wrapped with
an explicit per-mode keyword. `unified_operation`, full-image preparation,
decoding and restoration arithmetic remained unchanged. Actual forwarding was
four calls per TPE mode. Original arithmetic/reference selection stayed unchanged.

The last masks for all four groups were finite `bool[5,584,565]`. Shape/finite/
hash/count checks ran **outside measured clocks**, retaining the last output;
no extra forward was used for verification. The none last-mask SHA
`32590cf934ec802bf7010933361ecb06fcc078aacc178252f6a8811a904811c3`
exactly matches the prior completed CPU none benchmark. The original last mask
also matches its prior hash. All masks/digests/foreground counts are in
[results.json](results.json); they establish identity, not accuracy.

## Runtime, checkpoint and reference scope

One logical CPU affinity4, intra1/inter1/OpenCV1, nice17/idle IO, CUDA hidden and
uninitialized, native Python3.12.13/Torch2.6.0+cu124. Available RAM was5.6165GiB
before startup, exceeding the4GiB admission minimum. Exact peak whole-process
RSS was **2,967,872KiB (about2.83GiB)**. This is the entire measured process peak,
not a separate per-mode peak or a universal memory bound.

The unchanged TPE BEST58 checkpoint SHA is
`20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`.
It has1,965,637 parameters versus5,943,312 across the four original networks.
Its serialized model tensors are FP32 (7,874,468bytes); BF16 is **training
metadata**, not this inference precision. The checkpoint already contains no
optimizer/training state. No compression, quantization, optimizer stripping or
weight transformation was applied; its actual ZIP members are stored uncompressed
and its8,894,002byte artifact hash remained unchanged. All input/FOV/reference/
checkpoint and runtime-source hashes remained unchanged.

The original composed reference SHA starts`528233fe`: calibrated provided-FOV AV,
dedicated raw Disc0 and same-forward universal cup clipped inside that disc. Its
provenance is distinct from the flat`d14` VAL comparison; do not silently treat
those complete evidence records as identical. The original and none timing
reference is measured **within this run**, rather than borrowing an earlier median.

Timings include resident CPU preprocessing, inference, decoding and original-frame
mask restoration. They **exclude file IO and model loading** and do not measure
CLI startup. Fixed order, warm caches, three samples, one selected geometry and
external system load limit interpretation. No new original/VAL comparison,
clinical generalization, held-out result, final selection, release or default
promotion is established. Original historical exposure remains unresolved.

[Runtime summary](runtime-summary.json) and
[observed producer terminal](producer-terminal.json) retain actual calls, precise
metrics and whole private artifact SHA references. Operational paths, physical GPU
UUIDs and photograph identifiers are excluded. These sanitized projections do not
replace original private launch/runtime/exit proofs.
