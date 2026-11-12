# TPE7 LAST60: recovery versus own controls, remaining original vessel gap

This report evaluates **fully annotated LAST60 only**. TPE7 trained60 epochs,
122880 updates/491520 samples,2529TRAIN photos/5700 annotation IDs across21sources.
Original supervisor/outer/generic4bQA all returned0; LAST ownsall5700, minimum62
photo visits. Original BEST46 owns5699, missing
`riga_BinRushed3_image8_observer3_disc_cup`: it is explicitly ineligible and was
not substituted, evaluated here or promoted. Subsequent one-case fresh LAST
VAL completed with actual child0/92.731s and independent outer0,770VALphotos.
No TEST, new default or model promotion follows.

All primary models are terminal60, native FP32/NONE512/64/threshold0.5,25ppd raw.
TPE5/TPE6/TPE7 full source/runtime, data and inference dictionaries match:
frozen72 training3725/execution347f, evaluationb269, intra8/inter1. Their exact
original full training configs and semantic identities are retained; learning
rate/weight decay/vascular sampling differ, so these are not single-factor or
multi-seed causal claims. Whole original report/QA/protocol/checkpoint SHAs are
preserved in projections/provenance. No original wrapper was retagged.

| Metric | TPE5 LAST60 | TPE6 LAST60 | TPE7 LAST60 | Original182 |
|---|---:|---:|---:|---:|
| Vessel pooled Dice |.760802772|.756283114|.764368516|.770123400|
| Artery pooled Dice |.732653668|.701746604|.744788805|.689521830|
| Vein pooled Dice |.790739644|.768438394|.796321293|.697286369|
| Disc pooled Dice |.935337602|.938609946|.934706026|.894807908|
| Cup pooled Dice |.874872420|.876284250|.873188578|.435398984|
| Conditional AV |.821754997|.805443737|.833025294|.748566416|
| Macro dataset mean Dice |.808832542|.801973971|.815309799|.667029585|

Full-precision counters, source counters and values are in JSON. Worst six
point ratio versus exact original182 is **0.9925273219641642**, bottleneck vessel.
Other pooled heads/conditionalAV exceed original point estimates, but vessel
remains below original with negative paired interval. Macro/cup/AV gains do not
satisfy the all-task goal or modify the frozen objective/reference.

## Dense-vessel contrast

Existing reviewed math,2000draws/seed42, source-stratified reviewed split groups.
Dense removes **only known positive-only AVRDB**, leaving87vessel photos,
55known groups/8sources. Full vessel support is107photos/75groups/9sources;
AVRDB20photos/20groups has structural FP0/precision1, not dense specificity.
This supplementary domain does not redefine the benchmark/objective.

| Reference | Dense reference | TPE7 dense | Delta | Paired95%CI |
|---|---:|---:|---:|---|
| TPE5 LAST60 |.738511536|.747852679|+.009341143|[.004943520,.014531637]|
| TPE6 LAST60 |.742841480|.747852679|+.005011199|[.003135341,.007211806]|
| Original182 |.761326607|.747852679|−.013473928|[−.018460907,−.009072952]|
| raw512 LAST60 cross-context |.747051432|.747852679|+.000801247|[−.001080852,.002959754]|

Dense performance recovers against both own controls but **remains below the
original**. All eight dense vessel sources have positive point deltas versus
TPE5; this is not simultaneous all-source significance. Against original, vessel
intervals remain below0 on ARIA,CHASEDB1,DRHAGIS,HRF,RITE; IOSTAR point is lower
but its interval crosses0. Dense bootstrap uses55groups, not612full-VALgroups.

## All heads and adverse findings

The exact paired JSON files retain all heads, every source and conditional AV,
using shared2000/42 draws across612known groups/full770VAL. TPE7 vsTPE5 global
vessel delta+.003565743 has interval[−.000670509,.008759845], artery+.012135137
[.004586916,.019462092], vein+.005581649 [−.000778670,.011436204],
disc−.000631575 [−.007954961,.004995583], cup−.001683842
[−.015731665,.009223895], conditionalAV+.011270297 [.004886401,.016914401].
Macro+.006477257 [.001547055,.013240684] is not universal improvement.

Versus original, global vessel delta−.005754884 has negative interval
[−.010285069,−.001552774]; the other pooled head/conditionalAV intervals are
positive. Versus TPE6 vascular intervals are positive, but disc/cup points lower.
All adverse source/head/AV point results are retained (negative point/wholly
negative interval counts: TPE5 5/3;TPE6 7/2;original8/5;raw51217/8).
Important own-control adverses include AVRDBvessel/vein and G1020cup versusTPE5,
and ONHSDdisc/RIMONEcup versusTPE6. Intervals are exploratory and unadjusted for
multiple comparisons; no clinical or arbitrary noninferiority margin is claimed.

raw512 is explicitly cross-context: fd44training/f13evaluation/different GPU
runtime versus72/b269/native matched context. Same GT support permits comparing
specified pipelines, not isolating patch-size causation. TPE7's conditionalAV
is lower thanraw512 by−.011533406 [−.019057881,−.004494897], while global vessel
is higher and dense vessel interval crosses0: the pipelines are Pareto
incomparable. No favorable macro-only ranking or BEST/TTA/threshold substitution.

## Artifact scope

Paired/dense/adverse JSON files are **byte-identical numeric copies** of completed
private analysis20e689a; their SHA pairs are recorded. Aggregate/source counters,
full native configurations, roles/epochs/source+data identities and provenance
are compact scientific projections, clearly **NOT original evaluation, QA or
eligibility-gate wrappers**. They cannot replace externally pinned original
producer artifacts. Full original per-photo reports remain SHA-pinned; aggregate
projections alone are not sufficient to independently reconstruct bootstrap draws.
No pixels, weights, operational absolute paths or GPU UUIDs are exported. No
new analysis, inference, source/registry/docs changes or TEST was performed.
Repeated adaptive VAL use, fixed checkpoints, historical original exposure and
preprocessing/backend residuals qualify these contrasts. Groups preserve known
aliases; they are not proven independent patients or training-seed uncertainty.
