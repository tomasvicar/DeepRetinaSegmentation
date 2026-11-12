# Actual E1 gate passed; no complete model or quality promotion

Both training and separate CPU state check returned0, as did the external
observer and root launcher. Actual E1 finished15:01:34UTC on12November2026.
It delivered8192samples/2048updates, all2529TRAINphotos and3401contributing
annotations of5700 eligible. Do not borrow finalcoverage or a later epoch's state.
The source72/config/data remained unchanged, scratch lineage is independent of
historical teachers, and the original60epoch cosine horizon/RNG/used Adam states
remain resumable. No TEST evaluation or trained-weight release occurred.

Actual TRAIN90.989s +native per-epoch770-photo/993-record VAL75.760s; whole
operational run226.792s including setup/checks. This native source72 validation
is not the future common-source final validation/selection artifact. Cgroupmemory
peak4,514,189,312B, swap0, CPU939.527s; summed per-process RSS doublecounts shared
fork pages and is not physical memory usage. The actual8GiB cap stayed intact.

The protocol's conservative full-arm forecast is14,132.612s (~3h56m), below the
12h resource ceiling. It is a forecast, not completed60epochs. A separate bounded
resume to27 is being prepared to retain a stable diagnostic checkpoint; no quality
pruning fromE1/E27 and no automatic continuation are implied. Full60 remains the
intended controlled quality comparison. Frozen release models/results are unchanged.

[Machine-readable operational numbers](E1-operational-result.json) bind the original
private proof digest; full unit/identity/source/exit artifacts remain in the private
operations repository.
