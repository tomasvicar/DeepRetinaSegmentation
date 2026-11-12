# Shared spatial tile blending: completed negative VAL result

Retain the existing probability tile blend for raw512 BEST50. Neither predeclared
alternative improved the measured result: both slightly reduce source-macro Dice,
with negative paired intervals. Dense vessel Dice changes are tiny and their
intervals cross zero; this is not equivalence evidence. The experimental operators
remain separate from the shipped release with 13 models, frozen selection and TEST.

All three branches use the SAME original checkpoint 521b…2cea, frozen 6d/f13 source,
FP32, rot4, threshold0.35, tile512/overlap64, and canonical 17a/00d prepared data.
The control averages weighted joint probabilities over tiles. The first alternative
averages weighted raw logits within each view before the same activation. The
hybrid takes V/disc/cup from that logit alternative and retains the final prepared
conditional artery/vein arrays from control. View ensembling always averages
inverse-aligned joint probabilities. This is a fixed-checkpoint inference contrast,
not retraining or an architecture experiment.

The actual one-TRAIN image-only probe established exact five-array control parity,
72 declared forwards and unchanged state/RNG/inputs. See the
[original numerical proof](../../experiments/dual_tile_blend/TRAIN-numerical-probe-result.json).
An independent review then found and fixed the missing third count list before
VAL; the numerical predictor function text stayed byte-identical (SHA 1b6ba404…24452).
The count producer added a thin progress callback. Fifteen synthetic cases passed.
No trained NN was repeated for that bookkeeping correction.

One shared full 770 VAL / 993 record pass completed at 17:16:26 UTC with worker/e3b exit 0,
22,000 actual shared forwards and718.699s driver time. All three reports were
made durable before postchecks. All six control fields, including every photo and
source, exactly equal original 0a5c. Same GT/record/source/valid-positive support,
model state, CPU/CUDA RNG, whole frozen source and input bytes were verified.
Actual 3 GiB / no swap / 2 CPU / cores 2,3 / Nice 15 / idle I/O were checked before Torch. This whole
validation time is not a latency benchmark or evidence that the extra streams
are free: they add transfers, activation and count processing.

Evaluation used existing canonical PNG/NPZ data without an evaluation cache,
matching0a5c. Cache 18c/db0 is training lineage only; it was absent on Old and was
neither fabricated nor silently enabled. No TEST data/evaluation, threshold/TTA
grid, new model selection, default, weight or product change occurred.

## Complete paired result

The authorized count-only operation completed once at 17:21:47.224 UTC: numeric systemd/SSH exit 0, independent local tool exit 0, observedtrue, no timeout. Actual service 6.692 s, CPU 6.655 s, peak 174.9 MiB, swap 0. Before importing NumPy/count-source checks the unchanged resource wrapper verified1 GiB / no swap / CPU 100% / core 4 / Nice 17 / idle I/O / MainPID / cgroup. Exact property JSON is preserved, not inferred from the requested unit.

Root-approved profile 354512d3 and all original NN terminal/parity/support/source checks passed. Control matches all six original 0a5 fields exactly; 770 photos / 993 records / 612 known groups are retained. Four predeclared paired comparisons contain all five heads, every source, macro, conditional/end-to-end AV, all negative point estimates, dense Dice/P/R and separateAVRDB counts. No NN was repeated and no model/GT/image/GPU access was used.

| Stream | Macro | V | A | Vn | D | C | Conditional AV | End-to-end AV | Dense V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared probability control | 0.822834097 | 0.762398925 | 0.762216696 | 0.811759302 | 0.942348197 | 0.884582493 | 0.851194311 | 0.720949448 | 0.751315957 |
| spatial_logit_alternative | 0.822735976 | 0.762353269 | 0.761990848 | 0.811626665 | 0.942190535 | 0.884620849 | 0.851196215 | 0.720599905 | 0.751287623 |
| segmentation_logit_control_AV | 0.822727173 | 0.762353269 | 0.761973799 | 0.811604490 | 0.942190535 | 0.884620849 | 0.851194311 | 0.720581247 | 0.751287623 |
| Original6808 | 0.667031974 | 0.770123676 | 0.689530320 | 0.697297832 | 0.894807839 | 0.435398951 | 0.748572508 | 0.630316828 | 0.761326809 |

Relative to shared control, full-logit alternative macro delta−0.0000981210 (95% paired percentile CI[−0.0001699551,−0.0000373848]); hybrid macro delta−0.0001069235 [−0.0001629013,−0.0000605747]. Both have small negative vessel/artery/vein/disc point differences; cup point difference+0.0000383565 has an interval crossing 0. End-to-endAV is lower for both; hybrid conditionalAV is exactly unchanged. Every source/head adverse remains in the original full analysis JSON, including 30 negative point estimates vs control for full-logit and 29 for hybrid; these tallies are descriptive, not significance counts.

Both alternatives share dense Dice0.7512876227 vs control 0.7513159568: delta−0.0000283341 CI[−0.0000742042,+0.0000144222]. Dense precision increases+0.0003945537 CI[+0.0003546697,+0.0004376811] while recall decreases−0.0005246897 CI[−0.0006345171,−0.0004344474]. Dense 87 photos / 55 groups / 8 sources excludes onlyAVRDB 20 positive-only support. Compared with original 6808 dense0.7613268085, the remaining gap is−0.0100391858 CI[−0.0144318598,−0.0065538815]. The aggregate macro gains vs original do not erase that vessel deficit.

All intervals use2,000 source-stratified paired draws/seed 42. Groups are recoverable aliases, not proved independent patients; intervals are unadjusted exploratory contrasts conditional on these fixed operators/models, not clinical, equivalence, training-seed or independent-quality evidence. Two alternative operators were predeclared; no minimum over a grid or formal winner, adaptive promotion, selection/default/product change or TEST claim is made.


## Reproducible saved-count evidence

[Full paired analysis](paired-analysis.json) retains all four specified comparisons:
each alternative against shared control and original 6808. Every source/head adverse
and the AVRDB positive-only domain are included. The exact original analysis SHA
is49a3860829aea0f0435b1becd506aff763687b2841c09b4ea97990d84d49d048.

Byte-exact inputs and receipts are indexed in [artifact-index.json](artifact-index.json):

- [Shared probability control](probability-control.json), [all5 logit alternative](all5-logit.json), [hybrid](hybrid-control-AV.json).
- [Original6808 VAL](original6808-validation.json), [prior0a5c VAL](prior0a5c-validation.json), [six-field control equality](control-parity.json).
- [All point/source results](point-summary.json), [NN observed exit](NN-observed-exit.json), [NN actual resource properties](NN-resource-properties.json).
- [CPU analysis observed exit](CPU-analysis-observed-exit.json), [CPU actual resource properties](CPU-analysis-resource-properties.json), [root CPU byte review](root-independent-review.json), [root NN byte/count review](NN-root-independent-review.json).

Root independently rehashed all 18 collected NN originals and all seven CPU analysis
originals, recomputed allfive pooled counts/Dice for allthree branches and checked
actual exits/resources. No repeated NN or statistical computation was used for
this collection/review. Original report bytes retain their identities. Experimental
flat reports are not ordinary freeze/eligibility evidence; no historical report
is retagged. The unchanged comparison/dense helpers remain in the research repo.

These exploratory VAL variants were tested after the earlier frozen TEST had
already been reported. They have no new independent held-out verification. Prior
historical model exposure/preprocessing and fixed 25measurement qualifications
remain; original 6808 is a different complete pipeline, while the three blend
branches share weights and protocol. This experiment does not resolve the wider
vessel-quality objective.
