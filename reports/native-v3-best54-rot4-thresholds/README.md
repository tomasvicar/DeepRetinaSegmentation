# Completed rot4 scalar threshold VAL ablation

All770 VAL photos and eight thresholds completed with actual externally observed exit0 and child exit0, no timeout/error. Science689.185s; stage690.877s. Scientific JSON files are byte-preserved exports; private operator paths/logs are excluded. See completion-provenance.json for immutable source/input/output pins and the retained failed first attempt.

The exploratory worst-six-ratio grid rule chose **0.40**, not the largest macro (0.30). Default0.5, model registry and final selection remain unchanged. No TEST or new forward was performed by the auditor.

| Threshold | Worst-six ratio | Macro source/head Dice | Pooled vessel Dice |
| --- | ---: | ---: | ---: |
| 0.25 | 0.992977045 | 0.823178403 | 0.764714857 |
| 0.30 | 0.996639712 | 0.823432214 | 0.767535563 |
| 0.35 | 0.998537353 | 0.823016473 | 0.768996981 |
| 0.40 | 0.999074032 | 0.822148867 | 0.769410290 |
| 0.45 | 0.998438278 | 0.820921164 | 0.768920681 |
| 0.50 | 0.996576835 | 0.819411772 | 0.767487140 |
| 0.55 | 0.993573696 | 0.817385247 | 0.765174353 |
| 0.60 | 0.989377244 | 0.814860184 | 0.761942567 |

## Selected0.4 versus same-forward0.5

| Measure | Delta | Paired95% interval |
| --- | ---: | --- |
| artery pooled Dice | +0.010076338 | [+0.009304969, +0.010996820] |
| cup pooled Dice | -0.002733776 | [-0.004043001, -0.001529427] |
| disc pooled Dice | -0.000422663 | [-0.001352275, +0.000413628] |
| vein pooled Dice | +0.007726408 | [+0.007120730, +0.008311052] |
| vessel pooled Dice | +0.001923150 | [+0.000349697, +0.003582857] |

Macro rises+.002737095 CI[+.002065546,+.003694935]. Cup declines−.002733776 with entirely negative interval; pooled disc declines−.000422663 with interval crossing zero. Disc macro declines−.001622262 CI[−.002758383,−.000405732]. Vessel macro declines−.000407416 CI[−.001744134,+.000936894] despite pooled-vessel gain. Conditional A/V remains exactly0.8324716596 (2,186,307/2,626,284) at every threshold; changes to gated A/V scores are detection/gating effects, not learned class discrimination.

Against strongest frozen original182 reference, selected vessel0.769410290 remains below0.770123400: delta−.000713109, CI[−.007253184,+.006346865]. Neither superiority nor equivalence is established. Worst-six point rises0.996576835→0.999074032 but remains below1; no nonlinear-minimum CI is claimed.

## Adverse source/head points versus0.5

| Source/head | Dice delta | Paired95% interval |
| --- | ---: | --- |
| drhagis/vessel | -0.008836940 | [-0.016632497, -0.001836938] |
| chasedb1/vessel | -0.007434136 | [-0.008731735, -0.005706207] |
| g1020/disc | -0.005505517 | [-0.007813379, -0.003261189] |
| rimone/disc | -0.004950568 | [-0.006467094, -0.003467387] |
| g1020/cup | -0.003752530 | [-0.007474890, -0.000908233] |
| rimone/cup | -0.003640263 | [-0.008443932, +0.000996009] |
| iostar/disc | -0.002856031 | [-0.007815701, +0.002912196] |
| refuge/cup | -0.002470780 | [-0.003630782, -0.001230286] |
| drishti/disc | -0.001125485 | [-0.003680992, +0.001088182] |
| aria/disc | -0.000506074 | [-0.004317845, +0.003147236] |
| stare/vessel | -0.000340569 | [-0.003155424, +0.004248430] |

Independent saved-count QA passed: all8reports equal journal,770photo/record/source and valid/positive GT support match original andbaseline; integer pooled/source counts andDice/macro reaggregate, conditional correct/scored/eligible counts invariant perphoto, all six ratio points and selection recomputed. Native3180.5 counter equality was enforced before each journal entry using the same probability tensor, not a second NN.

Intervals use2000source-stratified paired draws/seed42/612knownsplitgroups. Grid/checkpoint selection and multiple comparisons remain adaptiveVAL; groups are not proven complete independent patients. Original historical exposure/preprocessing/backend limitations remain. No final quality promotion/default change follows.


## Reproduction and artifact scope

The unchanged science helper is on branch `experiments/native-v3-threshold-proposal` at `6de83d9` (`scripts/calibrate_native_v3_thresholds.py`). Runtime used the exact frozen318e342 package, with corrected operator bootstrap only. No new model/GT forward was performed for export. No model weights, private endpoints, operator profile bodies or source/run absolute paths are included.

One probability inference per photo means four neural views for rot4, reused across all eight thresholds; no eightfold threshold re-inference. There is no persistent probability cache. The native0.5 counter check uses injected SAME probabilities and does not establish global bitwise agreement with a separate earlier hardware/run. Conditional A/V invariance establishes that improved gated class scores here come from detection/gating, not improved class discrimination.

The largest macro on this grid is at0.30, not the declared worst-six winner0.40. The measured choice is exploratory; no application default, checkpoint alias or final selection/TEST protocol changes. A later final evaluation needs explicit reviewed protocol choice and fresh gate-eligible evidence rather than retagging these ablation reports.
