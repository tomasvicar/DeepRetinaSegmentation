# Completed TPE BEST58 inference TTA on VAL

The same immutable TPE72 BEST58 checkpoint completed four matched inference protocols on all770 VAL photos/612 known split groups. FP32, raw25ppd, tile512/overlap64, threshold.5, one GTX1060 6GB, CPU2/inter-op1. Every child and the separate supervisor observer returned actual0; total39m12s under a90min cap. Inputs/source retained their hashes and completed reports were read-only. No training, TEST, retries, cache build, default promotion or original-QA retagging occurred.

Checkpoint SHA`20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`; its own selected BEST58 covers2529 TRAIN photos/all5700 annotations from a genuinely completed60-epoch run. Original checkpoint training identity3725ad32 remains distinct from evaluator1f723ad/executionc4ebf061/broadtraining7bc7938e. The [completed standalone TPE proof and prior fresh VAL](../tpe72-completed-fresh-val) retain their original family/role. This BEST58 inference study is not the study terminal objective.

| Mode | Macro dataset Dice | Change vs fresh none [paired95% CI] | Conditional AV | Invocation seconds |
| --- | ---: | --- | ---: | ---: |
| none | 0.810897 | reference | 0.824920 | 220.43 |
| flips2 | 0.815087 | +0.004189 [+0.002374,+0.006076] | 0.833218 | 358.70 |
| rot4 | 0.818055 | +0.007158 [+0.005251,+0.008953] | 0.836113 | 625.06 |
| dihedral8 | 0.818854 | +0.007956 [+0.005841,+0.010113] | 0.837810 | 1147.47 |

| Mode | Vessel Dice | Artery Dice | Vein Dice | Disc Dice | Cup Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.762879 | 0.730324 | 0.792110 | 0.932671 | 0.880784 |
| flips2 | 0.764754 | 0.738924 | 0.799131 | 0.934783 | 0.883570 |
| rot4 | 0.766953 | 0.743989 | 0.803240 | 0.937129 | 0.885207 |
| dihedral8 | 0.767147 | 0.745566 | 0.804443 | 0.937411 | 0.885180 |

All five pooled and source-macro heads improve versus fresh none in each ensemble. Dihedral8 has the highest observed aggregate, but does not dominate rot4 on every head: pooled cup.885180 is slightly below rot4.885207. Vessel.767147 still trails frozen original.77012340; no all-original or held-out superiority is claimed. Its vessel gain vs none is+.00426812[.00367059,.00489744]; conditionalAV gain+.01289008[.01002521,.01603350].

## Adverse source results

| Mode | Source/head | Dice change [paired95% CI] | Supported photos/groups | Defined draws |
| --- | --- | --- | --- | ---: |
| flips2 | drhagis/artery | -0.002744 [-0.025460,+0.021155] | 3/3 | 1951 |
| flips2 | rimone/cup | -0.003840 [-0.011159,+0.002670] | 32/32 | 2000 |
| rot4 | rimone/cup | -0.004179 [-0.015938,+0.006254] | 32/32 | 2000 |
| dihedral8 | rimone/cup | -0.002523 [-0.013998,+0.007148] | 32/32 | 2000 |

Intervals crossing zero do not erase adverse point estimates. Complete aggregate source/head counts, macro and pooled intervals, conditional/end-to-end AV, defined draws and provenance digests are in [results.json](results.json). No photo identifiers, host paths or physical GPU UUIDs are included.

## Fresh same-device baseline, not prior Ada parity

The new GTX1060 none baseline is authoritative for these three comparisons. Against prior independently fresh Ada source72 BEST, all GT supports/attribution match, but global count parity is **false**:359 photographs/469 head-photo entries changed, macro+1.98535e−6 and conditionalAV+7.99609e−6. Source/backend/runtime/thread context differs; no counterfactual forward was used to assign a cause and no raw mask/probability bitwise claim is made. Exact pooled integer deltas are retained in the JSON.

## Operators, cost and scope

Full-frame preparation/photometry occurs once, then the entire prepared image is transformed before tiling. Flips2 uses identity/horizontal flip; rot4 uses rotations0/90/180/270; dihedral8 adds each rotation followed by horizontal flip. Inverse-aligned vessel, vessel×conditionalA, vessel×conditionalV, disc and cup are averaged in FP32, then conditional AV is recovered. Existing FOV restoration and decoder remain shared. Non-none enriched evidence is explicitly experimental; original standalone gates/defaultnone are unchanged.

These invocation times include a whole VAL process and integrity/preparation overhead. Fixed order and warm filesystem qualify them; they are **not resident latency**. The earlier [defaultnone CPU benchmark](../benchmark/strong-pipeline-cpu-drive21-tpe72-best58) measured only none; its8.416s must not be assigned to rot4/dihedral8 quality. A separately authorized direct CPU tradeoff run is distinct evidence, not assumed by this report.

The implementation remains on the separate [fd44-joint-tta source](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/1f723ad92898fad8b04a6ef8b39065f021f52a9c); this commit publishes reports only. The [earlier vascular4 TTA](../inference-tta-vascular4) used another checkpoint and retains its own matched baseline.

All intervals are exploratory source-stratified paired percentile intervals from2000 shared draws,seed42,of612 known groups. Multiple protocols/heads/sources and adaptive VAL tuning are unadjusted; known groups do not prove complete patient independence. These are conditional observations for one trained checkpoint, not training-seed robustness, clinical generalization, final selection, release promotion or TEST authorization.
