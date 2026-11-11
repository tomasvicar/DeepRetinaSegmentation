# Completed native V3 trial4 BEST54 inference TTA on VAL

Genuine `pool_72_v3` BEST54 completed none/flips2/rot4/dihedral8 on all 770 VAL photographs. A separate fixed-terminal LAST60 none case also completed all 770. All five observed child exits and the external supervisor exit were0; total2575.71s (42m55.7s), under the90min cap with60s cleanup reserve. Source/inputs were unchanged and original reports read-only. No training, TEST, retry, default promotion or final selection occurred.

BEST54 checkpoint SHA `21c1b3c6126494b8534a022b2256ada38687c9194120f7be63bf66879687bdfb`; LAST60 SHA `939c7aac465dd74127fffd2ac3d3da765849761681d4af1074a8e25a842b656e`. Both own selected roles cover all 5700 annotations/2529 TRAIN photos from a genuinely completed60-epoch native run. The [original typed eligibility](../v3-trial4-final60-eligibility) remains native pool-family evidence; no standalone receipt or role is substituted. BEST54 is not the study fixed-terminal objective.

Fresh evaluator [`318e342fe12d579a004ae7143fcf5e6155328a0b`](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/318e342fe12d579a004ae7143fcf5e6155328a0b), execution SHA `c3d311b3611abb561f54207366516ce0735fe9a3bb4cb2f5c32ea696aede9d2e`, evaluation identity `198fdd476505ab38437f8668945d24c39dfb344a5d8fd96a4bb2c9d3a7400d05`; original checkpoint training identity `3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28` is separate. Original none kind is `unified_validation_evidence`; non-none kind is `unified_tta_validation_evidence`. Sanitized exports are explicitly count views, not original gate inputs.

Every case uses raw25ppd, FP32/noautocast, tile512/overlap64/threshold0.5, GTX1060 6GB, CPU2/inter-op1. Preparation/photometry and FOV restoration occur once. Full prepared-image transforms precede tiling; inverse-aligned vessel, vessel×conditionalA, vessel×conditionalV, disc and cup are averaged inFP32 before recovering conditionalAV. Exact original descriptors and counts are retained.

|Role/mode|Macro dataset Dice|Change vs fresh BEST54 none [paired 95%CI]|Conditional AV|Invocation seconds|
|---|---:|---|---:|---:|
|best54-none|0.815058934|reference|0.823917368|220.334|
|best54-flips2|0.816065691|+0.001006758 [-0.000273205,+0.002376153]|0.826312387|359.000|
|best54-rot4|0.819411772|+0.004352838 [+0.001837136,+0.006218482]|0.832471660|624.695|
|best54-dihedral8|0.819308183|+0.004249250 [+0.001817569,+0.006302461]|0.832737434|1147.372|
|last60-none|0.804571577|separate terminal role|0.823323753|222.985|

|Role/mode|Vessel Dice|Artery Dice|Vein Dice|Disc Dice|Cup Dice|
|---|---:|---:|---:|---:|---:|
|best54-none|0.763144611|0.746224673|0.786902524|0.937357313|0.876624996|
|best54-flips2|0.765632932|0.748749174|0.789076005|0.939007889|0.877299223|
|best54-rot4|0.767487140|0.754223814|0.795406389|0.940039918|0.880924936|
|best54-dihedral8|0.767733750|0.754607033|0.795596593|0.940160255|0.881071136|
|last60-none|0.750502666|0.724275023|0.771122511|0.936111101|0.877352423|

## Direct dihedral8 minus rot4

Macro change **−0.000103588**, paired 95%CI **[−0.000497407,+0.000299564]**. The interval crosseszero; it does not establish equivalence. Vessel pooledDice has a small positive change, while all four other pooled head intervals cross zero. All source-macro head points except vessel are negative. ConditionalAV change+0.000265775 CI[−0.000307490,+0.000844988]. This differs from the earlier TPE checkpoint's positive direct 8 minus 4 macro result; those checkpoints are not interchangeable.

|Head|Pooled Dice change [paired 95%CI]|Source-macro Dice change [paired 95%CI]|
|---|---|---|
|artery|+0.000383219 [-0.000276011,+0.000979448]|-0.000175015 [-0.001443449,+0.001005762]|
|cup|+0.000146200 [-0.000270819,+0.000542508]|-0.000391343 [-0.001358264,+0.000586362]|
|disc|+0.000120336 [-0.000094478,+0.000359687]|-0.000116781 [-0.000784835,+0.000494642]|
|vein|+0.000190204 [-0.000350589,+0.000770339]|-0.000156269 [-0.000530368,+0.000373252]|
|vessel|+0.000246610 [+0.000116107,+0.000368509]|+0.000321467 [+0.000175155,+0.000481607]|

All adverse source/head8minus4 points are shown below; complete positive, null and adverse source/head/AV results remain in [the direct comparison](dihedral8-minus-rot4.json). IOSTAR artery and vein Dice intervals are wholly negative. IOSTAR conditional AV also falls by−0.001555724, CI[−0.002377373,−0.000765184], despite the small positive aggregate AV point.

|Source/head|Dice change [paired 95%CI]|
|---|---|
|aria/disc|-0.000737055 [-0.004313732,+0.001692285]|
|chasedb1/vessel|-0.000141091 [-0.000639922,+0.000314473]|
|drhagis/artery|-0.002358617 [-0.007555730,+0.002390950]|
|drhagis/vein|-0.001299931 [-0.001772373,+0.000549508]|
|drishti/cup|-0.000024404 [-0.001861445,+0.001476531]|
|erlangen/vessel|-0.000308694 [-0.000761944,+0.000087536]|
|iostar/artery|-0.000928398 [-0.001558109,-0.000236521]|
|iostar/vein|-0.001404228 [-0.002278435,-0.000645042]|
|refuge/cup|-0.000054689 [-0.000531090,+0.000410114]|
|rimone/cup|-0.002323214 [-0.005713211,+0.001134318]|
|rimone/disc|-0.000989992 [-0.002430902,+0.000347737]|
|rite/artery|-0.000241591 [-0.000893516,+0.000176394]|

## Same-mode comparison with first TPE BEST58

Each mode is paired directly with the [earlier first-TPE sweep](../tpe72-inference-tta-val) on exact770-photo/GT support and the same physical device class. The TPE checkpoint is BEST58 SHA20d4caab…, source1f723ad/executionc4ebf061; the V3 source is318e342/evaluation198fdd47. Different executed-source/model/runtime contexts remain explicit. Numeric microdrift cannot be isolated or attributed to model choice; these are not causal single-factor comparisons.

|Mode, V3 minus prior TPE|Macro change [paired 95%CI]|ConditionalAV change [paired 95%CI]|
|---|---|---|
|none|+0.004161514 [+0.000149384,+0.012844182]|-0.001002176 [-0.006135654,+0.003736548]|
|flips2|+0.000978787 [-0.002792547,+0.009450001]|-0.006905955 [-0.012797972,-0.001463170]|
|rot4|+0.001356551 [-0.001441812,+0.007475376]|-0.003641647 [-0.009182074,+0.002037588]|
|dihedral8|+0.000454402 [-0.002320830,+0.006314358]|-0.005072186 [-0.010664442,+0.000592488]|

V3rot4 has higher pooled artery/disc/vessel pointDice than TPErot4, but lower vein/cup Dice and conditionalAV. No blanket model dominance follows from a larger observed macro. Every all head/source interval and adverse outcome remains in the four `*-V3-minus-TPE.json` files.

## Count export and limits

Five flat `*-counts.json` files retain every one of770 per-photo count rows, all five channel TP/FP/FN/valid counts and conditional/end-to-endAV counters. Dense ordinal rows and known-group aliases are shared across the five exports; original photo/annotation/group identifiers and operational details are omitted. The612 group aliases preserve original sorted-group order and dataset-signature strata, allowing the same saved-count bootstrap grouping; see [grouping-semantics.json](grouping-semantics.json). Pooled integer counts were reconstructed exactly. Whole original report and comparison digests are in [results.json](results.json) and [provenance-digests.json](provenance-digests.json). These sanitized views cannot replace original gated validation evidence.

Fresh BEST54 none GT support matches the saved native sidecar exactly, but prediction-count parity is false (macro+0.0000016722). The saved sidecar is not an independently fresh baseline. New same-run none controls these TTA comparisons; no cause or raw-probability bitwise parity is asserted.

Intervals are exploratory source-stratified paired percentile intervals:2000 shared draws,seed42,612 known groups. Adaptive VAL tuning and many source/head comparisons are uncorrected; alias grouping does not prove complete patient independence. No training-seed uncertainty, clinical generalization, equivalence, default selection, release promotion or TEST permission is established. Invocation times include full VAL/preparation/integrity overhead with fixed order and warm filesystem; no resident latency or latencyCI claim. Threshold followup is a separate experiment. Its first attempt failed before model loading with actual outer exit1 and no quality result. A corrected v2 observer launched at18:29:57UTC; no threshold quality or completion is claimed by this snapshot.
