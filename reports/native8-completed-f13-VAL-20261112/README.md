# Native auxiliary8: completed BEST51 and LAST60 fresh VAL counts

These are **VAL-only observational results, not a model selection or a release**. Both original model roles own all5700 selected annotations/2529TRAIN photos after an actual full60 run. No new NN, model/tensor/image/GT-array or TEST access was performed for this analysis.

BEST51's original inference completed all770 and wrote its atomic report, then an added operator postcheck failed: original worker1/outer1 are preserved. The separate corrected CPU source/config/data/count adoption exited0; it is **not a replacement inference0**. LAST60 subsequently had a separate actual fresh worker0/outer0 (222.91s), with unchanged inputs and all770 support matching BEST.

## Five pooled segmentation Dice and conditional A/V accuracy

| Pipeline | Macro | Vessel Dice (incl. AVRDB) | Artery Dice | Vein Dice | Disc Dice | Cup Dice | Conditional A/V accuracy | Worst-six ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|native8_BEST51_NONE050|0.808526687|0.759924586|0.727786689|0.784407694|0.938985360|0.880359364|0.821227636|0.986756556|
|native8_LAST60_NONE050|0.802400558|0.754968321|0.724153680|0.773731964|0.932618644|0.868017465|0.821356335|0.980320882|
|original6808|0.667031974|0.770123676|0.689530320|0.697297832|0.894807839|0.435398951|0.748572508|1.000000000|
|raw512_BEST50_NONE050|0.811330533|0.756125755|0.734413532|0.788574982|0.938677651|0.876027589|0.843312833|0.981823801|
|raw512_BEST50_rot4_035|0.822834097|0.762398925|0.762216696|0.811759302|0.942348197|0.884582493|0.851194311|0.989969467|
|TPE5_LAST60_NONE050_crosscontext|0.808832542|0.760802772|0.732653668|0.790739644|0.935337602|0.874872420|0.821754997|0.987896874|
|TPE5_BEST58_rot4_035_crosscontext|0.822099157|0.769463926|0.759658718|0.814793346|0.936902210|0.884338615|0.836113307|0.999143320|

Worst-six is exactly the minimum of six **pooled** candidate/original6808 ratios (five head Dice plus conditional A/V accuracy), not a source ranking or a macro score. Native8 ratios remain below1 because vessel Dice is lower. Its other four pooled segmentation heads and conditional A/V accuracy exceed the original point values, but that does not waive the vessel goal.

## PRIMARY dense-vessel domain: 87 photos /55 known split groups /8 sources

PRIMARY here means this scientific supplement, not a changed frozen study objective. Exclude only known positive-only AVRDB. Every dense pipeline has60,325,745valid pixels and5,893,007GT-positive pixels. Preserve the full includeAll107vessel photos/75groups/9sources separately; AVRDB accounts for20photos/20groups/596,992positive valid pixels and **zero valid negative pixels**.

| Pipeline | TP | FP | FN | Precision | Recall | Dense Dice |
|---|---:|---:|---:|---:|---:|---:|
|native8_BEST51_NONE050|4,303,990|1,396,991|1,589,017|0.754956033|0.730355487|0.742452036|
|native8_LAST60_NONE050|4,360,302|1,450,574|1,532,705|0.750369135|0.739911220|0.745103484|
|original6808|4,681,051|1,723,029|1,211,956|0.730948239|0.794339969|0.761326809|
|raw512_BEST50_NONE050|4,366,739|1,428,163|1,526,268|0.753548378|0.741003532|0.747223306|
|raw512_BEST50_rot4_035|4,613,808|1,775,123|1,279,199|0.722156492|0.782929326|0.751315957|
|TPE5_LAST60_NONE050_crosscontext|4,283,570|1,423,973|1,609,437|0.750510333|0.726890363|0.738511536|
|TPE5_BEST58_rot4_035_crosscontext|4,564,527|1,743,233|1,328,480|0.723636759|0.774566703|0.748236074|

Native8 has higher dense precision but substantially lower recall than original6808. Compared with matched NONE/.5 raw512, BEST has slightly higher precision and lower recall; LAST is lower on both. No unmeasured threshold can be inferred from these counts. This is not the same two-sided deficit as raw512rot4/.35 versus original, and none of these contrasts establishes an optimal threshold.

## Paired uncertainty, with regressions retained

Existing source-stratified known-group method:2000draws, seed42, allheads shared draws. FullVAL770 uses612groups; dense87 recomputes its own55groups. They are recoverable split/photo/source groups, **not proven independent patients**. No multiplicity correction, noninferiority, equivalence or clinical generalization claim.

| Contrast (candidate minus reference) | Macro delta [95% interval] | Dense vessel Dice delta [95% interval] |
|---|---|---|
|native8_BEST51_NONE050__minus_original6808|+0.141494713 [+0.132486702, +0.151980277]|-0.018874772 [-0.025108597, -0.013322116]|
|native8_BEST51_NONE050__minus_raw512_BEST50_NONE050|-0.002803846 [-0.006023282, +0.000255902]|-0.004771269 [-0.007162970, -0.002276677]|
|native8_LAST60_NONE050__minus_original6808|+0.135368584 [+0.124658741, +0.146116604]|-0.016223325 [-0.021923952, -0.011263837]|
|native8_LAST60_NONE050__minus_raw512_BEST50_NONE050|-0.008929975 [-0.015892510, -0.004483526]|-0.002119822 [-0.003938985, -0.000278879]|
|native8_LAST60_NONE050__minus_native8_BEST51_NONE050|-0.006126129 [-0.012469147, -0.002372449]|+0.002651448 [+0.001440430, +0.003867241]|

The includeAll vessel ranking reverses after removing sparse AVRDB: BEST has0.759924586 versus LAST0.754968321, but dense BEST0.742452036 versus LAST0.745103484. LAST-minus-BEST dense gain is+0.002651448 [0.001440430,0.003867241]. AVRDB itself has BEST Dice0.939802342 versus LAST0.864746091, bothFP0/precision1 by annotation support. That precision cannot establish dense performance. Likewise BEST's includeAll vessel gain versus raw512NONE (+0.003798831 [0.001080003,0.006720232]) coexists with a dense deficit (−0.004771269 [−0.007162970,−0.002276677]).

All source/head point regressions and interval signs are retained in [all-source-point-regressions.json](all-source-point-regressions.json), not hidden by macro gains. [All source/head TP/FP/FN/support and P/R](all-head-source-counts.csv) and [dense source counts](dense-vessel-source-counts.csv) are exact. Complete [paired reports](paired/) include every head, source and conditional A/V accuracy/counts/interval.

## Protocol and provenance qualifications

- Original6808 is the actual final Oldf13 reproduction, kept distinct from historical182 (the vessel difference is+2TP/−2FN). Historical original exposure, preprocessing/backend and nominal geometry limitations remain.
- Native8 and raw512NONE share Oldf13/Python3.12.14/FP32/NONE/512/64/.5 inference context. Checkpoint, training geometry/hyperparameters/sourcefamily differ: this is a pipeline comparison, not a causal feature or architecture ablation.
- Raw512rot4/.35 changes both TTA and threshold relative to NONE/.5. Do not attribute that joint contrast to either factor alone.
- TPE5LAST60 uses an earlier Ada/b269/Python3.12.13 producer; TPE5BEST58rot4/.35 is earlier318 exploratory threshold calibration. Their same-GT count contrasts are explicitly cross-source/runtime/protocol observations, not a controlled training effect or newly freezeable evidence.
- Inference is FP32/noautocast; BF16 is original TRAIN configuration only. Native8 BEST freshmacro0.808526687 differs slightly from saved training-selection0.808531598; LAST freshmacro0.802400558 differs from saved0.802392671. No bitexact parity claim.

`analysis.json` preserves model/report/config/protocol/source hashes, distinct completion roles and all count points. `dense-vessel.json` preserves both support domains, AVRDB sensitivity-only evidence and all dense paired intervals. Scientific export omits operator paths, PIDs, GPUUUIDs and photo/group identifiers; original byte-complete reports, CPUadoption and actual exit receipts remain in the separately pinned private evidence. No frozen source, gate, benchmark objective, default, threshold or model registry changed. TEST remains untouched.
