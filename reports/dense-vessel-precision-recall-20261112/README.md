# Dense vessel gap: saved VAL precision/recall triage

Metadata-only analysis at source 7d9d37c. No NN, weights/image/GT-array access, training, host/Meta/DB query, source/gate/default change or TEST.

The domain is the already audited **87 photos / 55 known split groups / 8 vessel sources**, excluding only positive-only AVRDB. Every model has 5,893,007 GT-positive and 60,325,745 valid pixels. Known groups are not proven independent patients. This does not redefine the full objective.

## Pooled counts

| Pipeline | TP | FP | FN | Precision | Recall | Dice |
|---|---:|---:|---:|---:|---:|---:|
| original182 |4,681,049|1,723,029|1,211,958|0.730948155|0.794339630|0.761326607|
| original_final6808 |4,681,051|1,723,029|1,211,956|0.730948239|0.794339969|0.761326809|
| raw512_BEST50_rot4_035 |4,613,808|1,775,123|1,279,199|0.722156492|0.782929326|0.751315957|
| TPE5_BEST58_rot4_035 |4,564,527|1,743,233|1,328,480|0.723636759|0.774566703|0.748236074|
| TPE5_BEST58_rot4_050 |4,310,595|1,361,628|1,582,412|0.759948084|0.731476307|0.745440428|
| raw512_BEST50_none_050 |4,366,739|1,428,163|1,526,268|0.753548378|0.741003532|0.747223306|

Raw512 and TPE5 atrot4/.35 both have lower dense precision **and** recall than the original. Relative to182, raw512 adds67,241FN and52,094FP (56.35% of the119,335 extraFP+FN are misses); TPE5 adds116,522FN and20,204FP (85.22% of136,726 extraerrors are misses). These are descriptive error-count decompositions, not causal attribution or a proof that thresholding can recover those vessels. TPE5 has a predominantly recall deficit; raw512 has a material two-sided deficit.

The final-source original6808 is kept separate from historical182: vesselTP+2/FN−2 (ARIA+3TP, Erlangen−1TP), FPunchanged. It strengthens the reference by about0.00000020Dice; no substantive interpretation changes and no historical report is retagged. Exact raw support remains equal.

## Every dense vessel source

O=original182, R=raw512BEST50rot4/.35, T=firstTPE5BEST58rot4/.35. Exact counts, precision, recall and reference deltas are in by-source.csv and analysis.json.

| Source | Pipeline | TP | FP | FN | Precision | Recall | Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| aria |O|2,052,605|855,025|641,183|0.705937|0.761977|0.732888|
| aria |R|2,013,910|896,812|679,878|0.691894|0.747613|0.718675|
| aria |T|1,949,210|807,058|744,578|0.707192|0.723594|0.715299|
| chasedb1 |O|191,076|78,643|22,408|0.708426|0.895037|0.790873|
| chasedb1 |R|188,000|84,470|25,484|0.689984|0.880628|0.773736|
| chasedb1 |T|188,590|87,991|24,894|0.681862|0.883392|0.769653|
| drhagis |O|222,870|77,316|38,989|0.742440|0.851107|0.793068|
| drhagis |R|220,446|86,783|41,413|0.717530|0.841850|0.774734|
| drhagis |T|222,400|102,386|39,459|0.684759|0.849312|0.758210|
| erlangen |O|269,730|151,943|177,987|0.639666|0.602456|0.620504|
| erlangen |R|305,941|197,657|141,776|0.607510|0.683336|0.643196|
| erlangen |T|314,079|219,002|133,638|0.589177|0.701512|0.640456|
| hrf |O|866,049|258,483|154,423|0.770142|0.848675|0.807503|
| hrf |R|827,942|231,981|192,530|0.781134|0.811332|0.795947|
| hrf |T|832,062|242,081|188,410|0.774629|0.815370|0.794477|
| iostar |O|512,345|133,760|95,904|0.792975|0.842328|0.816907|
| iostar |R|502,880|121,138|105,369|0.805874|0.826767|0.816187|
| iostar |T|502,470|124,602|105,779|0.801296|0.826093|0.813505|
| rite |O|369,351|105,757|64,036|0.777404|0.852243|0.813105|
| rite |R|364,099|106,121|69,288|0.774316|0.840124|0.805879|
| rite |T|364,849|108,586|68,538|0.770642|0.841855|0.804676|
| stare |O|197,023|62,102|17,028|0.760340|0.920449|0.832768|
| stare |R|190,590|50,161|23,461|0.791648|0.890395|0.838123|
| stare |T|190,867|51,527|23,184|0.787425|0.891689|0.836320|

- ARIA is the largest count domain. Raw512 loses38,695TP and adds41,787FP; TPE5 loses103,395TP while reducingFP47,967. Its precision is slightly above original but recall drops0.03838. A uniform claim of precision-only failure would be wrong.
- HRF/IOSTAR show higher precision but lower recall for both new pipelines: missed-vessel recovery matters. DRHAGIS, especially TPE5, is mainly a precision problem (TPE5FP+25,070 with only470additionalFN). CHASE and RITE have deficits on both axes.
- Erlangen gains recall at a substantial precision cost and still improves Dice; STARE gains precision, loses recall and improves Dice. Global threshold reduction can worsen already poor precision, so source tradeoffs cannot be hidden by pooled/macro gains.

## What threshold evidence actually isolates

The firstTPE5 rot4 threshold0.35 and0.50 reports come from the **same BEST58 and same per-photo probabilities**. Their actual protocol descriptors differ only in scalar threshold; the completed sweep reports one inference/four neural views per photo, with exact same-forward native0.5counter parity onall770. This is a genuine decoder-threshold comparison, albeit exploratory VAL calibration.

Lowering0.50→0.35 adds253,932TP and381,605FP on dense support: precision0.759948→0.723637, recall0.731476→0.774567, Dice0.745440→0.748236. The threshold tradeoff is real but does not close the original gap. ConditionalAV classification remains threshold-invariant; vessel gating/head Dice may change.

For raw512 the existing contrast is **NONE/.50→rot4/.35**: both TTA and threshold change. Its denseTP+247,069/FP+346,960/Dice+0.004093 cannot be assigned to either change separately. There is no saved same-forward raw512rot4/.50 curve or reusable validation probability cache in these artifacts. Counts cannot reconstruct unseen threshold decisions or identify an unmeasured optimum.

## Already measured firstTPE5 dense curve (not raw512 calibration)

| Threshold | Precision | Recall | Dense Dice |
|---|---:|---:|---:|
|0.25|0.693500496|0.804647441|0.744950997|
|0.30|0.709547841|0.789228657|0.747270177|
|0.35|0.723636759|0.774566703|0.748236074|
|0.40|0.736460100|0.760220376|0.748151637|
|0.45|0.748394637|0.745889492|0.747139965|
|0.50|0.759948084|0.731476307|0.745440428|
|0.55|0.771228326|0.716597995|0.742910196|
|0.60|0.782545975|0.700791464|0.739415759|

Among these **already measured eight grid points**,0.35 has the highest TPE5 dense Dice0.748236074;0.40is0.748151637. This is not a new selection, a globally optimal threshold estimate, a confidence claim or a transferable raw512/native8 calibration. OriginaldenseDice0.761326607 remains higher. The earlier full-objective/worst-six choice also selected0.35 and did not meet all-six target. Macro improvement does not waive the dense-vessel goal.

## Existing machinery and smallest useful next validation

The existing HPO scripts/calibrate_vascular2_thresholds.py supplies photo_counts/aggregate/native_parity and streams one probability image into the eight predeclared thresholds0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60. It is pinned to vascular2BEST59 and cannot be pointed at raw512 by merely changing a filename. The private previously executed calibrate_standalone_tpe_thresholds.py supplies TTA-aware same-forward counterparity/all770 annotation tracing; it is likewise pinned to standalone72BEST58/evaluation318/original182 and must not be silently retargeted. Script artifact identifiers and hashes are recorded in machinery.json; operational paths remain private.

After the current arms finish and root reviews resources, the minimal missing part is a **new small external validation-only adapter**, using those counter functions under unchanged frozen6d/f13 source. It must bind actual source/role/checkpoint plus genuine curated_fd44 original index/QA for raw512; future native8/width24 uses genuine standalone72 protocol/terminal/QA, never a fake curated or pool envelope. Keep complete raw config/photometry and all770 source/GT support, fixedTTA/FP32/512/64, and inject the SAME probabilities into native0.5counter parity without a second NNcall. Stream eight count sets and discard each image probability after counting; keep allheads/source/adverse/dense metrics and threshold-invariant conditionalAV. No new training arm, sourceML code or heavy probability-cache system is needed.

Existing derived/photometry caches store prepared inputs, **not model probabilities**. Per-image prediction NPZ output is restored to original geometry and is not a verified canonical VAL probability cache; it cannot be casually reused for this comparison. No reusable all770 probability cache was found in the inspected completed calibration path.

The previously completed TPE5rot4 eight-point sweep took685.42s on Old. That is a measured precedent, not a promised raw512/native8 runtime. A future bounded sweep is still a separately reviewed actualVAL inference job; none is launched/prepared for execution here. Its reports must remain explicitly calibration/ablation evidence, not retagged standard freezeableVAL. After a protocol decision, genuine fresh typedVAL/freeze requirements still apply. No benchmark/objective/default or TEST route is changed.

## Validation limits

analyze.py is stdlib saved-JSON-only. It verifies all770 photo/record/source and vesselGT support between available complete original/TPE reports, nonnegative integer confusion consistency, identical per-source GTsupport and exact pooled87domain totals; it reproduces prior TPE5dense counts. Raw512 public projection omits photo IDs: this analysis checks its exact source-level support, while same87photo provenance comes from the existing completed full-report/dense audit. It does not pretend to independently reread missing per-photo records. Existing55group count is inherited from that pinned audit, not newly inferred from images or patients. Original input SHAs and all source counts remain in analysis.json. No new interval, clinical equivalence/superiority, threshold optimum, model promotion or source/exposure independence claim is made.
