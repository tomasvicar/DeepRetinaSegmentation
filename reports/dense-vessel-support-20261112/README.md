# Dense vessel-support supplement (saved VAL counts only)

**The remaining dense-domain vessel gap is larger than the all-source Dice suggests.** AVRDB scores only marked vessel pixels. It contributes zero negative GT pixels, so FP0 / precision1 is structural, not evidence of dense precision or specificity. Only AVRDB is excluded; frozen reports, selection gates and pool objectives are unchanged.

All four pinned inputs contain the same770VAL photos /993record variants. Exact per-photo record IDs, channel-source attribution, valid-pixel count and GT-positive count agree. Vessel-supported photos alone number107 /75known split groups /9sources; after AVRDB exclusion: **87photos /55groups /8sources**. The excluded20photos /20groups contain596992positive and0negative scored pixels. The full VAL612groups must not be reused as the vessel bootstrap group count.

| Actual pipeline / role | All-source vessel Dice | Dense-only vessel Dice | Dense precision | Dense recall | Dense Dice delta vs original (95% paired CI) |
|---|---:|---:|---:|---:|---|
| original182 | 0.770123 | 0.761327 | 0.730948 | 0.794340 | reference |
| TPE5_LAST60_none_050 | 0.760803 | 0.738512 | 0.750510 | 0.726890 | -0.022815 [-0.032233, -0.014894] |
| TPE5_BEST58_rot4_035 | 0.769464 | 0.748236 | 0.723637 | 0.774567 | -0.013091 [-0.018558, -0.008791] |
| raw384_BEST59_none_050 | 0.759770 | 0.741098 | 0.754729 | 0.727951 | -0.020228 [-0.028356, -0.013598] |

Dense pooled support is60325745valid pixels,5893007GT-positive pixels and54432738GT-negative pixels for every input. Included-all support is60922737valid /6489999positive. `analysis.json` retains exact TP/FP/FN, support and per-source counts for both domains and AVRDB separately.

| Vessel source | Photos / known groups | GTpositive / valid pixels | Original Dice | TPE5 LAST60 NONE .5 | TPE5 BEST58 rot4 .35 | raw384 BEST59 NONE .5 |
|---|---|---|---:|---:|---:|---:|
| aria | 28 / 16 | 2693788 / 27618662 | 0.732888 | 0.697903 | 0.715299 | 0.700739 |
| avrdb (positive-only excluded) | 20 / 20 | 596992 / 596992 | 0.872294 | 0.981407 | 0.988809 | 0.949885 |
| chasedb1 | 6 / 3 | 213484 / 2149450 | 0.790873 | 0.769517 | 0.769653 | 0.771909 |
| drhagis | 8 / 8 | 261859 / 3829875 | 0.793068 | 0.760154 | 0.758210 | 0.763964 |
| erlangen | 22 / 11 | 447717 / 4235000 | 0.620504 | 0.630708 | 0.640456 | 0.633862 |
| hrf | 9 / 3 | 1020472 / 11404397 | 0.807503 | 0.789823 | 0.794477 | 0.791629 |
| iostar | 6 / 6 | 608249 / 5858697 | 0.816907 | 0.806031 | 0.813505 | 0.809218 |
| rite | 4 / 4 | 433387 / 3364003 | 0.813105 | 0.801628 | 0.804676 | 0.802572 |
| stare | 4 / 4 | 214051 / 1865661 | 0.832768 | 0.833908 | 0.836320 | 0.839133 |

## Interpretation and limits

The source adapter explicitly sets AVRDB vessel validity equal to its positive vessel mask, for both binary and AV records. VAL manifest metadata identifies only AVRDB as sparse/unmarked; actual saved counts confirm valid=GTpositive and FP0 for all four pipelines on every AVRDB photo. All remaining vessel photos have scored negatives. Sparse width/caliber auxiliary resources do not contribute dense vessel masks. This establishes benchmark support, not completeness of every manual vessel annotation.

TPE5 BEST58 rot4 .35 is an exploratory VAL-selected checkpoint/TTA/threshold pipeline; its near-original included-all Dice (.769464 vs .770123) conceals a negative dense-only difference with a below-zero interval. This is not proof of clinical/noninferiority/equivalence. More vessel-loss work should monitor dense precision AND recall, not improve only positive-only recall. All new pipelines still regress against original182 in dense Dice. Source-specific reversals remain visible above; no macro change is used to erase them.

The optional matched-GT paired contrasts against TPE5 LAST60 NONE .5 are descriptive **pipeline** contrasts: raw384 BEST59 and TPE5 BEST58/rot4/.35 do not share its selected role or complete inference protocol. Their intervals are not isolated loss, model or capacity effects. No historical report was retagged. Exact original role/configuration/source/protocol metadata are preserved in the JSON.

2000seed42 paired percentile draws resample actual retained reviewed split groups, stratified by the dataset signature among included vessel photos in each group. Dense domain has55groups, included-all75. These are recoverable split groups, not established independent patients; source imbalance, adaptive VAL selection and multiple comparisons qualify the intervals. Original historical exposure, preprocessing and CPU/CUDA backend differences remain. This is a supplementary vessel-support analysis, not a new frozen score or change to V3worst-six objective.

Control LAST60 input is actual fresh report7570255b..., not its slightly different saved-training counts; full actual FP32 flags/version/timing context is copied from its public completion summary. Saved and fresh global counts were not bit-exact. Context retains deterministic_algorithmsFalse, CUDA TF32False, cuDNN TF32True, cuDNN benchmark/deterministicFalse. CPU/GPU inference timings are not compared here.

## Validation and reproduction

`analyze.py --manifest PATH` reads only SHA-pinned manifest JSON, existing report JSON and NumPy counts. Runtime was bounded120s, CUDAhidden, CPU1/core4/nice17/idleIO. No Torch/model/image/GT-array load, new inference, PBS/DB operation, auth mutation or installs. Six metadata tests reject changed GTpositives, valid support, source attribution, photo membership, boolean and impossible integer counts. Both raw integer aggregation and all-photo eligibility checks passed. Input SHAs, immutable source-code SHAs and precise policy are in `analysis.json`; original files and current jobs are untouched.
