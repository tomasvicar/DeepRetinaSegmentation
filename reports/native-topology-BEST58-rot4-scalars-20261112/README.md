# Native BEST58 rot4 scalar calibration: completed negative result

The completed scan did not demonstrate the intended dense-vessel improvement and does not replace the selected raw512 model. All eight dense-vessel confidence intervals against the original networks are negative. The largest observed dense point estimate remains below selected raw512, while macro Dice, artery, vein, disc, cup and conditional AV are lower. Pooled vessel Dice is higher than selected raw512, but that pooled domain includes positive-only AVRDB and must not substitute for the dense endpoint. The user explicitly accepted the small vessel tradeoff and requested finalization.
The selected raw512 model and original models are the final handover. Research
is closed for this handover; no further experiment is planned. The known dense
TEST deficit of -0.004220235 Dice (about -0.422 percentage points) remains
documented, with no superiority across all heads or equivalence claim.

The original native-topology BEST58 checkpoint is SHA256 `7358d7c1246f99fb69f31af4f7f68e7dc8fe5b53e509ad211336310f5ff80c63`. It has independently checked own coverage of all 2,529 TRAIN photos / 5,700 annotation records, as does LAST60. The genuine standalone72 completion chain retains original training-source SHA256 `3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28`, original protocol/training terminal and the new 412 completion QA. Training BF16 metadata does not change the actual FP32 inference precision.

The one completed pass used common source `fe33e94a2370bd0776c8865d11324ae597ee5f0e`, evaluation fingerprint `895f48cda155d15e18aebcec97302800b4495194312c34d4e927343bb0356ea4`, rot4, tile512/overlap64 and the fixed eight global scalar thresholds. It covered all 770 VAL photos / 993 records. Every photograph was sampled once; all eight decoders shared the same probabilities. Native 0.5 counters matched those probabilities for every photo without a second forward. Conditional AV was invariant across thresholds. Canonical manifest 17a inputs were used directly with no evaluation cache. All complete counts were made durable before postchecks; the successful worker checked the 22,000 shared tile/view calls, model/RNG state and source/input identities.

## All-head point results

Macro is the source/head macro score; the five named heads below are pooled Dice, not components to average into that macro. Dense support contains 87 photos / 55 known split groups / 8 annotation sources and excludes only the positive-only AVRDB vessel domain.

| Scalar | Macro | Vessel | Artery | Vein | Disc | Cup | Dense vessel | Dense precision | Dense recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.805661 | 0.766043 | 0.757686 | 0.804488 | 0.915667 | 0.832906 | 0.745435 | 0.689040 | 0.811885 |
| 0.30 | 0.806913 | 0.768829 | 0.753296 | 0.801294 | 0.920243 | 0.840173 | 0.748264 | 0.704514 | 0.797808 |
| 0.35 | 0.807159 | 0.770385 | 0.748951 | 0.798139 | 0.923112 | 0.845149 | 0.749803 | 0.718095 | 0.784440 |
| 0.40 | 0.806872 | 0.771031 | 0.744527 | 0.794942 | 0.925212 | 0.849425 | 0.750406 | 0.730620 | 0.771294 |
| 0.45 | 0.806190 | 0.770892 | 0.740018 | 0.791579 | 0.926780 | 0.853294 | 0.750186 | 0.742515 | 0.758018 |
| 0.50 | 0.805186 | 0.770112 | 0.735138 | 0.788105 | 0.928026 | 0.856825 | 0.749320 | 0.754301 | 0.744405 |
| 0.55 | 0.803566 | 0.768430 | 0.729916 | 0.784201 | 0.928682 | 0.859579 | 0.747492 | 0.765499 | 0.730312 |
| 0.60 | 0.801511 | 0.765962 | 0.724326 | 0.780075 | 0.928928 | 0.861683 | 0.744847 | 0.776721 | 0.715487 |

Conditional AV accuracy is **0.817751241 at every scalar**, compared with selected raw512 **0.851194311**. End-to-end AV varies with the scalar; every absolute value and source result is preserved in [the full all-eight analysis](all-eight-analysis.json) and the original threshold reports below. Original integer counts retain full precision.

## Paired intervals and tradeoffs

These are existing source-stratified paired percentile 95% intervals, 2000 draws / seed 42. All-head analysis uses 612 known VAL groups; dense analysis uses its 55 supported groups. These groups are not proved independent patients. Intervals are exploratory and unadjusted across eight scalars, two references and multiple endpoints; an observed grid maximum is not an automatically selected protocol, and a crossing-zero interval does not establish equivalence.

| Scalar | Dense difference vs original6808 [95% interval] | Dense difference vs selected raw512 [95% interval] |
|---|---|---|
| 0.25 | -0.015891 [-0.020717, -0.011669] | -0.005880 [-0.009459, -0.002714] |
| 0.30 | -0.013063 [-0.017872, -0.008977] | -0.003052 [-0.006379, -0.000225] |
| 0.35 | -0.011524 [-0.016328, -0.007314] | -0.001513 [-0.004684, +0.001143] |
| 0.40 | -0.010921 [-0.015952, -0.006552] | -0.000910 [-0.004145, +0.001783] |
| 0.45 | -0.011141 [-0.016442, -0.006477] | -0.001130 [-0.004445, +0.001763] |
| 0.50 | -0.012007 [-0.017618, -0.007038] | -0.001996 [-0.005372, +0.001149] |
| 0.55 | -0.013835 [-0.019914, -0.008387] | -0.003824 [-0.007607, -0.000247] |
| 0.60 | -0.016479 [-0.023109, -0.010432] | -0.006468 [-0.010767, -0.002444] |

At scalar 0.40 (the largest observed dense point), dense Dice is 0.750406180 versus selected 0.751315957 and original 0.761326809. Macro at 0.40 is lower than selected by 0.015961880. At scalar 0.35 (the observed macro maximum), macro is 0.807159282 versus selected 0.822834097. Neither observation promotes a model or scalar.

The all-head comparison at 0.40 makes the pooled/dense distinction explicit:

| Measure | Original6808 | Selected raw512 | Native 0.40 | Native − selected [95% interval] |
|---|---:|---:|---:|---|
| Source/head macro | 0.667032 | 0.822834 | 0.806872 | -0.015962 [-0.027188, -0.010298] |
| vessel pooled Dice | 0.770124 | 0.762399 | 0.771031 | +0.008632 [+0.004396, +0.013500] |
| artery pooled Dice | 0.689530 | 0.762217 | 0.744527 | -0.017689 [-0.024024, -0.010484] |
| vein pooled Dice | 0.697298 | 0.811759 | 0.794942 | -0.016817 [-0.023469, -0.009840] |
| disc pooled Dice | 0.894808 | 0.942348 | 0.925212 | -0.017136 [-0.023802, -0.011598] |
| cup pooled Dice | 0.435399 | 0.884582 | 0.849425 | -0.035157 [-0.049355, -0.024090] |
| Dense vessel Dice | 0.761327 | 0.751316 | 0.750406 | -0.000910 [-0.004145, +0.001783] |
| Conditional AV | 0.748573 | 0.851194 | 0.817751 | -0.033443 [-0.039166, -0.027117] |
| End-to-end AV | 0.630317 | 0.720949 | 0.718666 | -0.002283 [-0.011354, +0.007483] |

Every supported source/head and conditional/end-to-end AV interval, including adverse results and AVRDB, is retained in the 16 paired artifacts. Positive-only AVRDB cannot establish vessel specificity or dense precision. Dense support excludes AVRDB only for that supplementary vessel endpoint; the original full benchmark and all original counts remain unchanged.

## Method, actual cost and limits

The actual service and external observer completed with numeric zero at 19:29:11 UTC. Whole-service elapsed time was **1335.170 seconds** (about 22.3 minutes), including this eight-scalar experiment and checks; this is not an isolated per-image latency benchmark. Systemd reported peak charged memory 2 GiB and swap 0 within the actual 3 GiB limit. Physical GPU selection, FP32/noautocast, intra 2 / interop 1, cores 2,3, Nice 15 and idle IO were recorded before model loading. The successful count-only analysis took 35.056 seconds, peak 161.4 MiB, with actual 1 GiB/noSwap/one-CPU/core6 limits; it imported no Torch and read no models, images or GT masks. Exporting this report reran neither inference nor statistics.

Reference reports are the already published [original6808](../original-f13-Old-VAL-20261112/validation.json) and [selected raw5120a5c](../shared3stream-tile-blend-VAL770-20261112/prior0a5c-validation.json). They are linked rather than duplicated. These comparisons change training, checkpoint role, architecture and source/inference context: native common fe33/895 is compared with frozen f13 references. The original networks also retain historical exposure/preprocessing qualifications. They are specified-pipeline contrasts, not an isolated architecture, capacity, threshold or training effect. A single training seed does not measure training-seed variance.

The scalar scan is **exploratory VAL calibration, not ordinary freezeable evidence**. Original report kinds, descriptors and bytes remain unchanged; no model or count-projection retag was made. The private analyzer used explicit NOT-gate count projections, whose paths survive as provenance strings in the unchanged analysis JSON; those temporary projections are not published or admitted as evidence. The existing [comparison math](../../src/deepretina/comparison.py) and [dense saved-count method](../dense-vessel-support-20261112/analyze.py) supply the recorded intervals. A new chosen scalar would need ordinary fresh standard VAL, but no further experiment or selection is planned for this handover.

Earlier launch attempts stopped before model loading because of a CLI spelling mismatch and a missing report locator. Two CPU-only analysis accessor failures were also preserved and corrected using original per-photo integer counts. The complete private failure and execution history remains in `metacentrum-workflows`, branch `prep/legacy-BEST58-scalar-secondary`; no inference was repeated after valid results. This report publishes only completed scientific artifacts and their original runtime/exit receipts. No new TEST, training, Optuna/DB action, product/default change or promotion occurred.

## Exact artifacts

- Original final threshold reports: [0.25](threshold-0.25.json), [0.30](threshold-0.30.json), [0.35](threshold-0.35.json), [0.40](threshold-0.40.json), [0.45](threshold-0.45.json), [0.50](threshold-0.50.json), [0.55](threshold-0.55.json), [0.60](threshold-0.60.json), and [original summary](summary-original.json).
- [All-eight analysis](all-eight-analysis.json), all 16 paired reports listed below, and [independent root integer recount](root-count-review.json).
- [Actual runtime](runtime-before-model.json), [service resource receipt](service-entry-resources.json), [pre-Torch worker resources](science-resources-before-Torch.json), [worker exit](worker-observed-exit.json), [external observer exit](outer-observed-exit.json), [service numeric exit](service-numeric-exit.json).
- [CPU analysis resources](count-analysis-resources.json), [observed CPU exit](count-analysis-observed-exit.json), [unchanged inputs/no-Torch proof](count-analysis-inputs-unchanged.json), [export provenance](export-provenance.json), and [complete artifact SHA256/size index](artifact-index.json).

- Scalar 0.25: [versus original6808](paired/0.25-minus-original6808.json), [versus selected raw512](paired/0.25-minus-selected_raw5120a5c.json).
- Scalar 0.30: [versus original6808](paired/0.30-minus-original6808.json), [versus selected raw512](paired/0.30-minus-selected_raw5120a5c.json).
- Scalar 0.35: [versus original6808](paired/0.35-minus-original6808.json), [versus selected raw512](paired/0.35-minus-selected_raw5120a5c.json).
- Scalar 0.40: [versus original6808](paired/0.40-minus-original6808.json), [versus selected raw512](paired/0.40-minus-selected_raw5120a5c.json).
- Scalar 0.45: [versus original6808](paired/0.45-minus-original6808.json), [versus selected raw512](paired/0.45-minus-selected_raw5120a5c.json).
- Scalar 0.50: [versus original6808](paired/0.50-minus-original6808.json), [versus selected raw512](paired/0.50-minus-selected_raw5120a5c.json).
- Scalar 0.55: [versus original6808](paired/0.55-minus-original6808.json), [versus selected raw512](paired/0.55-minus-selected_raw5120a5c.json).
- Scalar 0.60: [versus original6808](paired/0.60-minus-original6808.json), [versus selected raw512](paired/0.60-minus-selected_raw5120a5c.json).
