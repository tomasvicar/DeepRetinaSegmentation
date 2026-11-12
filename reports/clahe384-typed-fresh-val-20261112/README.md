# CLAHE384 actual typed fresh VAL: both roles completed, adverse result

The original training60, both selected-role CPU audits and both freshVAL770 completed successfully. **CLAHE is worse than matched raw384 for every pooled head and conditional A/V accuracy**, with below-zero paired intervals for BEST and LAST. This is evidence against promoting this CLAHE recipe, not a reason to hide the successful execution or alter the benchmark.

Original e3b outer observed0 at00:41:13.859644UTC; actual stage cases BEST58=0/LAST60=0, no timeout/retry/interruption, source and original inputs unchanged. Stage elapsed925.503s. All original report/launch/runtime/case/outer bytes were copied and SHA-inventoried before count analysis. FIRST profile PermissionError and its later root-approved path-only continuation remain preserved separately. No new inference was launched by this analysis.

| Actual selected role | Macro | Vessel | Artery | Vein | Disc | Cup | Conditional A/V accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLAHE_BEST58 | 0.801035 | 0.751130 | 0.721123 | 0.770226 | 0.924293 | 0.855065 | 0.810557 |
| CLAHE_LAST60 | 0.798671 | 0.750185 | 0.717596 | 0.770123 | 0.924778 | 0.847977 | 0.810917 |

Matched training_best vs TPE5BEST58 (Old NONE/.5) macro difference−.0098623 CI[−.0157513,−.0017355]; matched terminal60 vs TPE5LAST60 (actual fresh Ada NONE/.5)−.0101611[−.0168537,−.0011092]. Original runtime/backend contexts differ and are preserved in the reports; neither comparison pretends to isolate photometry causally across backends. The direct raw384 contrast uses the same Old6d/f13/FP32/NONE/.5 full geometry, training recipe and selected-role semantics, with the declared raw-versus-CLAHE profile difference. BESTepochs differ58 vs59.

| CLAHE minus matched raw384 | Macro delta [95%CI] | Vessel | Artery | Vein | Disc | Cup | Conditional A/V accuracy |
|---|---|---|---|---|---|---|---|
| CLAHE_BEST58 | -0.013336 [-0.018061,-0.009167] | -0.008640 [-0.012261,-0.005878] | -0.014262 [-0.021601,-0.006257] | -0.026373 [-0.033583,-0.019483] | -0.016238 [-0.022479,-0.010684] | -0.025404 [-0.040509,-0.012916] | -0.020348 [-0.027370,-0.013062] |
| CLAHE_LAST60 | -0.014824 [-0.019714,-0.010405] | -0.009604 [-0.012714,-0.006950] | -0.026381 [-0.030979,-0.021551] | -0.025314 [-0.031980,-0.018762] | -0.012804 [-0.019316,-0.007409] | -0.024994 [-0.044226,-0.010338] | -0.021053 [-0.026365,-0.014868] |

## Sources and dense vessel support

All770photos /993record variants and all head-specific valid/positive GT counts match; full paired source-stratified bootstrap uses612known split groups,2000draws/seed42. They are not proven complete independent patients. Exact per-source TP/FP/FN, source support, all intervals and adverse-source rows are retained in the paired JSON files and `adverse-source-results.json`; no favorable-only filtering. Source intervals are unadjusted for multiplicity.

| Worst BEST58 source/head regressions vs raw384 BEST59 | Dice delta | 95%CI |
|---|---:|---|
| iostar / vein | -0.048218 | [-0.082306,-0.019747] |
| hrf / artery | -0.031893 | [-0.036061,-0.026560] |
| drhagis / artery | -0.031855 | [-0.040391,-0.023573] |
| drhagis / vein | -0.029144 | [-0.060172,-0.011145] |
| avrdb / vein | -0.027039 | [-0.041415,-0.016279] |
| refuge / cup | -0.026308 | [-0.042132,-0.013015] |
| g1020 / cup | -0.025931 | [-0.064557,+0.000365] |
| avrdb / vessel | -0.023781 | [-0.034962,-0.016965] |

AVRDB vessel validity is positive-only, so dense supplementary vessel support excludes only AVRDB:87photos /55groups /8sources. Full objective and all-source reports remain unchanged. Dense vessel Dice: original182 .76132661; CLAHEBEST .73412919 /LAST .73167579, matched raw384BEST .74109829 /LAST .73840472. Dense CLAHE-minus-raw differences: BEST−.0069691[−.0108195,−.0039166], LAST−.0067289[−.0097828,−.0039662]. Dense vs original differences−.0271974[−.0385548,−.0180851] and−.0296508[−.0417139,−.0195516]. These use55actual retained vessel groups, not612; no clinical/noninferiority/equivalence claim.

Relative to original182, CLAHE still improves A/V/disc/cup points while vessel regresses. Historical original exposure/preprocessing and backend limits remain. All head results and source negatives are shown, so overall macro gain versus older references is not used to hide the vessel deficit.

## Integrity, numeric limits and scope

Evaluation source6d3593f/f13 actual typed curated family, original fd44/fcb/2b training/config/index/cache/descriptor bindings and each selected role own2529photos/5700annotations are preserved. No V3/pool/standalone retagging. Both evals FP32, no autocast, NONE/512/64/.5, same GTX1060 device model, full-frame CLAHE profile/backend then normal canonical geometry; source/runtime metadata remain in original JSON.

Fresh and saved-training counts are not bitwise identical: BEST macro delta 1.4156675632870375e-06; LAST -2.1359921476005184e-07. `saved-fresh-drift.json` records actual per-head differing-photo counts and exact unchanged support. No causal explanation for these tiny runtime differences is claimed; no extra NN was run to chase them.

Paired analysis uses only existing saved JSON counts, standard `deepretina.comparison.compare_reports` math and the reviewed dense-counter helper. CPU1/core4/nice17/idleIO, CUDAhidden, bounded180s; the metadata-only analysis process had externally observed exit0. No Tensor/model/GT-array load, GPU query, new launch, TEST, DB/PBS operation or canonical-ledger edit occurred. Original files remain unchanged; derived views explicitly pin original report, source and checkpoint identities. No model/default/candidate promotion or final freeze.


## Public evidence scope

The two full count reports and derived count views preserve exact per-photo
confusion counts, source attribution, selected-role coverage, full training
configuration, photometric descriptor/backend/cache and checkpoint identities.
Only operational absolute paths and the physical GPU UUID are removed. The
exported wrappers are explicitly scientific copies, **not original gate inputs**.
`export-provenance.json` binds every export to its original byte SHA; original
receipts, operator commands and unsanitized proof bytes remain private.
No raw retinal pixels, tensor/model binaries, source changes, TEST or new model
inference are included. Segmentation values are Dice; conditional A/V is the
actual saved classification accuracy metric.
