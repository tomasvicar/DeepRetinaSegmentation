# Native auxiliary TPE6: completed60 and matched fresh VAL

Both original training and full60 QA completed with externally observed exit0. BEST52 and LAST60 each have actual coverage of all2529 TRAIN photos and all5700 annotation IDs. This report evaluates stopped immutable checkpoint bytes; no model, source, data or study changes.

The subsequent fresh operation completed with observed outer0 and both validation children0: LAST60 93.5423s; BEST52 95.8078s; total189.7979s. Both reports contain all770 VAL photos. FP32/NONE, threshold0.5, tile512/overlap64,25ppd raw prepared geometry, source72, physical Ada and intra8/inter1 are matched to first-TPE validation. Source/runtime, data identity and complete inference-protocol dictionaries were checked equal. The first TPE BEST epoch is58; next TPE6 BEST epoch is52. The terminal60 comparison uses60 for both.

| Role | Metric | First TPE | TPE6 | Delta | Paired95%CI |
|---|---|---|---|---|---|
| best | Macro | 0.810895434 | 0.808852833 | -0.002042601 | [-0.00622466151997259, 0.005136774342152181] |
| best | vessel | 0.762882197 | 0.761709005 | -0.001173192 | [-0.002830774541214903, 0.0004879149138673429] |
| best | artery | 0.730308385 | 0.722658604 | -0.007649781 | [-0.01425614033694215, -0.0014508776767572485] |
| best | vein | 0.792116542 | 0.779846004 | -0.012270539 | [-0.018585123883387428, -0.005957816268568386] |
| best | disc | 0.932670931 | 0.937703417 | +0.005032486 | [0.0003644197362920182, 0.01081255939925328] |
| best | cup | 0.880783279 | 0.878005084 | -0.002778196 | [-0.009593986380878203, 0.002533840709977073] |
| best | Conditional AV | 0.824911548 | 0.806978606 | -0.017932943 | [-0.024334005004868278, -0.011291297203012059] |
| fixed_terminal60 | Macro | 0.808832542 | 0.801973971 | -0.006858571 | [-0.012083146317975971, 8.631972220329375e-05] |
| fixed_terminal60 | vessel | 0.760802772 | 0.756283114 | -0.004519658 | [-0.008323130999988552, -0.0003110696854045379] |
| fixed_terminal60 | artery | 0.732653668 | 0.701746604 | -0.030907064 | [-0.03977517324571229, -0.021820689386231693] |
| fixed_terminal60 | vein | 0.790739644 | 0.768438394 | -0.022301250 | [-0.028969885698123282, -0.015882699820054032] |
| fixed_terminal60 | disc | 0.935337602 | 0.938609946 | +0.003272345 | [-0.0012525010054713986, 0.007685130040581368] |
| fixed_terminal60 | cup | 0.874872420 | 0.876284250 | +0.001411830 | [-0.007766716381445274, 0.008741248627569398] |
| fixed_terminal60 | Conditional AV | 0.821754997 | 0.805443737 | -0.016311260 | [-0.02299326663890082, -0.009478194013934809] |

The next proposal does not improve all tasks. BEST artery, vein and conditionalAV intervals lie below0; disc is positive while vessel, macro and cup intervals cross0. LAST vessel, artery, vein and conditionalAV intervals lie below0. Neither cup nor disc offsets these vascular regressions.

All source/head/AV results are retained. BEST has20 adverse source/head point deltas,7 with interval upperbound below0; LAST has17 and7. Paired bootstrap2000/seed42 uses612 known split groups and source strata, not proven independent patient groups. Repeated adaptive VAL use and prior model selection prevent an independent quality-improvement claim. No TEST or default/model promotion.

The two configurations differ in learning rate, weight decay and vascular sampling weight; exact full configurations are retained. These observations do not isolate a causal effect of any single parameter.

Saved BEST versus fresh BEST has tiny actual numerical drift: macro−5.3161e−8, vessel+5.9490e−8, cup−1.4522e−7, conditionalAV unchanged. Actual per-photo confusion counts differ on3 vessel,2 disc and1 cup photos; artery/vein/AV counts match. All photo/record/source/GT supports match. The saved sidecar is not relabeled as fresh.

Original wrapped reports and their byte SHAs are preserved. Separate derived metric views are explicitly labeled and hashed for the existing comparison API; they are not replacements for original producer evidence.
