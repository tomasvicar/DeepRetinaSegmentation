# Genuine V3 final60: saved VAL counts

These are metadata-only exploratory comparisons of the two genuinely completed source72 runs. All 12 pairs passed exact 770-photo membership,612 known split-group identity, annotation record/source attribution, per-head valid/positive GT support, pooled integer-count reconstruction, and conditional AV eligible/scored-domain checks. No new model prediction, GT mask decoding, TEST, selection freeze, or release change occurred.

Both BEST/LAST roles have independently checked `pool_72_v3` completion evidence and own full5700 selected-annotation coverage. Control0 macro-BEST is E59; trial4 macro-BEST is E54. LAST is E60 in both. Saved native per-photo counts were bound to their actual model payloads by the typed proof. They are not newly enriched final-runtime validation reports.

## Against the frozen strongest original pipeline

| Candidate role | Macro Dice (delta) | Vessel Dice (delta;95%CI) | Conditional AV (delta) | Worst of6 pooled ratios |
|---|---:|---:|---:|---:|
| control0 training_best | 0.801281 (+0.134252) | 0.760570 (-0.009553;[-0.014029,-0.005054]) | 0.801055 (+0.052489) | 0.987595 |
| control0 fixed_terminal60 | 0.801188 (+0.134159) | 0.758714 (-0.011409;[-0.015871,-0.007012]) | 0.808460 (+0.059894) | 0.985185 |
| trial4 training_best | 0.815057 (+0.148028) | 0.763145 (-0.006978;[-0.014658,+0.000704]) | 0.823922 (+0.075356) | 0.990939 |
| trial4 fixed_terminal60 | 0.804570 (+0.137540) | 0.750500 (-0.019623;[-0.026023,-0.014154]) | 0.823318 (+0.074752) | 0.974520 |

All four vessel Dice values remain below the frozen original. Macro gains therefore do not establish a win on every task. Trial4 BEST has materially better saved VAL performance than its LAST60; selecting an early macro-BEST cannot be presented as the study terminal worst6 objective.

## Matched vascular4 roles and V2 terminal control

| Pair | Macro delta [95%CI] | Vessel delta [95%CI] |
|---|---:|---:|
| control0-training_best-vs-v2terminal | +0.012518 [+0.008582,+0.017018] | +0.010098 [+0.007579,+0.012909] |
| control0-training_best-vs-vascular4best | -0.004696 [-0.009888,+0.002523] | +0.000183 [-0.001175,+0.001565] |
| control0-fixed_terminal60-vs-v2terminal | +0.012425 [+0.007103,+0.016176] | +0.008242 [+0.005740,+0.010782] |
| control0-fixed_terminal60-vs-vascular4last | +0.000354 [-0.005763,+0.006618] | +0.005528 [+0.002612,+0.009145] |
| trial4-training_best-vs-v2terminal | +0.026295 [+0.022002,+0.030668] | +0.012673 [+0.006834,+0.018650] |
| trial4-training_best-vs-vascular4best | +0.009081 [+0.002809,+0.016713] | +0.002758 [-0.002249,+0.007008] |
| trial4-fixed_terminal60-vs-v2terminal | +0.015807 [+0.012022,+0.019626] | +0.000028 [-0.004110,+0.003114] |
| trial4-fixed_terminal60-vs-vascular4last | +0.003735 [-0.002956,+0.011229] | -0.002686 [-0.006957,+0.001464] |

All five pooled heads, source-macro heads, conditional/end-to-end AV, exact source/head TP/FP/FN/valid counts and per-source intervals are retained in each paired JSON. Worst6 is the minimum candidate/reference ratio across pooled vessel/artery/vein/disc/cup Dice and conditional AV accuracy; only its ratio against the frozen original corresponds to the V3 study objective.

## Scope and qualifications

- Fixed threshold0.5, legacy25 raw canonical geometry, tiled 512/64 and FP32 inference are shared by unified variants; actual training configurations (including BF16 and worker/scalar differences) remain in count-view provenance. No causal single-factor claim follows from cross-run comparison.
- Original recipe is the calibrated provided-FOV AV + dedicated raw Disc0 + universal Cup clipped inside Disc0 pipeline; historical training exposure is unresolved and preprocessing/backend differ. Its original policy and report SHA are preserved.
- Source72 native Meta counts and source6b/source0b independently fresh reference reports were produced in different environments/devices. The inference semantics and actual GT count domain are matched; identical backend numerics are not claimed.
- 2000 paired source-stratified bootstrap draws, seed42, resampling612 reviewed split groups preserve known patients/photo aliases; unknown patient independence is not established. Intervals are exploratory after adaptive VAL tuning, conditional on these models, not training-seed uncertainty or multiplicity-corrected clinical evidence.
- The canonical corpusv2 has2529 eligible TRAIN photos; VAL770 remains unchanged. No TEST images/masks were read or scored; the fixed manifest metadata includes all partition identities.
- Completion proofs establish present evidence consistency, not cryptographic proof that TEST has never been seen. Control0 LAST auditor process exit is UNKNOWN as explicitly recorded; its proof was emitted and checked, and the original trainer/PBS exit0 is separately observed. Control0 BEST and both trial4 audit exits are parent-observed0.
