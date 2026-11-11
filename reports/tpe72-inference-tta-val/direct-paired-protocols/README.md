# Direct paired TTA protocol differences

This saved-count analysis directly compares **dihedral8 minus rot4** on the same immutable TPE BEST58/source1f723 reports, plus the inexpensive rot4-minus-flips2 contrast. All770 photos,612 known groups, aliases/annotation-source/head/conditional-AV supports, checkpoint20d4, configuration, actual runtime and non-TTA protocol fields match exactly. These are **direct paired differences in each shared bootstrap resample**, not differences between marginal versus-none intervals. No model, Torch import, new inference, GT-array decoding, TEST or image access occurred.

| Direct contrast | Macro Dice difference [paired95% CI] | Conditional AV difference [paired95% CI] |
| --- | --- | --- |
| dihedral8-minus-rot4 | +0.000798561 [+0.000285578,+0.001381514] | +0.001696313 [+0.000794524,+0.002644361] |
| rot4-minus-flips2 | +0.002968317 [+0.001909028,+0.003936313] | +0.002894965 [+0.000997779,+0.004899515] |

## All five pooled heads: eight minus four

| Head | Pooled Dice difference [paired95% CI] |
| --- | --- |
| vessel | +0.000194014 [+0.000094679,+0.000288590] |
| artery | +0.001577477 [+0.000363151,+0.002799104] |
| vein | +0.001202870 [+0.000271820,+0.002104763] |
| disc | +0.000281856 [-0.000102030,+0.000634882] |
| cup | -0.000026922 [-0.000401480,+0.000330251] |

The additional views have a small positive observed macro/conditional-AV difference with these exploratory intervals. They do not improve every pooled head: cup has a slightly negative point estimate, and its interval and the disc interval cross zero. Source-macro artery/disc/cup intervals also cross zero. An interval crossing zero does **not** establish equivalence or noninferiority.

## Adverse source points are retained

| Contrast | Source/head | Dice difference [paired95% CI] | Supported photos/groups | Defined draws |
| --- | --- | --- | --- | ---: |
| dihedral8-minus-rot4 | aria/vessel | -0.000017585 [-0.000195228,+0.000151032] | 28/16 | 2000 |
| dihedral8-minus-rot4 | drhagis/artery | -0.000720123 [-0.004832350,+0.006029995] | 3/3 | 1951 |
| dihedral8-minus-rot4 | drishti/disc | -0.000039444 [-0.000740397,+0.000682171] | 10/10 | 2000 |
| dihedral8-minus-rot4 | refuge/cup | -0.000197781 [-0.000596935,+0.000186926] | 400/400 | 2000 |
| rot4-minus-flips2 | rimone/cup | -0.000338911 [-0.008912180,+0.008084116] | 32/32 | 2000 |

All pooled/source-macro five-head statistics, exact aggregate source/head counts, conditional/end-to-end AV, support sizes and defined draw counts are retained in [results.json](results.json). SHA bindings preserve original whole reports/kinds/protocols. The explicitly temporary count projections were removed and never became new enriched validation evidence or eligibility receipts; original raw reports and prior versus-none analysis were unchanged. Operational paths/physical UUIDs/photo identifiers are excluded.

The frozen comparison math used2000 source-stratified paired group draws,seed42. Computation was CPU1/nice17/affinity6/CUDAhidden with actual exit0. These are exploratory unadjusted adaptiveVAL observations for one fixed trained checkpoint, not training-seed uncertainty, unknown-patient independence, clinical generalization or a final selection. Multiple comparisons remain uncorrected.

[Original four-protocol VAL results](..) and [separate one-photo CPU timing](../../benchmark/tpe72-tta-cpu-drive21) remain distinct evidence. This analysis supplies no latency CI and makes no claim that four and eight views have equal quality. No default, release, model freeze or TEST decision is made.
