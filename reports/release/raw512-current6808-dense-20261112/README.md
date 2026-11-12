# Exact current original6808: dense-vessel VAL uncertainty

This new **saved-count** computation binds the actual final-source original VAL
report `6808f293d750dec5a94a0593f5c52d054733696d21984a61e399284355f94405`
and selected raw512 BEST50 rot4/.35 fresh report
`0a5c65e6ecd0ceef68bba554504f0d2e2222905d5ebf27813b07242cb3e7aa0f`.
It does not substitute or retag historical original182 counters. Frozen model,
selection and original policy remain unchanged; no TEST result was read.

| Dense87 vessel support | Original6808 | Selected raw512 rot4/.35 |
|---|---:|---:|
| TP | 4,681,051 | 4,613,808 |
| FP | 1,723,029 | 1,775,123 |
| FN | 1,211,956 | 1,279,199 |
| Dice | .7613268085360378 | .7513159568139817 |
| Precision | .7309482392474798 | .7221564922206860 |
| Recall | .7943399693908390 | .7829293262336189 |

Candidate-minus-original Dice is **−.01001085172205618**, paired95% interval
**[−.014354340037928445,−.006563583431137865]**. Both precision and recall are
lower. The displayed interval happens to match the previous historical-baseline
interval numerically, despite original6808 differing by2TP/2FN pixels. This
report explicitly rechecks current6808 rather than assuming equivalence.

Exact domain:87 photos,55 known split groups,8 sources,5,893,007 GT positives and
60,325,745 valid pixels; exclude ONLY positive-only AVRDB. All770 photo/record/
source memberships, every head's GT support and integer pooled sums matched
before the dense subset/bootstrap. Existing source-stratified known-group
bootstrap code was reused unchanged with2,000 draws/seed42. See
[full numbers, source/helper SHA and strata](analysis.json) and
[externally observed corrected command exit0](actual-exit.json).

Actual CPU1/core4/nice17/idleIO execution finished in under1s with a120s cap.
Only JSON report/manifest metadata and saved counters were read. No model,
image, GT array, GPU, inference, training or heldout evaluation occurred. An
initial parser assumed dict-form manifest; actual list form raised TypeError
before bootstrap. Its failure is retained privately, not reported as PASS.
The corrected supported list/dict parser has its own actual0 receipt.

This is postfreeze uncertainty bookkeeping, not another selection decision or
proof of clinical utility, equivalence or noninferiority. AdaptiveVAL reuse and
multiplicity remain limitations. Known split groups do not establish complete
patient independence; these intervals do not measure training-seed uncertainty.
Historical reports are preserved; the original vessel model remains preferable
for the measured vessel-priority objective.
