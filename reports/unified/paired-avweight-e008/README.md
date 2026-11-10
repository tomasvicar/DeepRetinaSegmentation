# Paired E8 loss-weight comparison, saved validation counts only

Weight2 has the highest recorded E8 point source-macro score, 0.714549 versus
0.708936 for matched weight1. Its primary difference is +0.005613, with paired95%
interval **[-0.001368,+0.011723]**, which includes zero. This supports considering
a reviewed full-budget follow-up; it does not establish a superior final model.
An already-running weight1 control is not a reason to ignore the candidate.

The controlled training configs differ only in `av_weight`1→2: same source87,
old/GTX1060, LR0.0003, seed42, mild augmentation, workers2 and canonical6103 data.
Both terminal E8 checkpoints and identical photo/GT support were independently
verified earlier. All770 VAL photos and612 reviewed split groups are paired.

| Dice delta, weight2−weight1 | Pooled delta [paired95% CI] | Source-macro delta [paired95% CI] |
|---|---|---|
| Vessel | −0.008120 [−0.010449,−0.005621] | −0.012415 [−0.014880,−0.009481] |
| Artery | +0.013126 [−0.006717,+0.029916] | +0.020198 [+0.000102,+0.039962] |
| Vein | +0.003390 [−0.004899,+0.011115] | +0.003474 [−0.027972,+0.014051] |
| Disc | +0.015931 [+0.008495,+0.023233] | +0.012772 [+0.004837,+0.019904] |
| Cup | +0.008300 [−0.001209,+0.018557] | +0.004036 [−0.005338,+0.014347] |

The mean of five pooled Dice values changes by+0.006525 [0.000408,0.012186].
Vessel performance decreases under both summaries; the higher overall point
score partly reflects disc/cup gains. Most pooled AV intervals include zero.
Conditional AV accuracy changes from0.652636 to0.661485 (+0.008849), **point only**:
source87 saved reports lack per-photo ungated correct/scored counts, so no paired
conditional-accuracy CI can be recovered from gated TP/FP/FN.

The2000 draws (seed42) resample reviewed `split_group_id` within source-signature
strata; both checkpoints and all heads share every draw. These groups preserve
recoverable patients/aliases but are not proven independent patients. Intervals
are exploratory, conditional on these two seed42 realizations and unadjusted for
selection among multiple E8 candidates, endpoints or epochs. They do not estimate
training-seed uncertainty or predict the full60 ranking. No TEST scores were read.

The existing pinned count-only comparison module was copied locally with one
explicit extension: append the mean of pooled head Dice to the same bootstrap
score vector. Original macro/per-head procedures remain unchanged. The copied
module has its own SHA; it does not modify production comparison code. Two tiny
closed-form fixtures verify zero self-difference and a uniform+1/3 difference for
pooled mean, macro and every head. Tests passed in0.05s; actual comparison took
0.681s on Ada CPU1/nice10/CUDAhidden. No image, mask, checkpoint or model was loaded.

`result.json` contains all point values/intervals, group strata, input/checkpoint
hashes and qualifications. `provenance.json` records the config-only comparison
and runtime. `run_comparison.py` is the exact saved-count producer and refuses
unrecognized report/manifest/helper hashes. Its assertions require normal Python
execution (do not use`-O`). No clinical or final superiority claim is made.
