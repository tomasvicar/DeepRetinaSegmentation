# Completed V2 run: fresh VAL of its best E52 snapshot

The actual control run completed all 60 epochs and all 2,529 TRAIN photographs /
5,700 annotation records from 21 sources. Strict finite-checkpoint/optimizer QA,
full-budget history and independent immutable milestone verification passed.
`completion-qa.json` is the unchanged original typed run-completion evidence.

**The validation-best E52 snapshot itself contains only 5,699/5,700 contributing
annotation IDs.** It covers all 2,529 photographs and all 21 sources. One RIGA
observer variant had not contributed by that snapshot. See `selected-coverage.json`.
Do not describe the best snapshot as having seen every annotation. The terminal
E60 snapshot does have full contribution and is evaluated separately as a clearly
named fixed-terminal baseline. This report does not authorize final selection/TEST.

Best checkpoint SHA:
`9df2c04ee85b519c5d7f9965f098f56b627bd2524bf8f7e3fea718c117add35c`.
Fresh enriched VAL SHA:
`92663098395dea48816b598860f1accb4515b5cb5bd38b6c5aad69fe7b806691`.
Producer: unchanged frozen training source6b3dfad, CPU4. The small difference from
the training GPU score is preserved rather than substituted by a historical score.

## Measured comparison against the fixed original pipeline

Both reports cover the same 770 VAL photographs, annotation IDs and valid/positive
GT domains. Each pipeline is bound to its own reviewed context; original source
d14aa14 includes the independently verified nearest correction. Confidence
intervals use 2,000 source-stratified paired split-group draws, seed42.

| Metric | Original | V2 best E52 | Difference / paired 95% CI |
| --- | ---: | ---: | --- |
| Macro dataset/head Dice | 0.667030 | 0.797217 | +0.130188 [0.121783, 0.140943] |
| Pooled vessel Dice | 0.770123 | 0.757961 | **−0.012163** [−0.016918, −0.007573] |
| Conditional A/V accuracy | 0.748566 | 0.792155 | +0.043589 [0.030825, 0.056446] |

The saved comparison contains all five head/source counts and intervals, including
disc/cup. Aggregate improvement does not satisfy the vessel objective. This is
single-seed, validation-selected exploratory evidence, not independent TEST proof
or a claim that all patients are independent. Keep original models available.

`reviewed-context.json` and its SHA sidecar bind the complete original enriched
report and typed completion bytes. Their role is explicitly `best`; the new
fixed-terminal route requires a different explicit context and candidate SHA.
All inputs remain unchanged. No model forward or GT-array read occurs in the
comparison itself; actual fresh VAL production is separate and recorded by
`producer-terminal.json`.
