# Reviewed release: unified-v2-control

This adds a separately named **fixed terminal E60 V2 control**, not an optimized
final model or all-task win. It preserves the original nine PT files and aliases.
Root reviewed preview commit `94f73da` and fast-forwarded it into `for_Leipzig`. The [model card](../../../docs/UNIFIED_V2_CONTROL.md) has the colleague
prediction command and per-head caveats.

The published 14,385,264-byte checkpoint is the unchanged terminal `last.pt`, SHA
`10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2`.
It has actual contribution by all 2,529 TRAIN photos and 5,700 annotation records,
with at least 66 visits per photo and 2 contributions per annotation. The VAL-best E52 has only 5,699 contributors;
it is a reference in metadata, not the bundled all-annotation model.

Evidence in this directory:

- `qa-summary.json`: public-safe strict terminal completion/coverage/source checks.
- `metrics.csv`: actual 60-epoch training/VAL history, without operational receipts.
- `fresh-validation-summary.json`: independently produced E60 VAL evidence,
  preserving the original full report's SHA and excluding per-photo arrays/images.
- `paired-comparison-summary.json`: fixed-terminal baseline-role paired comparison,
  all five pooled/macro head intervals and conditional AV, including qualifications.
- `preserved-models.json`: original nine model hashes and aliases, byte-verified.
- `package-check.json`: 59 focused CPU tests and one built wheel with all 10 model SHA checks.
- `real-cli-smoke.json`: actual raw TRAIN DRIONS CPU CLI, original-frame output
  checks and file hashes only; no raw photograph, prediction image or GT scoring.
- `real-gpu-cli-smoke.json`: one GTX 1070 raw TRAIN CLI check; finite outputs and
  hierarchy pass, zero binary-mask disagreements with CPU on this case, explicitly
  non-bitwise floating probabilities. No training, GT score or other GPU used.
- `release-plan.json`: prior draft ownership/gates completed with actual pins;
  root review and completed fast-forward are explicit.

Fresh VAL macro 0.788763 exceeds the frozen original reference 0.667030, while
pooled vessel 0.750472 is lower than 0.770123 (paired difference interval excludes 0).
Artery's point estimate is lower too, with interval crossing 0. Vein/disc/cup and
conditional AV are higher. No blanket all-task, clinical or TEST claim is made.
The original prepared-input numerical ports and all original aliases remain.

The published model carries geometry policy `2591d846`; the DRIONS smoke records its
nominal 30° assumption. No private worker/Meta endpoints or operational receipts
are copied into these public reports. Runtime sources of the live experiments
and immutable data remain unchanged. Further optimization is ongoing separately.
