# Completed standalone TPE72: fresh VAL770

The actual original source72 standalone run completed all60 epochs; original outer CLI exit0 and generic4b full60 QA exit0 were observed. BEST58 and LAST60 each cover all5700 selected annotation records and2529 TRAIN photographs. This is a genuine standalone family, not a retagged pool receipt.

Two independently fresh VAL770 evaluations used the frozen source72 producer, same physical local GPU,8 intra/1 interop threads,FP32/noautocast, tile512/overlap64,threshold0.5 andseed42. Both validation CLIs and their outer supervisor returned observed0;91.41s LAST and91.12s BEST. Original checkpoint/config/terminal/protocol/QA bytes were copied into an independent readonly snapshot and remained unchanged. No new training, TEST, selection freeze or release change occurred.

## Against the frozen strongest original

| Role | Macro Dice | Vessel Dice(delta;95%CI) | Conditional AV | Worst6 ratio |
|---|---:|---:|---:|---:|
| best | 0.810895 | 0.762882 (-0.007241;[-0.014591,-0.000693]) | 0.824912 | 0.990597 |
| fixed_terminal60 | 0.808833 | 0.760803 (-0.009321;[-0.018376,-0.001638]) | 0.821755 | 0.987897 |

Vessel Dice remains below the original in both roles, including these exploratory paired intervals. The increased macro score and conditional AV accuracy do not establish superiority on every task. Worst6 is the minimum ratio across five pooled-head Dice and conditional AV accuracy; the vessel head remains the bottleneck. BEST is not the fixed-terminal study objective.

## Other qualified prior models

| Pair | Macro delta [95%CI] | Vessel delta [95%CI] |
|---|---:|---:|
| best-vs-control_terminal60 | +0.022133 [+0.014456,+0.026939] | +0.012410 [+0.006993,+0.017250] |
| best-vs-vascular2_best | +0.006997 [-0.000601,+0.010815] | +0.006169 [+0.002260,+0.009360] |
| fixed_terminal60-vs-control_terminal60 | +0.020070 [+0.011089,+0.025173] | +0.010331 [+0.003420,+0.015920] |
| fixed_terminal60-vs-vascular2_terminal60 | +0.010432 [+0.004004,+0.015420] | +0.004580 [-0.000386,+0.008239] |
| best-vs-vascular4 | +0.004919 [-0.001841,+0.012222] | +0.002495 [-0.002733,+0.006084] |
| best-vs-V3control0 | +0.009614 [-0.001367,+0.015538] | +0.002312 [-0.003556,+0.006496] |
| best-vs-V3trial4 | -0.004162 [-0.012850,-0.000147] | -0.000263 [-0.002350,+0.001635] |
| fixed_terminal60-vs-vascular4 | +0.007998 [-0.000965,+0.016480] | +0.007617 [+0.001139,+0.013274] |
| fixed_terminal60-vs-V3control0 | +0.007644 [+0.001786,+0.012663] | +0.002089 [-0.005442,+0.007424] |
| fixed_terminal60-vs-V3trial4 | +0.004262 [-0.003791,+0.008842] | +0.010302 [+0.006569,+0.014042] |

All12 pairs passed the exact770-photo/612-known-group membership, per-photo record/source attribution, every head valid/positive GT support, conditional AV eligible/scored domain and pooled/source count reconstruction. Each paired JSON retains all five heads, source-macro and pooled statistics, conditional/end-to-end AV, exact TP/FP/FN/valid counts and per-source uncertainty. The existing source72 standalone checker bound the fresh candidate to its original generic completion QA/protocol/terminal and selected role. Prior genuine pool72 count views retain their previously verified original typed eligibility provenance; they were not converted into standalone QA.

## Small saved/fresh numerical differences

Fresh counts are authoritative for these fresh reports. Global bitwise count parity is explicitly false: LAST changed6 photos/7 head-photo entries and macro−4.29046e−8; BEST changed5 photos/7 entries and macro−1.00598e−7. All photo IDs, annotation/source attribution, valid pixels and GT-positive counts remain exact. The exact integer deltas are published separately.

Fresh actual flags: deterministic algorithmsFalse/warn-onlyFalse,cudnn benchmarkFalse,cudnn deterministicFalse,matmul TF32False,cudnn TF32True. The pinned training implementation calls deterministic algorithmsTrue/warn-onlyTrue and cudnn benchmarkFalse; historical flags were not separately measured. Training read the immutable derived cache; fresh evaluation read prepared RGB without that cache. No extra prediction/counterfactual was run to assign a cause.

## Qualifications

- These are exploratory development VAL results after adaptive proposal/selection. Percentile95% intervals use2000 shared paired source-stratified resamples,seed42,of612 known split groups. They do not prove unknown-patient independence, clinical generalization, training-seed robustness or multiplicity-corrected discovery.
- BEST versus BEST uses different selected epochs; only LAST60 comparisons match epoch budgets. Configurations, devices, training precision, augmentation, samplers and worker counts can differ across models. No single-factor causal claim is made.
- Original calibrated AV/dedicated Disc0/clipped universal Cup reference has unresolved historical training exposure and different preprocessing/backend. Source72 native Meta counts and fresh local FP32 counts are distinguished; no backend bitwise identity is claimed.
- The canonical corpusv2 manifest includes all partition metadata. No TEST image/mask was read or scored. No model binary, operator host/storage path or physical GPU UUID is published here.
- Original whole-report/checkpoint/QA/protocol/terminal SHAs are retained. The count views are derived public artifacts and cannot substitute for the original enriched evidence files or typed completion receipts in a final selection gate.
