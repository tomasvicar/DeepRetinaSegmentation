# Fresh full original VAL reproduction

Actual run from frozen source `193662c`, completed 2026-11-11 00:17 UTC;
770 validation photographs, no TEST evaluation. The fixed pipeline uses native
vessel, native disc0, universal cup intersected with that same disc0 prediction,
and calibrated A/V with the provided canonical FOV and predicted vessel support.

`validation.json` SHA256:
`53b5c0a64a8aeccb96774c965025093607299bd378a18406206d308ca40a04c6`.

`comparison.json` compares the fresh report with the previously measured strongest
original VAL reference (SHA256 `528233fe50876b15f127f8567fc43d7b8a34a8713cfceff54f02956a0a0bfa2c`).
All 770 photo/annotation inventories, all 3850 photo/head TP/FP/FN/valid counts,
and all four numeric conditional A/V count fields match exactly. For 728 non-A/V
photos the dictionary representation differs (empty versus explicit zeros);
these are metadata schema differences, not different numeric scores.

Macro dataset/head Dice is 0.6670315865775472. Pooled vessel Dice is
0.7701233997938133; conditional A/V accuracy is 0.7485732693037006 on
2,626,284 known pixels. This is VAL reproduction, not TEST performance or a
claim of complete MATLAB raw-image numerical equivalence.

A separate frozen-source run evaluates the subsequently verified MATLAB nearest
resize correction. Its result must be retained with its own source identity;
this historical report is not rewritten or retagged to include that correction.
