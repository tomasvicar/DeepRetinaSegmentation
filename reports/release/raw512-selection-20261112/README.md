# Frozen raw512 selection before held-out evaluation

On 12 November 2026 at13:34:52UTC, the unchanged research6d create/check and
coupled original-reference freeze completed with actual exits0. Original checkpoint
SHA and both genuine frozen document SHAs are recorded in selection-summary.json.
The selected original is raw512 training_best **epoch50**, from the completed
60-epoch run; its own coverage is2,529 TRAIN photos /5,700 annotation records
from21 sources. Inference is FP32 **rot4 / threshold0.35 / tile512 / overlap64**.
No TEST evaluation was used for this selection.

On770 VAL photos, macro Dice is0.82283410 and conditional A/V0.85119431.
On the identical dense vessel support (87photos,55known groups,8sources),
Dice is**0.75131596**, versus original**0.76132681**. Difference−0.01001085,
paired95% interval[−0.01435434,−0.00656358]; both precision and recall are lower.
These exploratory VAL intervals do not prove complete patient independence.
Original vessels remain preferable when measured vessel quality is the priority.
The scientific vessel-improvement goal remains unmet.

The finite width24 standard validation and eight NONE thresholds did not improve
this dense-vessel result. Native V3 had a slightly higher artery score; this is a
balanced smaller five-head alternative, not universal superiority or a global
optimum. Existing originals and previews stay separately available.

[Raw512 threshold curve](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/raw512-BEST50-rot4-threshold-calibration-20261112)
and [completed width24 curve](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/capacity24-BEST58-NONE-threshold-calibration-20261112)
retain adverse results. The private operations repository preserves full original
completion/selection/reference/command/exit receipts. The selected original bytes
and a subsequently exported deployment artifact have distinct SHAs.

At this commit, held-out856 TEST, matched final latency and deployment packaging
remain pending. This report is selection evidence, not a completed release claim.

[Current original6808 dense-vessel uncertainty](../raw512-current6808-dense-20261112/README.md)
was independently recomputed from the exact saved VAL counters after freeze.
It confirms the negative interval without changing selection, source or policy;
no heldout result or new inference informed that bookkeeping check.
