# Measured TRAIN-only BN recalibration, exploratory VAL770 result

The fixed parent gradient60/all 5700 model was recalibrated using TRAIN only;
all learned parameters/non-BN states remained byte-identical. See the separate
TRAIN summary. A once-only actual frozen6b CPU4 evaluation then measured the
exact derived artifact on all 770 VAL photos, threshold 0.5, with unchanged
inference config and canonical/GT support. No TEST was evaluated.

| Pooled metric | Original d14 pipeline | Parent terminal60 | Derived BN |
|---|---:|---:|---:|
| Vessel Dice |0.770123|0.750472|0.755255|
| Artery Dice |0.689522|0.677109|0.677527|
| Vein Dice |0.697286|0.754314|0.759364|
| Disc Dice |0.894808|0.934122|0.935622|
| Cup Dice |0.435399|0.872808|0.874727|
| Conditional A/V accuracy |0.748566|0.788037|0.788139|
| End-to-end A/V accuracy |0.630307|0.642235|0.649706|
| Macro across datasets/heads |0.667030|0.788763|0.792075|

Against parent terminal60, paired grouped/source-stratified 2000-draw seed42
bootstrap macro difference is+0.00331242, descriptive 95% CI
[+0.00220383,+0.00553789]. Pooled vessel difference is+0.00478292,
CI[+0.00381753,+0.00586108], primarily fewer false positives; vessel recall
changes slightly downward. Vein/disc/cup pooled Dice also improve. Pooled artery
and conditional A/V differences have intervals crossing zero: conditional A/V
+0.00010281, CI[-0.00165135,+0.00189365]. End-to-end A/V improves+0.00747139,
CI[+0.00529277,+0.00956108]. Per-source/task counts and intervals are retained.
These repeated-VAL/exploratory intervals do not account for all prior selection,
multiple comparisons or prove independent clinical improvement.

The intervention shows a modest actual improvement over its fixed parent on
this VAL. It does NOT prove that BN noise caused the earlier E52→60 drift:
weights also changed between epochs, and this calibration used basic rather
than mild augmentation. Original vessel quality is still better: derived versus
d14 pooled difference-0.01486831, CI[-0.01913895,-0.01054963]. Artery also remains
weaker in macro-source Dice. Aggregate superiority over the original is driven
largely by disc/cup and vein; the original's historical training exposure remains
incompletely recoverable.

Largest remaining source vessel regressions (original→BN, VAL photos):
DRHAGIS 0.793068→0.758500 (8); CHASEDB1 0.790873→0.764374 (6);
HRF 0.807503→0.785539 (9); RITE 0.813105→0.791698 (4);
ARIA 0.732888→0.711485 (28). Exact existing source intervals are in
`largest-vessel-regressions.json`; no extra inference/bootstrap was run to rank them.

## Actual supervisor defect and evidence boundary

The completed producer's ORIGINAL report has a nested `metrics` object, not the
flat shape initially described in preparation notes. The private supervisor
incorrectly checked top-level `photos` AFTER the producer returned successfully
and crashed with KeyError. No NN run was repeated. The original report, failed
supervisor traceback and sealed deployed helpers were preserved; no original
terminal receipt was fabricated. Producer return0 is explicitly **INFERRED**
from reviewed sealed control flow and traceback, not a retained original terminal.
`postcheck-recovery.json` is a NEW honest metadata-recovery record with kind
`completed_report_failed_supervisor_postcheck` and exact original hashes. This
operational defect is separate from actual completed770/source-postchecked counts.
The future private postcheck was fixed; original deployed helpers remain unchanged.

`validation.json` preserves exact original bytes, SHA
f1c7076c69ef1e380b619cf551311ee8aee3c12a932188ebd127de4c314acc60.
Derived checkpoint SHA:
eecfebac1f9ee77f26de49654d1caf0dd1c5e35a733900bfce85afc3970cabe3.
Calibration JSON SHA:
a16aca3948df32556feb6868e744c0e4d2c80aa3a870a6b3c09811f1baf8e14a.
Generic comparisons use explicit metadata-only unwrapped metric views and retain
both those view hashes and full original wrapper hashes. They do not relabel
this research report as standard training evidence or change original QA.

Only portable metadata/counts are public here: no raw images, checkpoint binary,
operational paths/endpoints or raw process logs. Original and derived weights
remain independently backed up outside Git/OwnCloud. This is a research model,
not a new final selection or TEST authorization, and no shipping alias is changed.
