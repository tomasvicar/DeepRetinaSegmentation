# Fresh raw256 VAL: lower than the parent raw384 model

Retain the current model/defaults. Raw256 BEST52 and LAST60 both scored below the parent first-TPE BEST58 on this paired VAL comparison. No checkpoint, threshold, inference default or model release is changed. TEST was not used.

Both actual fresh evaluations finished on 2026-11-11 with original observed child and external supervisor return codes 0. Together they took 443.677s including setup, metadata checks and cleanup. Each evaluated all 770 VAL photos; source and input hashes remained unchanged. Original completion is curated, not native V3 or standalone72. Both selected raw256 checkpoints independently had all 5,700 eligible TRAIN annotation IDs and 2,529 photo IDs positively represented at their own selected epochs. This is distinct from merely having full terminal-run coverage.

Reference is the already completed **Old fresh NONE** evaluation of first-TPE BEST58 (`20d4caab…`), original report SHA `195e86548f74401ab8c63ee087ca2482821d88519f2b49303d65d900511d653c`. Raw256 roles are BEST52 (`132c7879…`) and LAST60 (`b60be82c…`). Every comparison uses RAW25, FP32 without autocast, NONE, tile512/overlap64/threshold0.5, identical data identity and inference protocol. Full configuration and photo/source descriptors are retained. The sole serialized training-config difference is patch384→256. Parent was trained at source72/evaluated at1f723; raw256 was trained/evaluated at fd44. Different source builds, training/runtime hardware and individual training realizations remain qualifications: this is not a seed-averaged, source-independent causal estimate of patch size.

Existing `deepretina.comparison.compare_reports` used 2,000 source-stratified paired draws, seed42, over 612 reviewed split groups. It required exact photo/record/source attribution and GT-valid/positive support. All heads, pooled and macro-source results, conditional/end-to-end AV and all per-source intervals are included. Intervals are exploratory VAL summaries conditional on these selected checkpoints, not independent TEST evidence, training-seed uncertainty or proof of unknown-patient independence.

|Metric|Parent raw384|BEST52 raw256|Δ BEST52 [95% CI]|LAST60 raw256|Δ LAST60 [95% CI]|
|---|---:|---:|---:|---:|---:|
|macro dataset/head Dice|0.81089742|0.80123186|-0.00966556 [-0.01466524, -0.00419559]|0.79943252|-0.01146490 [-0.01843136, -0.00303751]|
|vessel pooled Dice|0.76287902|0.75943735|-0.00344167 [-0.00726836, -0.00079767]|0.75821587|-0.00466314 [-0.00784413, -0.00259255]|
|artery pooled Dice|0.73032355|0.70784684|-0.02247672 [-0.03329546, -0.01208663]|0.71606133|-0.01426222 [-0.02110593, -0.00807211]|
|vein pooled Dice|0.79211047|0.77587916|-0.01623131 [-0.02270444, -0.00873703]|0.77601385|-0.01609662 [-0.02148158, -0.01086617]|
|disc pooled Dice|0.93267149|0.93322281|+0.00055132 [-0.00882536, +0.00840232]|0.92866113|-0.00401036 [-0.01605098, +0.00595349]|
|cup pooled Dice|0.88078398|0.86657012|-0.01421386 [-0.03614048, +0.00160638]|0.84822386|-0.03256012 [-0.06461335, -0.00963881]|
|conditional AV|0.82491954|0.80150205|-0.02341750 [-0.03076571, -0.01590718]|0.80457445|-0.02034510 [-0.02531547, -0.01539574]|
|end-to-end AV|0.69963111|0.68370329|-0.01592783 [-0.02363610, -0.00733795]|0.68608231|-0.01354880 [-0.01923209, -0.00817045]|

BEST52 has negative intervals for macro, vessel, artery, vein and conditional AV. Its small pooled-disc gain crosses zero; cup is lower with an interval crossing zero. LAST60 additionally has a negative cup interval. These results do not support promotion or improvement of all tasks.

All adverse source/task differences are listed, including conditional AV: 28 negative BEST52 point differences (9 unadjusted intervals below zero), 25 for LAST60 (15). Notable BEST52 losses include HRF artery −0.04115 and conditional AV −0.04002 (9 photos/3 groups), AVRDB vein −0.02136 and conditional AV −0.02068, and Erlangen vessel −0.01199. G1020 cup −0.05278 crosses zero. LAST60 loses AVRDB artery −0.03469, IOSTAR vein −0.02984, Erlangen vessel −0.01660 and REFUGE cup −0.01502. Full supported-photo/group counts and intervals are retained; source intervals are descriptive without multiplicity correction.

Saved BEST52 macro was 0.801233054810338; actual fresh macro is 0.8012318632503567 (−1.19156e−6). Conditional AV changes +3.04613e−6. All 770 photo/record/source/GT-valid/positive supports match, but prediction counts are **not identical**: changed-count photos vessel84/artery37/vein36/disc199/cup93; AV36. The full integer count deltas and paired drift report are preserved. This small numerical drift is not retraining or a quality gain; no specific runtime cause is asserted.

Scientific exports preserve every original per-photo count and configuration, omitting operational environment/GPU UUID fields. They are explicitly derived scientific files, not original gate-input bytes or newly fabricated receipts. Original whole-report SHAs and export SHAs are separate in artifact-provenance.json. Paired comparison numbers are unchanged; their report SHA fields bind the original private derived views. Reproduction from exported count views gives the same numbers with different view hashes. No source, model weights or new NN execution is included in this export.
