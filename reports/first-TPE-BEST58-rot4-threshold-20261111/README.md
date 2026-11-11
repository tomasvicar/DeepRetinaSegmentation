# Actual first-TPE BEST58 rot4 scalar-threshold VAL sweep

Observed external exit0 and science child exit0; all770 VAL photos completed in685.42s (science683.81s). Original checkpoint20d4caab was trained under standalone source72; evaluation source318e342/package198fdd47 uses FP32 tiled512/overlap64 and rot4 joint-AV fusion. This is exploratory threshold calibration of the same fixed BEST58 model, not a new checkpoint selection. No TEST/default changes.

Eight global scalar thresholds reused one probability inference per photo (four network views), with exact native0.5 counter parity on all770. Conditional AV is threshold invariant. Objective: maximum minimum of six pooled ratios versus reviewed original182ad346, ties closest to0.5.

| Threshold | Macro | Vessel | Artery | Vein | Disc | Cup | Conditional AV | Worst6 ratio |
|---|---|---|---|---|---|---|---|---|
| 0.250000000 | 0.822824616 | 0.766135148 | 0.769739356 | 0.822255173 | 0.934872664 | 0.881524477 | 0.836113307 | 0.994821282 |
| 0.300000000 | 0.822774827 | 0.768455490 | 0.764666365 | 0.818451263 | 0.936239031 | 0.883301916 | 0.836113307 | 0.997834231 |
| 0.350000000 | 0.822099157 | 0.769463926 | 0.759658718 | 0.814793346 | 0.936902210 | 0.884338615 | 0.836113307 | 0.999143678 |
| 0.400000000 | 0.821095722 | 0.769449263 | 0.754633826 | 0.811124210 | 0.937179868 | 0.884868378 | 0.836113307 | 0.999124638 |
| 0.450000000 | 0.819700014 | 0.768539168 | 0.749443529 | 0.807300221 | 0.937251573 | 0.885089195 | 0.836113307 | 0.997942886 |
| 0.500000000 | 0.818055221 | 0.766953119 | 0.743988548 | 0.803239842 | 0.937128824 | 0.885206589 | 0.836113307 | 0.995883412 |
| 0.550000000 | 0.815858464 | 0.764541479 | 0.738090176 | 0.798820582 | 0.936660501 | 0.884889485 | 0.836113307 | 0.992751913 |
| 0.600000000 | 0.813189763 | 0.761226596 | 0.731539629 | 0.794128448 | 0.935832786 | 0.884228891 | 0.836113307 | 0.988447561 |

The exploratory selection is0.35. Vessel0.769463926 remains below original0.770123400 (delta−0.000659473, paired95%CI[−0.006530228,+0.005516923]); worst6 ratio0.999143678 remains below1. The grid does not meet the all-six point-estimate target.

Versus default0.5, selected0.35 increases pooled vessel by0.002510807 (CI[−0.000030872,+0.005286348]), artery by0.015670169 and vein by0.011553504. Disc decreases0.000226614 and cup0.000867974, both pooled intervals cross0; disc macro decreases0.003022001 with interval below0. Conditional AV remains0.836113307. There are11 negative source/head point deltas versus0.5,6 with upper95%CI below0, including DRHAGIS/CHASE vessel and G1020/RIMONE disc/cup.

Versus original there are8 adverse source/head points,5 intervals fullynegative (vessel on DRHAGIS, CHASEDB1, ARIA, HRF, RITE). All head/source/conditional-AV intervals and per-photo counts are retained. Paired bootstrap2000/seed42 uses612 known split groups; these are not proven complete independent patients. No CI is claimed for the nonlinear worst6 minimum. Grid selection and interval interpretation carry multiplicity/selection bias; historical original exposure/preprocessing/backend qualifications remain. No independent all-VAL or held-out quality improvement claim.

Original files are byte-preserved, with local-versus-remote SHA proof. No models are stored in this report.
