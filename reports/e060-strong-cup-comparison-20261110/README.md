# Interim corpusv1 E60 baseline versus the completed clipped-cup original reference

The fixed0.5 baseline has a higher five-head source-macro VAL score, but vessel segmentation regresses: pooled Dice0.770123→0.760819 (delta−0.009305, paired95%CI[−0.013968,−0.004556]). Artery source-macro Dice also decreases by0.006052 with an interval crossing zero. The overall increase is dominated by cup and disc, with a vein gain; it does not establish improvement on every task.

This is interim exploratory **corpusv1** evidence, not a final HPO/model selection or independent quality claim. The immutable dataset has17 registered dense TRAIN selectors and15 dataset labels in VAL. The770 VAL photos form612 known split groups. Fourteen selected per-head annotation sources remain after DRIVE/RITE alias and AV-priority annotation handling; DRIVE is a canonical bootstrap stratum while the shared vascular annotations are attributed to RITE. DRIONS-DB and GAMMA labels were subsequently identified for a separate corpusv2 audit; this report does not claim all available annotated archives were used.

The completed60-epoch baseline selected its best checkpoint at epoch56 (SHA`bfe957460dfed168f2344b1b84ef99b7f502e9dbc8ce1fd4d68632c3234708b1`). The terminal60 QA independently records all2221 training photos and491520 delivered samples. It is compared against the declared calibrated four-network reference: dedicated vessels, calibrated original AV, dedicated disc0/raw and universal/raw cup intersected with the same-forward predicted disc0. Original defaults/weights remain unchanged. Historical original training exposure cannot be fully reconstructed.

## Main comparison: fixed threshold0.5

Five-head source-macro score: original **0.667032**, baseline **0.800123**, difference **0.133091**, paired95%CI **[0.123940, 0.143635]**. The score first averages source Dice within each head, then averages the five heads; it is not a pixel-pooled score.

| Head | Original pooled | Baseline pooled | Δ pooled [95%CI] | Original source macro | Baseline source macro | Δ macro [95%CI] |
|---|---:|---:|---|---:|---:|---|
| vessel | 0.770123 | 0.760819 | -0.009305 [-0.013968, -0.004556] | 0.786657 | 0.777935 | -0.008721 [-0.018718, 0.000604] |
| artery | 0.689529 | 0.698608 | 0.009079 [-0.006460, 0.023881] | 0.680873 | 0.674821 | -0.006052 [-0.022219, 0.013605] |
| vein | 0.697298 | 0.769099 | 0.071801 [0.059272, 0.083520] | 0.699547 | 0.751748 | 0.052200 [0.034732, 0.070131] |
| disc | 0.894808 | 0.937701 | 0.042893 [0.035440, 0.051945] | 0.729530 | 0.924394 | 0.194864 [0.181979, 0.209239] |
| cup | 0.435399 | 0.874385 | 0.438986 [0.426201, 0.452111] | 0.438552 | 0.871717 | 0.433165 [0.407708, 0.464899] |

Pooled confusion counts preserve the exact common valid/positive GT domains. Full precision, recall, IoU, count changes and all source/head counts are in the JSON files.

| Head | Original TP / FP / FN | Baseline TP / FP / FN | Δ TP / FP / FN |
|---|---:|---:|---:|
| vessel | 5142829 / 1723029 / 1347170 | 4913376 / 1512648 / 1576623 | -229453 / -210381 / 229453 |
| artery | 809366 / 342002 / 386858 | 730110 / 163852 / 466114 | -79256 / -178150 / 79256 |
| vein | 846023 / 150493 / 584037 | 1031624 / 220998 / 398436 | 185601 / 70505 / -185601 |
| disc | 11598114 / 846577 / 1880332 | 12454133 / 630552 / 1024313 | 856019 / -216025 / -856019 |
| cup | 2868818 / 7304851 / 135394 | 2668856 / 431468 / 335356 | -199962 / -6873383 / 199962 |

## Conditional classification versus vessel gating

All per-photo scored and eligible AV domains match exactly (2,626,284 pixels). Conditional accuracy is measured on those known GT vessel pixels, independently of the predicted vessel gate. Counts also pass exclusive artery/vein decoder constraints and the end-to-end correctness equals arteryTP+veinTP. The following intervals use the same paired group draws as all Dice metrics.

| Metric | Original | Baseline0.5 | Δ [95%CI] |
|---|---:|---:|---|
| conditional_accuracy | 0.748573 | 0.798662 | 0.050089 [0.036145, 0.064373] |
| end_to_end_accuracy | 0.630316 | 0.670809 | 0.040493 [0.027331, 0.052700] |

The conditional classification result is identical at threshold0.35. The end-to-end result changes with vessel gating; those quantities must not be conflated. Per-source conditional/end-to-end counts and paired intervals are also retained in each JSON.

## All supported source/head changes at fixed0.5

Every supported cell is shown, including negative point differences and intervals crossing zero. These are descriptive intervals without a multiple-comparison correction. Cells with few split groups have correspondingly limited uncertainty information.

| Source | Head | Photos / groups | Original Dice | Baseline Dice | Δ Dice [95%CI] | Δ TP / FP / FN |
|---|---|---:|---:|---:|---|---:|
| aria | vessel | 28 / 16 | 0.732888 | 0.714590 | -0.018298 [-0.023781, -0.012884] | -170247 / -162813 / 170247 |
| aria | disc | 22 / 12 | 0.738979 | 0.877765 | 0.138786 [0.074366, 0.205982] | 78598 / -9079 / -78598 |
| avrdb | vessel | 20 / 20 | 0.872294 | 0.954820 | 0.082527 [0.062596, 0.118422] | 83600 / 0 / -83600 |
| avrdb | artery | 20 / 20 | 0.655753 | 0.628106 | -0.027647 [-0.084327, 0.026781] | -10394 / -3498 / 10394 |
| avrdb | vein | 20 / 20 | 0.730565 | 0.767192 | 0.036627 [0.006321, 0.068704] | 50547 / 53457 / -50547 |
| chasedb1 | vessel | 6 / 3 | 0.790873 | 0.764508 | -0.026364 [-0.034555, -0.018208] | -12376 / -3337 / 12376 |
| drhagis | vessel | 8 / 8 | 0.793068 | 0.764802 | -0.028266 [-0.038716, -0.018813] | -13030 / -272 / 13030 |
| drhagis | artery | 3 / 3 | 0.669177 | 0.616979 | -0.052198 [-0.089211, 0.011900] | -4130 / -2652 / 4130 |
| drhagis | vein | 3 / 3 | 0.696295 | 0.695740 | -0.000555 [-0.065679, 0.055132] | 1396 / 2688 / -1396 |
| drishti | disc | 10 / 10 | 0.960409 | 0.975158 | 0.014749 [0.002304, 0.029694] | 2321 / -1752 / -2321 |
| drishti | cup | 10 / 10 | 0.692782 | 0.917031 | 0.224250 [0.135448, 0.339535] | -7215 / -57419 / 7215 |
| erlangen | vessel | 22 / 11 | 0.620504 | 0.634955 | 0.014451 [0.000979, 0.030555] | 38655 / 63315 / -38655 |
| g1020 | disc | 204 / 86 | 0.857905 | 0.936227 | 0.078322 [0.058541, 0.102906] | 427669 / -195082 / -427669 |
| g1020 | cup | 170 / 84 | 0.360485 | 0.832593 | 0.472107 [0.447333, 0.495078] | -32857 / -2128276 / 32857 |
| hrf | vessel | 9 / 3 | 0.807503 | 0.785968 | -0.021535 [-0.031874, -0.016520] | -55424 / -26838 / 55424 |
| hrf | artery | 9 / 3 | 0.713607 | 0.768138 | 0.054531 [0.037607, 0.066666] | -27444 / -110657 / 27444 |
| hrf | vein | 9 / 3 | 0.683103 | 0.812410 | 0.129307 [0.124863, 0.134932] | 96426 / 6207 / -96426 |
| iostar | vessel | 6 / 6 | 0.816907 | 0.806238 | -0.010669 [-0.025761, 0.005653] | -47369 / -53539 / 47369 |
| iostar | artery | 6 / 6 | 0.681296 | 0.651529 | -0.029767 [-0.056609, 0.001318] | -29163 / -34821 / 29163 |
| iostar | vein | 6 / 6 | 0.662059 | 0.690443 | 0.028384 [-0.009339, 0.067105] | 15703 / 8419 / -15703 |
| iostar | disc | 6 / 6 | 0.004701 | 0.889287 | 0.884586 [0.848715, 0.914212] | 154051 / 16768 / -154051 |
| onhsd | disc | 17 / 9 | 0.852989 | 0.914595 | 0.061606 [0.031630, 0.094208] | 18715 / -2248 / -18715 |
| refuge | disc | 400 / 400 | 0.932061 | 0.942258 | 0.010198 [0.007016, 0.013241] | 158916 / 5020 / -158916 |
| refuge | cup | 400 / 400 | 0.461106 | 0.887901 | 0.426796 [0.410448, 0.442457] | -183226 / -4668711 / 183226 |
| rimone | disc | 32 / 32 | 0.759667 | 0.935465 | 0.175798 [0.137559, 0.223970] | 15749 / -29652 / -15749 |
| rimone | cup | 32 / 32 | 0.239833 | 0.849341 | 0.609508 [0.567585, 0.650033] | 23336 / -18977 / -23336 |
| rite | vessel | 4 / 4 | 0.813105 | 0.790363 | -0.022742 [-0.032721, -0.012362] | -41044 / -36676 / 41044 |
| rite | artery | 4 / 4 | 0.684530 | 0.709350 | 0.024821 [0.007257, 0.044528] | -8125 / -26522 / 8125 |
| rite | vein | 4 / 4 | 0.725715 | 0.792954 | 0.067239 [0.051989, 0.087986] | 21529 / -266 / -21529 |
| stare | vessel | 4 / 4 | 0.832768 | 0.785173 | -0.047595 [-0.133958, 0.009112] | -12218 / 9779 / 12218 |

## Separate exploratory threshold0.35

Threshold0.35 was selected from six values on this same VAL. Its score **0.804955** versus original **0.667032** gives difference **0.137924**, unadjusted paired95%CI **[0.128636, 0.148593]**. These intervals do not account for threshold selection. This does not replace the fixed0.5 main comparison.

| Head | Baseline0.35 pooled | Δ versus original pooled [95%CI] | Baseline0.35 source macro | Δ versus original macro [95%CI] |
|---|---:|---|---:|---|
| vessel | 0.760989 | -0.009135 [-0.014637, -0.003948] | 0.774767 | -0.011890 [-0.024264, -0.000834] |
| artery | 0.715390 | 0.025862 [0.010747, 0.040352] | 0.691281 | 0.010409 [-0.005668, 0.029983] |
| vein | 0.780908 | 0.083610 [0.070606, 0.095957] | 0.763783 | 0.064236 [0.045684, 0.083067] |
| disc | 0.939165 | 0.044357 [0.037107, 0.053446] | 0.924535 | 0.195005 [0.182143, 0.209339] |
| cup | 0.872534 | 0.437135 [0.424815, 0.449543] | 0.870410 | 0.431859 [0.406645, 0.463420] |

Lowering the threshold increases the same-VAL macro by0.004833, but vessel source-macro versus original declines further (delta-0.011890). Cup pooled/source-macro also decline slightly relative to the0.5 baseline. Conditional AV is unchanged; gated accuracy becomes0.696432. All0.35 source/head regressions, precision/recall/count changes and intervals are present in `threshold-035.json`.

## Pairing, uncertainty and limits

Both saved reports must match the actual supplied manifest SHA, canonical schema2 fingerprint,770 unique photo IDs, complete eligible record-ID sets, source attribution, and each head’s per-photo valid-pixel count and GT-positive count. Source/head attribution must be eligible under the manifest task. The enriched baseline’s selected checkpoint/source/data identities are bound to the terminal60 QA. This checks saved evidence consistency; it does not reopen masks or independently certify their original creation.

Bootstrap:2000 paired percentile draws, seed42. Resample reviewed `split_group_id` within the signature of canonical dataset labels for the whole group. All heads, both pipelines, source summaries and AV metrics use the same draw. Known patients and aliases remain grouped, but612 split groups are not proven612 independent patients; opaque identifiers and unknown cross-source relationships remain. Sources themselves are not resampled, so intervals are conditional on these represented sources, these checkpoints and their VAL selection. They do not cover training-seed variability or all model/threshold/postprocessing selection.

The original reference is a qualified composition: vascular/disc counts retain their historical calibrated reference, and cup uses the completed paired fresh cup/disc0 clipping experiment. That study separately logged native backend drift (two cup FP pixels) and clipping losses (80,891 cup TP pixels removed along with1,450,920 FP pixels). Original preprocessing and CPU/backend numerical behavior differ from unified raw RGB inference. The comparison measures declared pipelines rather than isolating architecture. Unknown original exposure can include these validation photographs.

No speed comparison, clinical generalization, TEST selection, new model forward, GT read or final promotion is made. `execution-proof.json` records metadata-only CPU1/nice10 execution with Torch absent. `threshold-050.json` and `threshold-035.json` retain artifact/code hashes, source-level confusion counts and paired uncertainty.

## Reproduce without prediction

```bash
uv run python scripts/compare_completed_val.py \
  --reference reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json \
  --candidate reports/unified/terminal60-threshold-watch/complete-six/threshold-050.json \
  --completion-qa reports/unified/terminal60-threshold-watch/e060-verified.json \
  --manifest /ABS/corpusv1/manifest.json --draws 2000 --seed 42 \
  --output /ABS/new-fixed050-comparison.json
```

Use the0.35 candidate and a fresh output path for the separate exploratory comparison. No raw-image, label-array or checkpoint file is needed.

Focused verification:35 CPU-only tests passed in0.80seconds, covering strict pairing, source/task eligibility, source regression preservation, exclusive AV decoder constraints, scored-domain mismatches, conditional accuracy headers, completed60 versus earlier best checkpoint selection, and stale completion/data/source/coverage rejection. No test model forward or GT access occurs. A prepublication strict-family diagnostic and its correction are retained separately; final execution uses the real completed paired-report family and produces the accepted JSONs.
