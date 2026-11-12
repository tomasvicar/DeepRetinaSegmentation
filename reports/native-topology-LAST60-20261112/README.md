# Native topology from scratch: fixed LAST60 comparison

The native topology did not improve the full five-task result. Its small dense-vessel gain against the matched first-TPE5 control is uncertain, while artery, vein, disc, cup and the macro score are worse. Do not promote this checkpoint or change the shipped model.

Both models completed 60 epochs and have their own full 2,529-photo / 5,700-annotation TRAIN coverage. Primary inference is matched common fe33/ 895, FP32/no autocast, NONE/0.5/tile512/overlap64 on the same Old GTX1060, with exactly the same 770 VAL photos / 993 records and support. Other training settings are equal; architecture and requested width differ. Native widths 32/64/96/128, pooling, activation and receptive field differ together, so this is not an isolated single-layer intervention.

| Measure | First TPE5 LAST60 | Native LAST60 | Native − control, paired 95% interval |
|---|---:|---:|---|
| Five-head source macro Dice | 0.808838 | 0.792277 | -0.016561 [-0.022604, -0.011053] |
| vessel pooled Dice | 0.760804 | 0.762912 | +0.002107 [-0.001060, +0.005582] |
| artery pooled Dice | 0.732667 | 0.720543 | -0.012124 [-0.019510, -0.004385] |
| vein pooled Dice | 0.790737 | 0.774518 | -0.016219 [-0.025154, -0.006585] |
| disc pooled Dice | 0.935338 | 0.917212 | -0.018126 [-0.026342, -0.011714] |
| cup pooled Dice | 0.874873 | 0.841007 | -0.033866 [-0.051646, -0.020460] |
| Dense vessel Dice | 0.738513 | 0.740915 | +0.002402 [-0.001044, +0.006186] |
| Dense vessel precision | 0.750515 | 0.752532 | +0.002017 [-0.002827, +0.006960] |
| Dense vessel recall | 0.726888 | 0.729651 | +0.002762 [-0.007155, +0.013122] |

Dense support contains 87 photos / 55 known split groups / 8 sources; only the 20 positive-only AVRDB photos are excluded from this dense endpoint. AVRDB, every head/source and conditional/end-to-end A/V remain in [the full comparison](comparison.json), including all negative point estimates. All-task intervals use 612 known VAL groups. Source-stratified paired bootstrap: 2000 draws, seed 42; intervals are exploratory and unadjusted for repeated comparisons. Known groups are not proved independent patients, and one training seed does not measure training-seed uncertainty. A crossing-zero interval does not prove equivalence.

The generic saved dense helper retains wording about differing roles/protocols in its `scope` string. For the PRIMARY contrast the actual report wrappers prove matching fixed-LAST60 roles, source and inference protocol; that generic string is not a description of a role mismatch in this pair. The original JSON is preserved without editing.

Secondary references use the previously selected raw512 BEST50/rot4/0.35 protocol and the original networks. These comparisons change training, role and inference/source context and are not isolated architecture tests:

- Against selected_raw512_BEST50_rot4_035: dense Dice 0.751316 → 0.740915, delta -0.010401, interval [-0.015735, -0.005171].
- Against original6808: dense Dice 0.761327 → 0.740915, delta -0.020412, interval [-0.028216, -0.013573].

Native has 1,304,421 parameters versus 1,965,637 for the compact control. Actual full-VAL worker time was 369.406 s versus 222.770 s for the matched control; fewer parameters did not make this implementation faster here. These are whole-validation costs, not a population latency benchmark. Native charged peak memory was 1015.1 MiB with swap 0 under 3 GiB; the inference observer and service returned 0. [Postcheck](validation-postcheck.json) verifies original support/source and unchanged inputs.

The [native original report](native-LAST60-validation.json) and [matched control original](TPE5-LAST60-validation.json) are byte-exact copies. Secondary originals are already published as [selected raw512](../shared3stream-tile-blend-VAL770-20261112/prior0a5c-validation.json) and [original networks](../shared3stream-tile-blend-VAL770-20261112/original6808-validation.json). [The count-only comparison](comparison.json) retains input/helper hashes, source/configuration contexts, projection hashes, all source intervals and adverse estimates. It completed under an actual 1 GiB / no swap / CPU 100% / core 4 scope, with [observed exit 0](count-analysis-observed-exit.json) and [resource receipt](count-analysis-resources.json). [Root recount](root-count-review.json) independently verifies all five native pooled counts and the dense 87-photo counts without another NN/GT read.

Exact execution scripts and complete original resource/completion/CPU/NN receipts are in the private `metacentrum-workflows` repository under `reports/legacy-LAST60-matched-counts-20261112` and `reports/legacy-scratch-LAST60-common-fe33-pending-20261112`. No new TEST, selection, Optuna update or product/default change occurred. The separately prepared BEST58 scalar calibration has no result in this report.
