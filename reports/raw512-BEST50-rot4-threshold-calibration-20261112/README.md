# Raw512 BEST50: exploratory rot4 threshold calibration

One completed VAL-only run evaluated the genuine curated_fd44 raw512 BEST50 checkpoint (SHA 521b8881…132cea), using unchanged source 6d/f13, FP32 rot4 and tile 512/overlap 64. All 770 photos and 612 known groups were included. The checkpoint owns 2,529 TRAIN photos and 5,700 annotations. Eight predeclared scalar thresholds share one probability prediction per photo (four neural views); no probability cache or extra NN calls were used. Same-probability native 0.5 counts matched on every photo. The 0.35 result exactly reproduces the prior same-case standard report across all six checked count/metric fields.

Actual child and external observer both returned observed 0, with no timeout/retry and unchanged source/inputs. Completion was 10:41:32 UTC. Science took 701.617 s and the supervisor 703.617 s; these are operation durations, not latency benchmarks. Full original byte pins, counts, protocols and exact scores are in [results.json](results.json). This public file is a sanitized aggregate projection, **not ordinary/freezeable VAL evidence**. It contains no private host paths, UUIDs, photo/record IDs or operations commands.

Paired intervals reuse the existing saved-count comparison and dense-support methods: 2,000 draws, seed 42, source-stratified known groups. Main analysis uses 770 photos and 612 groups; the vessel supplement uses 87 supported photos and 55 groups, excluding only positive-only AVRDB. These are exploratory uncorrected intervals, with no independent-patient, adaptive-minimum, seed-variance or equivalence claim. Original legacy exposure/preprocessing/training differences remain. The two explicit scalar contrasts use identical weights/source/rot4 probabilities, changing only threshold. Conditional A/V stays fixed; end-to-end A/V changes with vessel gating. All source/head adverse intervals are retained.

## All eight observed points

| Threshold | Macro | Vessel | Artery | Vein | Disc | Cup | Dense vessel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 6808 | 0.66703197 | 0.77012368 | 0.68953032 | 0.69729783 | 0.89480784 | 0.43539895 | 0.76132681 |
| 0.25 | 0.82525673 | 0.76040449 | 0.77518912 | 0.82204630 | 0.94224781 | 0.88091909 | 0.74751169 |
| 0.30 | 0.82433893 | 0.76196840 | 0.76860275 | 0.81684128 | 0.94244041 | 0.88347929 | 0.75001879 |
| 0.35 | 0.82283410 | 0.76239893 | 0.76221670 | 0.81175930 | 0.94234820 | 0.88458249 | 0.75131596 |
| 0.40 | 0.82094736 | 0.76194106 | 0.75580328 | 0.80668375 | 0.94205073 | 0.88514180 | 0.75165706 |
| 0.45 | 0.81881666 | 0.76074427 | 0.74933560 | 0.80152757 | 0.94164415 | 0.88541492 | 0.75121149 |
| 0.50 | 0.81632116 | 0.75875965 | 0.74253252 | 0.79614192 | 0.94113913 | 0.88537121 | 0.74995583 |
| 0.55 | 0.81334914 | 0.75599597 | 0.73531002 | 0.79038341 | 0.94038011 | 0.88488764 | 0.74789706 |
| 0.60 | 0.80987621 | 0.75237458 | 0.72761162 | 0.78407611 | 0.93945567 | 0.88409419 | 0.74499562 |

Conditional A/V accuracy is 0.851194311 at every threshold. All dense vessel intervals versus original 6808 are negative. At 0.35 the dense difference is −0.010010852, 95% CI [−0.014354340, −0.006563583]. At 0.40 it is −0.009669744, CI [−0.014350217, −0.005929346]. A larger macro therefore does not close the vessel gap.

## Requested paired tradeoffs

| Metric | 0.25 minus 0.35: difference [95% CI] | 0.40 minus 0.35: difference [95% CI] |
|---|---|---|
| Macro | +0.00242264 [+0.00162919, +0.00338543] | -0.00188674 [-0.00228717, -0.00155141] |
| Vessel | -0.00199443 [-0.00351085, -0.00039243] | -0.00045787 [-0.00117387, +0.00020232] |
| Artery | +0.01297242 [+0.01218926, +0.01381698] | -0.00641342 [-0.00691808, -0.00595402] |
| Vein | +0.01028700 [+0.00935187, +0.01129093] | -0.00507555 [-0.00549097, -0.00467684] |
| Disc | -0.00010039 [-0.00084679, +0.00056781] | -0.00029747 [-0.00055858, -0.00000241] |
| Cup | -0.00366340 [-0.00530791, -0.00211037] | +0.00055931 [-0.00002617, +0.00118879] |
| Dense vessel | -0.00380426 [-0.00539974, -0.00210302] | +0.00034111 [-0.00040107, +0.00104607] |

Lowering the scalar to 0.25 increases macro and artery/vein Dice, while reducing global vessel, dense vessel and cup Dice. On RIMONE, disc falls by 0.0083073 and cup by 0.0079171; both paired intervals are negative. Raising it to 0.40 gives only a small dense point increase whose interval crosses zero, while macro and artery/vein Dice fall. Conditional A/V remains exactly unchanged; end-to-end A/V changes with vessel gating. All source intervals, including adverse AVRDB outcomes, are retained in the [paired results](paired/) and [scalar contrasts](paired/).

No threshold is selected or promoted here. The already measured standard 0.35 case remains unchanged pending the width24 comparison; no new fresh NN run was chosen. Calibration outputs are explicitly exploratory, not freezeable standard VAL. Any final chosen protocol still requires fresh standard VAL. No TEST, final freeze, default change, equivalence or clinical claim is made.

## Evidence and limitations

The actual immutable raw reports and observed-exit proof remain in the private operational evidence commit4f9dcb4df860b1584f919d61cd0a3fd9d3e873c1. [Artifact hashes](artifact-sha256.json) pin this report; each threshold entry also retains its original whole-report SHA. The [comparison implementation](../../src/deepretina/comparison.py) defines the paired method. Dense support and original exposure qualifications remain as in the existing [dense-vessel supplement](../dense-vessel-support-20261112/README.md). No new tests, models, NN/GT/TEST execution, freeze or source changes were made for this publication.
