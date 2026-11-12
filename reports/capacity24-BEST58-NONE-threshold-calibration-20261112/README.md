# Capacity24 BEST58 NONE: eight-scalar VAL calibration

The single approved run completed with actual science return code 0, stage status completed, and external observed return code 0 at **13:17:01.923601 UTC on 12 November 2026**. Source and original inputs remained unchanged; no timeout, retry, cleanup escalation or additional inference occurred. Science took 367.724 seconds and the supervising stage 369.808 seconds. These are operational durations, not latency benchmarks.

The checkpoint is genuine standalone72 capacity24 BEST58, SHA `7811d2e4784f6c0d5d15e101c44a9958e3b7a32419d2a8ef463f114c8cae7d68`, with original full60 completion and own 2,529-photo/5,700-annotation eligibility. The original training identity 3725 remains distinct from the immutable 6d/f13 evaluator. Runtime confirmed FP32, no autocast, TTA none, tile/overlap 512/64, physical Old GPU b606, CPU 2/inter 1 and affinity 2,3.

The unchanged e9b science adapter and 7be counter made **one probability call and one network view per photo**, over all 770 VAL photos. Eight fixed scalar counters were streamed without a full probability cache. The same-probability native 0.5 parity check passed on all 770 photos. At 0.5, all six fields exactly match the original standard report `98297a0c380be3a2af9df9032cad5a99717349ac9ca109d84c1ea0ee72047155`: channels, by_dataset, per_photo, macro, conditional AV and AV accuracy. No extra model call was used for either check.

## Full curve

V/A/vein/disc/cup are pooled Dice; macro is the existing source/head macro. Conditional AV is invariant at 0.8370221956193618. Dense vessel is the supplementary 87-photo/55-group domain, excluding only positive-only AVRDB vessel support.

| Scalar | Macro | V | A | Vein | Disc | Cup | Dense V | Dense precision | Dense recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.82371158 | 0.76408878 | 0.77102097 | 0.82196287 | 0.93645704 | 0.87664099 | 0.74333609 | 0.68778060 | 0.80865524 |
| 0.30 | 0.82323932 | 0.76648478 | 0.76618586 | 0.81835157 | 0.93658890 | 0.87851322 | 0.74582362 | 0.70242283 | 0.79494085 |
| 0.35 | 0.82239000 | 0.76785452 | 0.76144864 | 0.81486653 | 0.93644601 | 0.87980720 | 0.74725463 | 0.71540704 | 0.78206983 |
| 0.40 | 0.82127676 | 0.76844820 | 0.75661535 | 0.81140168 | 0.93616287 | 0.88044697 | 0.74789302 | 0.72735380 | 0.76962593 |
| 0.45 | 0.81993209 | 0.76834973 | 0.75171118 | 0.80780452 | 0.93572551 | 0.88085373 | 0.74780936 | 0.73864506 | 0.75720392 |
| 0.50 | 0.81828217 | 0.76747467 | 0.74657679 | 0.80392778 | 0.93519266 | 0.88098204 | 0.74693716 | 0.74946629 | 0.74442504 |
| 0.55 | 0.81632981 | 0.76600795 | 0.74102726 | 0.79990216 | 0.93452122 | 0.88081413 | 0.74545154 | 0.76021477 | 0.73125079 |
| 0.60 | 0.81402834 | 0.76379825 | 0.73499187 | 0.79550184 | 0.93369973 | 0.88048094 | 0.74319550 | 0.77101053 | 0.71731749 |

Original6808 dense Dice is 0.761326809, precision 0.730948239 and recall 0.794339969. Every tested scalar remains below its dense Dice, with all eight paired 95% intervals wholly negative. At 0.40 the dense difference is −0.013433788 [−0.018200453, −0.009489972]. Raw512 BEST50 rot4/.35 dense Dice 0.751315957 also exceeds every point on this curve. These are explicit protocol contrasts, not an isolated capacity or speed effect.

At 0.25 recall exceeds the original while precision drops substantially; at 0.30 recall is close to the original but precision remains lower. At 0.40 both dense precision and recall are below the original. Raising the scalar improves precision at a recall cost. Neither the observed macro maximum at 0.25 nor the dense maximum at 0.40 is automatically selected.

All five heads, conditional AV, all source results and intervals are retained in `paired/` and `results.json`. For example, 0.40 vessel versus original is adverse on DRHAGIS (−0.031713422, paired interval [−0.038161781,−0.025730897]); DRHAGIS artery is −0.026545901. ARIA, CHASE, HRF and other adverse sources remain in full, as do AVRDB positive-only results. No sign-based filtering is applied.

## Evidence and limitations

The private original collection contains the eight exact threshold reports, summary/native parity, original eligibility proof, prior 0.5 comparison, runtime, admission/launch, True profile, stage and external exits. `results.json` records every original byte SHA. The local watcher sampled no more frequently than 120 seconds and stopped naturally after external completion and observer absence. It issued no control actions or GPU queries. No model or GT was read for the subsequent count analysis.

The point-table reader initially expected a wrapper around the bare original6808 metric report. That metadata-only KeyError and its wrapper-or-bare correction are preserved in the private original metadata-failure receipt. Original report hashes were not relaxed and no inference was repeated. Count analysis then exited 0, using the existing source-stratified paired 2000-draw/seed42 method on all 612 known groups and the dense 87-photo/55-group supplement. Known groups are not proven independent patients. Intervals are exploratory and uncorrected for this VAL grid; no seed variance or equivalence is inferred.

Calibration reports are explicitly **not ordinary freezeable VAL evidence**. A newly chosen scalar needs standard fresh VAL; the already-existing exact standard 0.5 report may be reused for its unchanged protocol. No scalar, model, default, final selection, freeze or TEST was selected or changed by this task.

## Public projection

This compact aggregate export preserves all exact integer head/source counts, full-precision metrics, eight paired all-head/source/AV outputs and dense intervals. Original per-photo rich reports remain private and are bound by SHA; no photo identifiers, operator paths, physical UUIDs or process IDs are published. Paired files are byte-identical copies of the recorded count analysis. Their temporary count-view digests are not original report digests. This public projection is not selection-gate evidence. Full private actual evidence is commit `3fccc3d`; no extra inference or statistics were run for export.
