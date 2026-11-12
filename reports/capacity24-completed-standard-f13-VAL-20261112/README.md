# Capacity24 BEST58 and LAST60: standard f13 VAL

Both actual cases completed with observed return code 0. The original external observer recorded return code 0 at **13:00:16.443782 UTC on 12 November 2026**, without timeout or cleanup error. The supervisor took 573.805 seconds, preserved source and input hashes, and verified identical photo, record, source and GT support across both full 770-photo reports. No retry occurred. This elapsed time is an operational duration, not a latency benchmark.

Both checkpoints are genuine standalone source72 width24 full60 roles, each with own coverage of 2,529 TRAIN photos and 5,700 annotations. BEST58 is `7811d2e4784f6c0d5d15e101c44a9958e3b7a32419d2a8ef463f114c8cae7d68`; LAST60 is `43fad085d43e2f5de8fd9eee4e10ba2e7cc380f5ff64aa382b358cd194a36786`. Original generic4b QA, protocol, training terminal, stopped custody and both CPU eligibility chains remain unchanged and pinned in the prepared packet. Original source72 training identity is separate from the current immutable 6d/f13 evaluator.

Every case used FP32 without autocast, TTA none, threshold 0.5, tile/overlap 512/64, physical Old GPU b606, intra2/inter1 and affinity 2,3. Exact operational originals are retained privately; their SHA digests are retained in `export-provenance.json`. Sanitized actual exits and runtime are in `producer-terminal.json`. The local watcher stopped naturally after observing completion and the original observer absent. It issued no controls or GPU queries.

## Saved-count results

Macro is the existing source/head macro; V/A/vein/disc/cup are pooled Dice. Dense vessel is the supplementary 87-photo, 55-group domain excluding only AVRDB; it does not redefine the objective or selection gate.

| Candidate | Macro | V | A | Vein | Disc | Cup | Conditional AV | Dense V | Dense precision | Dense recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original6808 | 0.66703197 | 0.77012368 | 0.68953032 | 0.69729783 | 0.89480784 | 0.43539895 | 0.74857251 | 0.76132681 | 0.73094824 | 0.79433997 |
| raw512_BEST50_NONE050 | 0.81133053 | 0.75612576 | 0.73441353 | 0.78857498 | 0.93867765 | 0.87602759 | 0.84331283 | 0.74722331 | 0.75354838 | 0.74100353 |
| raw512_BEST50_rot4_035 | 0.82283410 | 0.76239893 | 0.76221670 | 0.81175930 | 0.94234820 | 0.88458249 | 0.85119431 | 0.75131596 | 0.72215649 | 0.78292933 |
| capacity24_training_best | 0.81828217 | 0.76747467 | 0.74657679 | 0.80392778 | 0.93519266 | 0.88098204 | 0.83702220 | 0.74693716 | 0.74946629 | 0.74442504 |
| capacity24_fixed_terminal60 | 0.81756740 | 0.76454497 | 0.75254356 | 0.80273176 | 0.93553168 | 0.87675029 | 0.83555320 | 0.74291157 | 0.74768171 | 0.73820191 |

BEST58 dense Dice minus original6808 is −0.014389647, paired 95% interval [−0.019964295, −0.009624942]. Its contrast with raw512 NONE/.5 is −0.000286144 [−0.002314117, +0.001721332], and with raw512 rot4/.35 is −0.004378795 [−0.007683056, −0.001256007]. LAST60 minus original is −0.018415242 [−0.025997139, −0.011582093]. BEST58 pooled vessel versus original is −0.002649011 [−0.008023948, +0.002616987]; the global interval crossing zero does not remove the dense deficit.

All five heads, source results and conditional AV intervals are retained in `paired/`; complete point results are in `all-head-point-results.json`. Adverse BEST58 point deltas versus original include vessel ARIA −0.020781, CHASE −0.017331, DRHAGIS −0.022177, HRF −0.013951 and RITE −0.008720; DRHAGIS artery is −0.032548. AVRDB vessel is +0.102350 and must be read with its positive-only support qualification. Adverses against both raw512 protocols are retained in full rather than filtered by sign.

## Method and limits

The recorded private count-analysis scripts use saved counts and canonical manifest grouping only, without model loading, GT decoding or additional inference. Every comparison checks exact photo/record/source and GT support. The original reference is the fresh full-770 report `6808f293d750dec5a94a0593f5c52d054733696d21984a61e399284355f94405`; historical182 is not substituted. Whole-domain intervals use the unchanged comparison implementation with 2,000 source-stratified paired draws, seed42 and all 612 known groups. Dense intervals use the existing dense helper on 87 photos/55 groups. Intervals are exploratory and uncorrected; known groups are not proven independent patients. Training histories, source versions and inference protocols differ, so these contrasts do not isolate a capacity effect. No equivalence, clinical, speed, selection, promotion, freeze or TEST claim is made.

Temporary metrics projections exist only for the existing count-comparison API and do not retag original reports as selection evidence. The first count-analysis attempt omitted the manifest field because it resides in the wrapper’s data object; its metadata failure and precise correction are recorded in the private analysis receipt. No inference was repeated. The historical staging failure remains separately documented in the prepared packet.

## Public export

The two ordinal-count projections retain exact integer counts, source labels and stable grouping semantics, with all aggregate values unchanged. They are explicitly not the original selection-gate reports. Original report SHA values and the private evidence commit are recorded in `export-provenance.json`. Paired results are byte-identical copies from the recorded count-only analysis; their temporary-projection digests must not be confused with original report digests. The legacy candidate-checkpoint field in paired metric projections is null; exact original checkpoint bindings are retained in the ordinal reports and provenance. No operator paths, physical device IDs or photo IDs are published.
