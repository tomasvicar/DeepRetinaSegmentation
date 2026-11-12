# Raw512 actual fresh VAL: mixed result, no promotion

Both genuine curated_fd44 selected-role CPU checks and both fresh VAL runs
completed successfully. **BEST50 has lower full macro Dice than raw384 BEST59,
while the supplemental dense vessel domain improves.** Full-source vessel and
AVRDB regressions remain explicit. These results do not establish overall
quality superiority, equivalence or a default model choice.

Each actual report covers all770 VAL photos. Both case exits and the external
observer returned0, with no timeout/retry and unchanged source/input bytes;
actual outer completion was2026-11-12T00:51:37.556145UTC. The two-case stage took
447.7713s; this is not a controlled prediction-speed benchmark.

| Genuine selected role | Macro | Vessel Dice | Artery | Vein | Disc | Cup | Conditional A/V accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw512 BEST50 |.81133053|.75612576|.73441353|.78857498|.93867765|.87602759|.84331283|
| raw512 LAST60 |.80977932|.75395337|.73054792|.78953870|.93817141|.87737908|.84455870|

The aggregate JSON reports preserve full-precision metrics and actual pooled
and per-source TP/FP/FN/support counters. BEST checkpoint SHA521b8881… and LAST
6846423c… retain their original roles, full saved raw configuration, descriptors,
source family and own2529-photo/5700-annotation contribution. Original fd44
training source and actual6d3593f/f13 evaluation are recorded separately.
Both evaluations use FP32/no autocast, NONE, scalar.5, full geometry and tiles
512/overlap64. No configuration stripping, family conversion or TEST evaluation.

## Matched-role comparisons

Existing count analysis uses source-stratified paired2000 draws/seed42 across
770 photos/612 known split groups. All five heads, conditional AV, per-source
intervals and adverse rows are retained. Groups are not proven independent
patients; intervals are exploratory and uncorrected for multiplicity.

| raw512 minus matched raw384 | BEST50 minus BEST59 delta [95%CI] | LAST60 minus LAST60 delta [95%CI] |
|---|---|---|
| Macro |−.003041 [−.006524,−.000158]|−.003716 [−.007047,+.000479]|
| Vessel |−.003644 [−.007432,+.000668]|−.005836 [−.009969,−.001350]|
| Artery |−.000971 [−.006357,+.004809]|−.013429 [−.019976,−.007263]|
| Vein |−.008024 [−.012387,−.003678]|−.005898 [−.011617,−.000622]|
| Disc |−.001853 [−.003238,−.000424]|+.000589 [−.001145,+.002516]|
| Cup |−.004442 [−.008875,−.000574]|+.004408 [−.001275,+.009918]|
| Conditional AV |+.012408 [+.008262,+.016361]|+.012589 [+.007557,+.017916]|

Against firstTPE BEST58 NONE/.5, raw512 BEST macro difference+.000433 has
CI[−.003301,+.007562]; pooled vessel difference−.006753 has
CI[−.011285,−.002482]. Terminal-role comparisons are also retained. Reference
firstTPE BEST uses prior Old1f evaluation and LAST uses Ada72; their source/runtime
contexts differ from f13 and are preserved. These comparisons describe the
candidate pipelines rather than an isolated architecture or patch-size effect.

Against original182, BEST pooled vessel difference is−.013998,
CI[−.018711,−.009761], despite macro/AV/disc/cup gains. Historical original
exposure/preprocessing limitations remain. Both roles and every head are present
in the six paired JSON files; macro improvements do not erase vessel deficits.

## Dense vessel support and adverse AVRDB result

The pre-existing supplemental dense vessel domain includes87 photos/55 known
groups/8 sources, excluding only known positive-only AVRDB vessel support. It
does not replace the full objective, change supervision or prove clinical
noninferiority. BEST dense vessel Dice is.74722331 versus raw384 BEST.74109829,
firstTPE BEST.74135018 and original182.76132661.

| Dense vessel contrast | BEST delta [95%CI] | LAST delta [95%CI] |
|---|---|---|
| minus matched raw384 |+.006125 [+.002624,+.010051]|+.008647 [+.005468,+.012607]|
| minus matched firstTPE |+.005873 [+.003029,+.009184]|+.008540 [+.005245,+.012506]|
| minus original182 |−.014103 [−.019403,−.009956]|−.014275 [−.020402,−.009082]|

The contrasting full-domain result is retained: BEST versus raw384 BEST has
AVRDB vessel−.094050,CI[−.118846,−.074303], artery−.040335 and vein−.043450
with negative intervals. Against firstTPE BEST its AVRDB vessel difference is
−.120689,CI[−.162185,−.093405]. The partial-positive annotation limitation
qualifies interpretation; adverse counts are not silently removed. All negative
source/head estimates are in `adverse-source-results.json`.

## Provenance and limits

Saved-training versus fresh support matches exactly for record/source identity,
GT valid/positive counts and eligible AV support. Numeric bitparity does not:
BEST macro delta+4.3084741e−7 and LAST−8.5144726e−8. Per-head differing-photo
counts are preserved in `saved-fresh-drift.json`; no cause for the small runtime
numeric differences is asserted.

This compact public projection omits operational details and individual photo
identifiers/rows, retaining all aggregate/source counters. Full770 original typed
reports and observed terminal bytes remain immutable private evidence, pinned by
original SHA in `export-provenance.json` and `completion.json`. Public projections
are explicitly not original eligibility/freeze inputs. Existing paired/dense/
adverse/drift JSON statistics are copied byte-for-byte; no statistics or inference
were rerun for publication. No raw images, weights, private commands, GPU UUID,
model/source changes, default promotion, final selection, TEST or new experiment
is included. The separate prospective rot4/.35 packet is not evidence of a result.
