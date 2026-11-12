# Completed VAL candidates: all-head and dense snapshot

Report-only snapshot from research8f187cd. The [previous snapshot](../all-head-candidate-comparison-20261111/README.md)
and its **nine candidate objects and21 byte citations are preserved exactly**.
Eight completed BEST/LAST/protocol cases are added (17rows total), in family/role
order, not a ranking or frozen selection. Display values are rounded;
[comparison.json](comparison.json) keeps full-precision copied head counts,
metrics, checkpoint/protocol/config/coverage evidence and50 byte citations.
No new inference, bootstrap, data/model/source/registry/default or TEST action.

| Candidate / actual protocol | Vessel | Artery | Vein | Disc | Cup | Conditional AV | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
|Original reviewed composed reference|0.770123|0.689522|0.697286|0.894808|0.435399|0.748566|0.667030|
|First TPE BEST58 none / .5|0.762882|0.730308|0.792117|0.932671|0.880783|0.824912|0.810895|
|First TPE BEST58 rot4 / .5|0.766953|0.743989|0.803240|0.937129|0.885207|0.836113|0.818055|
|First TPE BEST58 rot4 / .35|0.769464|0.759659|0.814793|0.936902|0.884339|0.836113|0.822099|
|Native V3 trial4 BEST54 rot4 / .5|0.767487|0.754224|0.795406|0.940040|0.880925|0.832472|0.819412|
|Native V3 trial4 BEST54 rot4 / .4|0.769410|0.764300|0.803133|0.939617|0.878191|0.832472|0.822149|
|Native TPE6 BEST52 none / .5|0.761709|0.722659|0.779846|0.937703|0.878005|0.806979|0.808853|
|Curated raw256 BEST52 none / .5|0.759437|0.707847|0.775879|0.933223|0.866570|0.801502|0.801232|
|Curated LC384 BEST60 none / .5|0.745545|0.736124|0.786222|0.939089|0.859474|0.818850|0.806022|
|Curated raw384 BEST59 none / .5|0.759770|0.735385|0.796599|0.940531|0.880469|0.830904|0.814371|
|Curated raw384 LAST60 none / .5|0.759789|0.743976|0.795437|0.937582|0.872971|0.831970|0.813496|
|Curated CLAHE384 BEST58 none / .5|0.751130|0.721123|0.770226|0.924293|0.855065|0.810557|0.801035|
|Curated CLAHE384 LAST60 none / .5|0.750185|0.717596|0.770123|0.924778|0.847977|0.810917|0.798671|
|Curated raw512 BEST50 none / .5|0.756126|0.734414|0.788575|0.938678|0.876028|0.843313|0.811331|
|Curated raw512 LAST60 none / .5|0.753953|0.730548|0.789539|0.938171|0.877379|0.844559|0.809779|
|Curated raw512 BEST50 rot4 / .35|0.762399|0.762217|0.811759|0.942348|0.884582|0.851194|0.822834|
|Native TPE7 LAST60 none / .5|0.764369|0.744789|0.796321|0.934706|0.873189|0.833025|0.815310|

All unified rows have completed60 training runs and actual **own selected**
2529-photo/5700-annotation coverage. BEST50/58/59 weights are those stated epochs,
not relabelled epoch60 because the enclosing run completed60. Original reference
has no comparable established own-TRAIN coverage. TPE7 **BEST46 owns5699/5700**
(missing riga_BinRushed3_image8_observer3_disc_cup), is ineligible/not selected
and has no performance row here; its genuine **LAST60** is used. Ongoing loss2
has no completed-result row. Macro is separate from pooled head Dice.

## Separate dense vessel domain

Dense means87 vessel-supported VAL photos/55known splitgroups/8sources, excluding
only known positive-only AVRDB20photos/20groups. IncludeAll vessel domain has
107photos/75groups/9sources. AVRDB FP0/precision1 is structural, not dense
specificity. This supplement does not rewrite the benchmark/objective.

| Exact candidate context | Dense87/55 Dice | Existing dense minus original95%CI |
|---|---:|---|
|Original reviewed composed reference|0.761327|—|
|First TPE BEST58 none / .5|—|—|
|First TPE BEST58 rot4 / .5|—|—|
|First TPE BEST58 rot4 / .35|0.748236|[-0.018558, -0.008791]|
|Native V3 trial4 BEST54 rot4 / .5|—|—|
|Native V3 trial4 BEST54 rot4 / .4|—|—|
|Native TPE6 BEST52 none / .5|—|—|
|Curated raw256 BEST52 none / .5|—|—|
|Curated LC384 BEST60 none / .5|—|—|
|Curated raw384 BEST59 none / .5|0.741098|[-0.028356, -0.013598]|
|Curated raw384 LAST60 none / .5|0.738405|—|
|Curated CLAHE384 BEST58 none / .5|0.734129|[-0.038555, -0.018085]|
|Curated CLAHE384 LAST60 none / .5|0.731676|[-0.041714, -0.019552]|
|Curated raw512 BEST50 none / .5|0.747223|[-0.019403, -0.009956]|
|Curated raw512 LAST60 none / .5|0.747051|[-0.020402, -0.009082]|
|Curated raw512 BEST50 rot4 / .35|0.751316|[-0.014354, -0.006564]|
|Native TPE7 LAST60 none / .5|0.747853|[-0.018461, -0.009073]|

A dash means **no exact existing aggregation/interval identified**, not zero,
equivalence or permission to borrow a nearby role/backend/TTA/threshold result.
The native first-TPE BEST58none/.5 row does not borrow the slightly different
Old none counters used by later curated comparisons. Raw384LAST60 has an existing
dense point but no exact original-CI identified here. Every available non-original
dense-versus-original interval above is negative. The highest shown unified dense
point (raw512BEST50rot4/.35,.751316) remains below original.761327. Neither
favorable macro nor near-original global vessel establishes all-task improvement
or clinical/equivalence/noninferiority.

## Context and adverse outcomes

TPE5/TPE6/TPE7 native reports use matched source72/evaluationb269,FP32/NONE512/64,
intra8/inter1; TPE7's [matched LAST-only report](../TPE7-LAST60-matched-val-20261112/README.md)
shows dense recovery versus own LAST controls but negative dense/global vessel
intervals versus original. First-TPE and V3 threshold rows use their exact
Old318/198 rot4 contexts. Older raw256/LC384 use1475; new raw384/CLAHE/raw512 use
f13 and fd44 training, with saved full-image CLAHE profile for CLAHE only.
Raw512rot4/.35 changes TTA **and** threshold relative to none/.5. These mixed
contexts/configurations do not isolate patch size, photometry or architecture.

Original182 remains the reviewed composed four-network reference, with historical
exposure/preprocessing/backend qualifications. Exact existing global vessel
intervals and source/derived-SHA bindings are preserved/copied in JSON. No parent/
none interval is relabelled versus original. All source/head/AV adverse outcomes
remain in the linked full reports; this table is not a ranking or simultaneous
all-source success claim. Intervals are exploratory2000/seed42 source-aware
known-group contrasts, not training-seed uncertainty or proven independent
patients; adaptive VAL reuse/multiplicity remain limitations.

Detailed completed evidence: [raw384](../curated-raw384-typed-fresh-VAL-20261111/README.md),
[CLAHE](../clahe384-typed-fresh-val-20261112/README.md),
[raw512 none](../raw512-typed-fresh-val-20261112/README.md),
[raw512 rot4/.35](../raw512-BEST50-rot4-035-fresh-val-20261112/README.md),
[TPE7LAST60](../TPE7-LAST60-matched-val-20261112/README.md).
The previous snapshot and its measured CPU example are retained by reference;
no speed/parameter measurement is extrapolated to new candidates. Historical
benchmark script3a0324 is resolved at commit64e834db, not replaced by the changed
current benchmark script. No old row/citation/source file was edited.

Run `python reports/all-head-candidate-comparison-20261112/checks.py` for stdlib-only
byte/metric/role/coverage/support validation. These projections cannot replace
original typed completion, fresh evaluation or selection gates. Documentation
outside this report is intentionally unchanged to avoid concurrent overview edits.
