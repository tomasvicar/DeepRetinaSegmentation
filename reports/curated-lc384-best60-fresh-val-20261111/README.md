# LC BEST60 fresh VAL: vessel/cup regressions; planned LAST timed out

**The planned two-role validation did not complete:** BEST60 completed successfully on all770 VAL photographs with actual child exit0. LAST60 exhausted the original remaining budget and produced **no completed fresh report or metrics**. The overall observer recorded exit1; the supervisor reported timeout, source/input bytes unchanged and no retry. There was no outer-watchdog timeout or cleanup escalation/error. Do not describe this as a successful two-role validation.

Training is separate: the original curated model completed60 epochs and both original roles passed their own CPU completion checks (2529 TRAINphotos/5700annotations). BEST is also E60, checkpoint `dfef53e453c51e87980c5132b8d829f7720b0f32b172982f4e1e1594b7449386`. The inference budget outcome does not undo that training completion. [Original TRAIN completion evidence](../curated-fd44-completion/genuine-lc384-two-role-audit.json) remains qualified independently of this fresh result.

## Actual paired VAL evidence

|Metric|Parent TPE BEST58|LC BEST60|ΔLC-parent [95%CI]|Raw256 BEST52|ΔLC-raw256 [95%CI]|
|---|---:|---:|---|---:|---|
|macro|0.81089742|0.80602164|-0.00487578 [-0.01102689,+0.00813885]|0.80123186|+0.00478978 [-0.00237478,+0.01425226]|
|vessel|0.76287902|0.74554493|-0.01733409 [-0.02276002,-0.01251788]|0.75943735|-0.01389242 [-0.01911906,-0.00910100]|
|artery|0.73032355|0.73612450|+0.00580095 [-0.00225836,+0.01467466]|0.70784684|+0.02827766 [+0.01672685,+0.03942401]|
|vein|0.79211047|0.78622209|-0.00588838 [-0.01433952,+0.00439825]|0.77587916|+0.01034293 [+0.00154896,+0.01814319]|
|disc|0.93267149|0.93908870|+0.00641721 [+0.00061265,+0.01258408]|0.93322281|+0.00586589 [+0.00012777,+0.01345820]|
|cup|0.88078398|0.85947379|-0.02131018 [-0.02669815,-0.01612826]|0.86657012|-0.00709633 [-0.02209548,+0.01299730]|
|conditional AV|0.82491954|0.81885013|-0.00606941 [-0.01457741,+0.00359882]|0.80150205|+0.01734809 [+0.00738353,+0.02613801]|
|end-to-end AV|0.69963111|0.69897010|-0.00066101 [-0.00952513,+0.00922961]|0.68370329|+0.01526682 [+0.00478722,+0.02468834]|

Vessel Dice regresses against **both** references with intervals entirely belowzero. Cup also regresses against the parent with a negative interval; pooled disc improves. Macro intervals crosszero for both comparisons. These results do not support LC promotion or an all-task improvement. Against raw256, artery/vein/conditionalAV improve, but vessel worsens; those gains cannot be substituted for the vessel result. Conditional AV class discrimination and vessel-gated end-to-end accuracy are reported separately.

All770 photos,993 annotation records, per-head source attribution, valid/GTpositive domains and AV scored/known domains match exactly. Pooled and per-source integer sums were independently checked against the original headers. Existing frozen comparison math uses2000paired draws/seed42, stratified by source signatures of612 reviewed split groups. Known aliases/patients are preserved, but unknown-patient independence is not proven. Intervals are exploratory conditional on VAL-selected checkpoints, not training-seed uncertainty, multiplicity-corrected source discoveries or independent clinical generalization.

## Adverse sources and full counts

Every source/head is retained in the comparison JSON, with TP/FP/FN/valid counts, precision/recall, supported photo/group counts and intervals. `*-adverse-sources.json` lists ALL negative source/head and conditional/end-to-end AV entries. Against parent there are22 negative entries,13 with unadjusted interval upperbound belowzero; against raw256 there are16/7. Examples of parent regressions: STAREvessel−.05458 (4photos/4groups), DRHAGISvessel−.04846 (8/8), REFUGEcup−.02953 (400/400), AVRDBvessel−.02680 (20/20), all with intervals belowzero. ARIAdisc−.04390 (22/12) has an interval crossingzero.

The comparison references are the exact existing [parent TPE BEST58 NONE](../curated-raw256-fresh-val-20261111/parent-TPE72-BEST58-fresh-NONE.json) and [raw256 BEST52 fresh](../curated-raw256-fresh-val-20261111/raw256-BEST52-fresh.json) scientific exports, whose original byte SHAs are recorded in each result. The parent is actual GTX1060 NONE report195e, not slightly different prior Ada/card values. The LC original whole report SHA is `a0a347f465962de58836d39101e6ad899a2a4644f8164f112fc0d697b59e6446`.

## Protocol, source and partial-run qualifications

This is corpus-v2/21-source TRAIN and unchanged15-source VAL, fixed RAW25 geometry, FP32, NONE,512/64 tiling and threshold.5. The input PHOTOMETRY differs deliberately: LC uses complete `legacy_lc_clahe_v1`, full prepared image before tiling and original image-derived/official FOV, never GT-validity FOV. All saved descriptor/backend/config fields are retained. The historical training photometry-cache identity was preserved; actual validation ran on the fly without substituting a runtime cache.

Parent and LC training configs differ only photometric_profile; raw256 also differs patch256→384. Both protocol comparisons differ only the explicitly recorded photometry field. Trained weights/selected epochs, source/evaluation builds and training hardware differ; this is not a seed-averaged, source-independent causal comparison. No protocol equality, old-family retagging or freeze admission is asserted.

LC used FP32/no autocast on a GTX1060 with CPU2/inter1. Evaluation elapsed1030.283s, plus canonical/source pre/post checks. Its worker actual nice19 versus prior reference15 is documented; no matched latency ratio or cache speedup is claimed. The whole planned operation stopped after1650.4439s within its original30-minute outer bound. LAST had no remaining completed validation: no LAST score, count view or quality claim is exported. Original stage/outer receipt SHAs and the partial outcome are preserved in `final-operation-context.json`.

The saved training-VAL BEST macro.8060176818157359 and fresh.8060216403688077 differ slightly. No bitwise count parity or quality gain from that numerical drift is claimed; fresh measured counts govern this comparison.

## Export and reproduction scope

`lc384-BEST60-fresh.json` retains the full scientific header/config/descriptor and unchanged770per-photo confusion/AV counts. Operational paths, process IDs, device UUIDs and environment fields were removed; stable content-hash photo IDs/public dataset annotation aliases remain for pairing, as in earlier exports. Original whole-byte SHA provenance explicitly marks sanitized exports as **not original gate-input bytes**. Original reports/receipts and all operator proofs remain private. No weights, images or GT arrays are included.

Pure metadata reproduction: project the unchanged `metrics` from these scientific reports, adding the original wrapper data.manifest_sha256 and checkpoint.sha256, then call existing `deepretina.comparison.compare_reports(reference,candidate,canonical_manifest,repeats=2000,seed=42,include_sources=True,include_av=True)`. The full header/protocol difference must stay explicitly attached to any derived count view. Actual code/input SHAs and count-integrity evidence are included; the operation originally used frozen fd44 comparison math.

There was no additional NN for the comparison, no retry/LAST rerun, no model/default/threshold change, no final-selection or TTA admission, and no TEST execution. HPO remains ongoing.
