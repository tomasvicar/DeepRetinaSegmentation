# Current full-data research evidence

Snapshot: **2026-11-11, 15:29 UTC**, with completed reports
reviewed for this update. A last observed scheduler state is not a claim about
current execution. The final optimized model, selection freeze and held-out TEST
comparison are **not finished**. No released default is changed by this overview.

## Product and common data

Use [`for_Leipzig`](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/for_Leipzig)
for installation, prediction and the original verified checkpoints plus explicitly
identified unified baselines/previews. This research branch is separate. Inference
needs no MATLAB. The original stored-input arithmetic is numerically checked;
calibrated raw A/V retains measured photometric differences and unresolved
historical training-file exposure. See the product's `docs/MATLAB_AV_PROFILE.md`.

Canonical V2 has **2,529 TRAIN photographs / 5,700 annotation records from21
sources**, **770 VAL** photographs and856 TEST photographs. Supported annotations
alone supervise each output. Recovered patient/photo aliases remain grouped;
unknown patient identity is not thereby resolved. No new TEST evaluation is
reported here. Conditional A/V accuracy scores reference class-labeled vessel
pixels and does not measure vessel discovery by itself.

One further genuine native TPE proposal now continues the same auxiliary study
after all six completed E27 observations. Width16/BatchNorm/basic stays fixed in
this suggestion; LR becomes0.000411624945, weight decay0.0000823809785 and vascular
sampling1.05918806. It is distinct from all five patch/photometry arms. No second
full training has started; six observations do not establish optimality.
[Native continuation provenance](../reports/tpe72-completed-fresh-val/next-native-TPE-proposal).

A separate TPE BEST58 TTA sweep actually started on Old at15:28UTC, with its
first child verified live, all770 VAL planned per case and a90min cap. Results
are pending. This extends the completed vascular4 TTA experiment below; it does
not reuse its scores or change a released inference default. No TEST is involved.

The completed TPE BEST58 also has its own measured CPU example: median8.4163s
versus59.6351s for the selected original four-network pipeline within the same
run, one DRIVE TRAIN21 raw photo/official FOV and one CPU thread. Parameters:
1,965,637 versus5,943,312. This is default `none` inference, not the ongoing TTA
experiment's latency or a general throughput claim. All samples, actual exit0,
FP32 checkpoint metadata and scope are in
[the TPE benchmark](../reports/benchmark/strong-pipeline-cpu-drive21-tpe72-best58).

## Completed full60 models: keep roles and evidence distinct

| Model and role | VAL evidence | Macro dataset Dice | Pooled vessel Dice | Conditional AV |
| --- | --- | ---: | ---: | ---: |
| V3 control0 BEST59 | Saved native counts, typed pool eligibility | .801281 | .760570 | .801055 |
| V3 control0 LAST60 | Saved native counts, typed pool eligibility | .801188 | .758714 | .808460 |
| V3 trial4 BEST54 | Saved native counts, typed pool eligibility | .815057 | .763145 | .823922 |
| V3 trial4 LAST60 | Saved native counts, typed pool eligibility | .804570 | .750500 | .823318 |
| Auxiliary TPE candidate BEST58 | Independently fresh VAL770 | .810895 | .762882 | .824912 |
| Auxiliary TPE candidate LAST60 | Independently fresh VAL770 | .808833 | .760803 | .821755 |

Both V3 runs genuinely completed60 with their own full5,700-annotation/2,529-photo
coverage in BEST and LAST. Their [typed eligibility](../reports/v3-control0-final60-eligibility)
and [trial4 eligibility](../reports/v3-trial4-final60-eligibility) bind the original
pool receipts, scheduler/trainer exits and source. Control0 LAST auditor exit is
**unknown** despite its checked emitted proof; do not invent an observed0.
Control0 BEST and both trial4 audits have separate observed0 exits. These saved
native count views are **not** new enriched final-runtime evaluations.
[All12 saved-count comparisons](../reports/v3-completed-native-val-comparison)
retain support checks, source/head regressions and paired uncertainty.

The standalone [TPE candidate completed60 and received fresh BEST/LAST VAL770](../reports/tpe72-completed-fresh-val).
Original trainer, generic completion QA, both fresh validation CLIs and their
supervisor have observed0 exits. BEST58 and LAST60 each cover every selected
annotation. The fixed27 post-startup proposal used five imported observations in
six dimensions; this is actual native TPE, not evidence of convergence. Its
[auxiliary Optuna trial now has a genuine COMPLETE27 proxy result](../reports/tpe72-completed-fresh-val/auxiliary-fixed27-result)
(.97513476245,5,686/5,700 annotations at27, cosine horizon60). That bookkeeping
value is neither the later full60 result nor macro-BEST; the main V3 study was
not modified and no additional proposal was asked.

Fresh TPE counts differ slightly from saved counts: LAST changes6 photos/7
head-photo entries, macro−4.29046e−8; BEST changes5 photos/7 entries,
macro−1.00598e−7. All GT supports and attribution remain exact. Actual fresh flags,
cache/backend distinctions and integer deltas are disclosed; no counterfactual
forward was run to assign a cause. Use the fresh counts for those fresh reports.

The frozen original vessel reference is .77012340. Every V3/TPE vessel value above
still falls below it. TPE BEST's vessel difference is−.007241 with exploratory
paired95% interval[−.014591,−.000693]. Its macro difference against trial4 BEST is
−.004162[−.012850,−.000147], while its LAST improves on trial4 LAST in vessel Dice.
These are different roles/configurations/environments, not a causal single-factor
experiment. A macro-BEST cannot replace the study's fixed-LAST worst-of-six
objective. None establishes superiority over all original heads.

Earlier [control LAST60](../reports/v2-control60-terminal-val),
[vascular2 BEST/LAST](../reports/vascular2-paired-val),
[vascular4 BEST/LAST](../reports/vascular4-paired-val), and the
[corrected original](../reports/frozen-original-val-nearest) remain separate
references. Sampling4 BEST55 improved vessel over sampling2 BEST59, but its macro
interval crossed zero and G1020 disc/cup regressed. The
[threshold study](../reports/vascular2-thresholds) and
[TRAIN-only BatchNorm recalibration](../reports/vascular2-bn-recalibration) did not
justify a new default. The single-photo
[CPU benchmark](../reports/benchmark/strong-pipeline-cpu-drive21-vascular2-best59)
is a scoped latency/size measurement, not general throughput or quality evidence.

## Completed inference experiments on the same vascular4 BEST55

The [four tiling protocols](../reports/vascular4-inference-tiling-val) completed on
all770 VAL photographs with observed child and parent exits0. None had a macro
interval excluding zero;1024/256 reduced vessel Dice by .00075635 with an entirely
negative paired interval. No tiling default was promoted.

The separate [spatial TTA experiment](../reports/inference-tta-vascular4) also
completed all four cases with observed child/parent exits0, unchanged inputs and
read-only reports. FP32, tile512/overlap64, threshold.5, one GTX1060, CPU2/inter-op1:

| Mode | Macro Dice | Change vs fresh none [paired95% CI] | Conditional AV | Invocation seconds |
| --- | ---: | --- | ---: | ---: |
| none | .805973 | reference | .822879 | 204.60 |
| flips2 | .809128 | +.003156[.001669,.004872] | .829177 | 324.17 |
| rot4 | .812370 | +.006397[.004181,.008699] | .834518 | 553.15 |
| dihedral8 | .813082 | +.007110[.004973,.009423] | .835442 | 1001.73 |

All five pooled and macro-head point estimates improved, but DRHAGIS AV regressed
in every ensemble. Dihedral8 artery difference is−.000791
[−.024093,.012267], vein−.003535[−.012605,.009922], with3 supported
photos/groups and1,951 defined draws. Those adverse outcomes accompany the gain.
Full-frame preparation/photometry occurs once; inverse-aligned **joint** AV,
vessel, disc and cup probabilities are averaged before conditional recovery.
Fresh none matches the prior same-device512/64 count/support traces and protocol;
no raw mask/probability bitwise claim is made. Fixed order/warm filesystem qualifies
timing. TTA source/API remains on the separate
[`experiment/inference-tta-science` branch](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiment/inference-tta-science);
this branch imports **reports only**, not that feature. Default remains `none`.
No TTA result is retagged as original-protocol standalone certification.

## Patch/photometry preparation and last observed scheduler state

The [controlled five-arm plan](../reports/patch-photometry-study) fixes the TPE
configuration, batch4 and60×8,192 samples: raw patches256/384/512, plus384 HSV
CLAHE and384 illumination correction+CLAHE. Samples/updates are matched; pixel
exposure, context, memory and compute are not. This historical design document
predates the subsequent completed builds/submissions recorded below.

Both complete **3,299-photo TRAIN/VAL** caches and encoded archives are now built
and verified from frozen `fd44cdf`, with actual builder/archive exits0 and
unchanged canonical metadata. Cache fingerprints are
`b5e781330d2e28b65d70301384fe1e841b64f0e8f631fe04dac3381847664ec2`
(CLAHE) and `b2e42864c0e53b8fec30f55114e30f1ebbb624dd7fbefeccd31c7206cc324b01`
(illumination+CLAHE). The [implementation review](../reports/photometry-cache-root-review-20261111)
is earlier implementation evidence; subsequent operational completion proof is
retained privately (summary SHA
`75389c22e9b535f82a0ce35d7c926027156b3ffea72c5e33124250cdd09d1a54`).
Cache construction is not measured training quality or loader-speed benefit.

All five scheduler submissions were accepted with observed submission exit0 and
were **last jointly observed Q at14:07 UTC**. Raw512 was separately observed Q at
**14:51 UTC**. A **15:05 UTC** status attempt failed with SSH255 and expired
credentials/server-connection error; it returned **no actual job state**. Current
states are **unknown**, not asserted queued or running. No cancellation was
attempted and no own-GPU raw512 fallback has started. Operator job handles,
commands, allocation accounting and access-failure evidence remain in the private
workflow repository. This documentation update performs no GPU or scheduler action.

## RAW35 remains a tiny TRAIN-only canary

The [separate RAW35 branch proof](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/484145dc40117c39f46041d48c046b753870d4fd/reports/raw35-train-canary/README.md)
completed a CPU-only canary on **3 TRAIN photographs /7 annotation rows**, with an
observed owned child exit0. Exact frozen25 RGB/target/valid/FOV replay preceded
native RAW→35 preparation. No VAL/TEST pixels, model, GPU or NN were accessed.
This is not full-corpus35 preparation or native35 trainer/inference support.
Current unified contracts remain strict25; resizing the25 cache would not add
native information. A future35 comparison needs an explicit reviewed producer
and a common frozen25/native VAL GT support grid, not retagged native35 supports.

## Remaining decision and evidence limits

Source0b vascular baselines, source72 V3/TPE and sourcefd44 photometry are distinct
frozen families. Execute each from its declared source/runtime; an overview does
not make an older trainer support new checkpoint fields. Do not alter live or
frozen checkouts to accommodate a different family.

All paired intervals above are exploratory source-stratified percentile intervals
from2,000 shared draws, seed42, resampling612 known split groups. Repeated adaptive
VAL tuning, uncorrected multiple comparisons, unknown patient independence,
historical original-model exposure and cross-backend numerical differences limit
interpretation. No training-seed uncertainty or clinical generalization is shown.
Finish and audit actual arms, retain all heads/sources and adverse outcomes, then
freeze the selected checkpoint/threshold/inference protocol before the separate
reviewed TEST gate. No final selection, TEST permission or release promotion is
created by this snapshot.
