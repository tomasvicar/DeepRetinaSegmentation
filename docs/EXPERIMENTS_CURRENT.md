# Current full-data research evidence

Snapshot: **2026-11-11 22:06 UTC; TPE6 fresh VAL and LC BEST assessed; four training jobs still active**. Completed VAL and
CPU tradeoff reports are reviewed below. A last observed scheduler state is not a
claim about current execution. The final optimized model, selection freeze and held-out TEST
comparison are **not finished**. No released default is changed by this overview.

## Product and common data

Use [`for_Leipzig`](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/for_Leipzig)
for installation, prediction and the original verified checkpoints plus explicitly
identified unified baselines/previews. This research branch is separate. Inference
needs no MATLAB. The original stored-input arithmetic is numerically checked;
calibrated raw A/V retains measured photometric differences and unresolved
historical training-file exposure. See the product's `docs/MATLAB_AV_PROFILE.md`.

The completed first TPE BEST58 is now also available there as
`unified-v3-tpe-preview`, with exact original20d4caab… weights and optional
`--tta rot4`. The [model card and retraining recipe](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/for_Leipzig/docs/UNIFIED_V3_TPE_PREVIEW.md)
keep it explicitly experimental, with no TEST/default-winner claim. It is not the
later native trial6 run described below. All eleven previous model blobs
and original aliases remain unchanged.

Canonical V2 has **2,529 TRAIN photographs / 5,700 annotation records from21
sources**, **770 VAL** photographs and856 TEST photographs. Supported annotations
alone supervise each output. Recovered patient/photo aliases remain grouped;
unknown patient identity is not thereby resolved. No new TEST evaluation is
reported here. Conditional A/V accuracy scores reference class-labeled vessel
pixels and does not measure vessel discovery by itself.

One further genuine native TPE proposal now continues the same auxiliary study
after all six completed E27 observations. Width16/BatchNorm/basic stays fixed in
this suggestion; LR becomes0.000411624945, weight decay0.0000823809785 and vascular
sampling1.05918806. It is distinct from all five patch/photometry arms. The new
observer actually launched at16:24:39UTC; a later16:26:54UTC observation
confirmed the training NN live on the physical local Ada GPU with8 intra/1
interop threads and8 workers. E1 was then observed complete at16:30:31UTC, with
2,048 updates, finite training loss and157.11s TRAIN+64.32s VAL. This first epoch
is neither full-budget completion nor evidence of final quality; its very early
internal VAL score is not used as a model-quality claim. Six observations do not
establish optimality.
[Native continuation provenance](../reports/tpe72-completed-fresh-val/next-native-TPE-proposal).

The same Ada run [completed60 with actual original exits0 and generic QA0](../reports/next-native-tpe72-completed-20261111)
at20:08:57UTC. BEST52 and LAST60 each independently cover2529photos/5700annotations.
The genuine fixed-LAST27 proxy.943596514663 was then told to native auxiliary
trial6 with actual CLI0, below prior trial5.975134762451; the auxiliary study now
has7 COMPLETE observations (5 imported,2 native), previous6 and sampler unchanged.
[Matched fresh TPE6 BEST/LAST VAL](../reports/native-TPE6-matched-fresh-VAL-20261111)
then completed both770-photo roles with actual child/outer exits0. BEST macro.808852833,
vessel.761709005 and conditionalAV.806978606 trail the matched first-TPE roles.
BEST conditionalAV change−.01793294[−.02433401,−.01129130]; LAST vessel
−.00451966[−.00832313,−.00031107]. Macro intervals cross zero; the full all-head
and source differences, known612-group uncertainty and tiny saved/fresh count drift
are retained. No quality promotion follows.

A third actual native proposal, trial7, continued that same sampler with all7
previous observations preserved. It chose LR.0001249535964, WD.000001641515876,
vascularsampling2.6995567465 with width16/BatchNorm/basic, without manual changes.
The original03UTC start window was missed and its disabled packet/defer retained.
A separate explicitly reviewed own-Ada window ends04UTC/max12h, with the same
326s/epoch+1800sQA+300scleanup conservative budget. Training actually started
21:13:51UTC; GPU/config/intra8/inter1/workers8/BF16 were observed. E1/2048updates
completed with finite loss and236.834s TRAIN+VAL. At22:06UTC the same observer,
supervisor and trainer were live, with E13/26624 updates completed and finite
loss/VAL. This is unfinished training,
not a quality result. No extra Meta job, main-study mutation or TEST was used.

The [native V3 trial4 BEST54 TTA sweep](../reports/v3-trial4-inference-tta-val)
completed all four modes on all 770 VAL with observed child and external supervisor
exits0. Separate LAST60none also completed770 VAL. Total42m55.7s under the unchanged
5400s/60s envelope; source/input hashes stayed unchanged. Genuine `pool_72_v3`
completion/selected-role proofs are retained, with own5700 annotations/2529 TRAIN
photos in both BEST54 and LAST60. Fresh source318e342/evaluation198fdd47 remains
separate from original checkpoint training3725; no standalone evidence is fabricated.

Fresh BEST54 none macro.815058934 increases to.816065691 with flips2,
.819411772 with rot4 and.819308183 with dihedral8. Direct8minus4 macro is
**−.000103588 [−.000497407,+.000299564]**; the interval crossing zero does not
establish equivalence. Vessel gains a small+.000246610[+.000116107,+.000368509],
while IOSTAR artery/vein Dice and conditional AV have negative direct 8 minus 4
intervals (IOSTAR AV−.001555724[−.002377373,−.000765184]). Aggregate conditionalAV
rot4.832471660 versus8.832737434 has a crossing zero difference interval. RIMONEcup
also has an adverse point. Full all head/source/AV counts and intervals are exported.
Separate LAST60none macro.804571577 is below BEST54none; roles are not substituted.

All GT support matches the saved native sidecar, but fresh BEST54none prediction
counts are not bitwise equal (macro+1.67222e−6); the saved sidecar is not a previous
independently fresh control. Same-mode V3minusfirstTPErot4 macro+.001356551 has
CI[−.001441812,+.007475376], with lower pooledvein/cup and conditionalAV despite
higher artery/disc/vessel pointDice. Source/model contexts differ; no isolated model
causality or blanket dominance is claimed. The earlier TPE8minus4 result is distinct.

The [reviewed TTA selection route](FINAL_SELECTION_TTA.md) binds exact inference
descriptors to fresh validation and later frozen execution. No final model, default
promotion or TEST selection is made here. The completed
[V3 BEST54 rot4 scalar threshold sweep](../reports/native-v3-best54-rot4-thresholds)
used eight prespecified thresholds on all770 VAL, with same-array native.5 count
parity for every photograph and observed child/outer exits0. A prior attempt
failed before model loading on a sibling import; its failed evidence is retained.
The declared worst-six ratio chose.4: macro.822148867, vessel.769410290 and
conditionalAV.832471660. Against.5 vessel gains.001923150 with exploratory paired
CI[.000349697,.003582857], while cup loses.00273378 with entirely negative CI.
DRHAGIS/CHASE vessel and G1020/RIM disc also regress. Vessel still falls below
the original.77012340 point reference; its difference interval crosses zero.
This does not justify a default, final selection or TEST. The
[first-TPE BEST58 rot4 threshold study](../reports/first-TPE-BEST58-rot4-threshold-20261111)
completed all770 VAL with actual child/outer exits0. The declared worst-six rule
chose.35: vessel.7694639263, macro.82209916, conditionalAV.8361133069.
Vessel change versus same-forward.5 is+.00251081 with an interval crossing zero;
versus original it remains−.00065947[−.00653023,+.00551692]. ConditionalAV is
exactly invariant over all8thresholds; source regressions and all paired intervals
are retained. Worst-six point.999143678 remains below1. This distinct standalone
model is not the V3 trial4; no default, equivalence or TEST claim follows.


## Completed patch256 and pending patch/photometry arms

[Raw256 fresh BEST52/LAST60 validation](../reports/curated-raw256-fresh-val-20261111)
completed on all770 VAL with actual child/outer exits0. Original fd44 training
completed60, PBS finished Exit0, and both roles have their own2529-photo/5700-
annotation coverage. The full saved configuration and preprocessing are retained.
Fresh BEST52 macro.801231863, vessel.759437346 and conditionalAV.801502046 all
trail the parent TPE384 NONE control (.810897419/.762879016/.824919544).
Paired BEST macro change is−.00966556[−.014665,−.004196], vessel−.00344167
[−.007268,−.000798], conditionalAV−.0234175[−.030766,−.015907]. LAST60 macro
is.799432520 and also regresses. All-head/source counts and adverse source
intervals accompany the report. Different training/evaluation source contexts
prevent an isolated causal patch-size claim. Raw256 is not promoted or bundled.
Saved/fresh BEST counts have small numeric drift (macro−1.19e−6), with identical
GT support; the report preserves this distinction.

MetaCentrum authentication was renewed and independently checked. LC384 finished
PBS Exit0 at19:57:52UTC, allocation3h24m40s. The original completion receipt and
all60 history rows are preserved. Both BEST and LAST are E60 with their own
2529-photo/5700-annotation coverage, independently rechecked on CPU with actual
CLI exits0; finite state, optimizer/horizon and the complete nonraw configuration,
operator/backend/cache were retained. See the [genuine LC audit](../reports/curated-fd44-completion/genuine-lc384-two-role-audit.json).
[LC BEST fresh validation](../reports/curated-lc384-best60-fresh-val-20261111)
completed all770 photos with actual child0. Fresh macro.806021640, vessel.74554493
and cup.85947379 show adverse vessel/cup changes against parent raw TPE NONE:
−.0173341[−.0227600,−.0125179] and−.0213102[−.0266982,−.0161283]. Disc improves
+.0064172[+.0006127,+.0125841]; macro/artery/vein/conditionalAV intervals cross0.
The planned LAST inference did not complete before the original work deadline;
whole observer returned1, stage timed_out=true, BEST0 retained, inputs unchanged.
This partial inference outcome is distinct from successful full60 training.
No automatic retry, budget change, LAST metric or promotion is claimed. Full
counts, descriptor/config, source adverse intervals and outcome are published.

Raw384, raw512 and CLAHE384 remain unfinished; their last actual scheduler/progress
snapshot21:57UTC was R/E56, R/E39 and R/E40. No new Meta job was submitted.
Current elapsed scheduler time is not model quality, and unfinished arms are not
ranked. Product `for_Leipzig` supports saved full-image photometry profiles without
stripping config fields. Real TRAIN inference parity checks for the existing TPE
and genuine raw256 checkpoint were bitwise exact (max probability error0).
The reviewed independent curated completion adapter and compatible HPO photometry
port are integrated. The [explicit curated eligibility adapter](CURATED_FD44_SELECTION_PROPOSAL.md)
now requires its distinct original index/SHA, role-owned coverage and freshly bound
VAL header.105 focused integration tests and a genuine LC CPU adapter check passed
(actual0,118.04s, no forward/GT decoding/GPU/selection/TEST). Existing family/raw
protocol contracts remain. Historical untyped reports are not newly eligible;
no actual candidate/source has been frozen or held-out data evaluated.

The separate [TPE BEST58 TTA sweep](../reports/tpe72-inference-tta-val) completed
all four770-VAL cases with observed child and parent exits0, unchanged inputs and
read-only reports. Fresh same-device none macro.810897 improved to.815087 with
flips2,.818055 with rot4 and.818854 with dihedral8. The8-view change is+.007956
[.005841,.010113]; conditional AV.837810 and vessel.767147. All five pooled/source-
macro heads improved versus none, but RIMONE cup regressed (8-view−.002523,
interval[−.013998,.007148]) and vessel still trails frozen original.77012340.
The same-device none counts differ slightly from prior Ada source72 counts
(359 photos/469 head-photo entries, macro+1.98535e−6); GT supports match exactly.
Those contexts are not retagged as bitwise parity. No default, original-protocol
certification, final selection or TEST status changed.

The new [direct resident CPU tradeoff](../reports/benchmark/tpe72-tta-cpu-drive21)
completed one run on one DRIVE TRAIN raw photograph/official FOV, one CPU thread,
with16 operations and actual exit0: same-run original59.59455s, TPE none8.74876s,
rot433.19967s and dihedral865.53144s medians. **Eight views were about10% slower
than original on this photo**; rot4 about1.80× faster. None's8.75s is not8-view
latency. Every group used1warmup+3measured calls, finite bool masks and checks
outside clocks; defaultnone exactly matches the prior completed CPU mask hash.
Peak whole-process RSS2,967,872KiB (~2.83GiB), FP32 inference with unchanged BF16
training metadata, no optimizer state/compression/quantization, no GPU/GT/TEST.
These timings exclude IO/model loading and measure no CLI startup or quality.
The earlier [none-only CPU example](../reports/benchmark/strong-pipeline-cpu-drive21-tpe72-best58)
remains separate historical evidence. No rot4 default or4-versus8 quality
equivalence is inferred; consult the actual exploratory VAL intervals/adverse heads.

## Completed full60 models: keep roles and evidence distinct

| Model and role | VAL evidence | Macro dataset Dice | Pooled vessel Dice | Conditional AV |
| --- | --- | ---: | ---: | ---: |
| V3 control0 BEST59 | Saved native counts, typed pool eligibility | .801281 | .760570 | .801055 |
| V3 control0 LAST60 | Saved native counts, typed pool eligibility | .801188 | .758714 | .808460 |
| V3 trial4 BEST54 | Saved native counts, typed pool eligibility | .815057 | .763145 | .823922 |
| V3 trial4 LAST60 | Saved native counts, typed pool eligibility | .804570 | .750500 | .823318 |
| V3 trial4 BEST54 none | Fresh Old VAL770, typed pool role | .815059 | .763145 | .823917 |
| V3 trial4 BEST54 rot4 | Fresh Old VAL770, typed pool role | .819412 | .767487 | .832472 |
| V3 trial4 LAST60 none | Fresh Old VAL770, typed pool role | .804572 | 0.750503 | .823324 |
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
not modified by that bookkeeping. A later additional proposal was asked separately,
as recorded in the native continuation above.

Fresh TPE counts differ slightly from saved counts: LAST changes6 photos/7
head-photo entries, macro−4.29046e−8; BEST changes5 photos/7 entries,
macro−1.00598e−7. All GT supports and attribution remain exact. Actual fresh flags,
cache/backend distinctions and integer deltas are disclosed; no counterfactual
forward was run to assign a cause. Use the fresh counts for those fresh reports.

The frozen original vessel reference is .77012340. Every V3/TPE vessel value above
still falls below it. TPE BEST's vessel difference is−.007241 with exploratory
paired 95% interval[−.014591,−.000693]. Its macro difference against trial4 BEST is
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
all 770 VAL photographs with observed child and parent exits0. None had a macro
interval excluding zero;1024/256 reduced vessel Dice by .00075635 with an entirely
negative paired interval. No tiling default was promoted.

The separate [spatial TTA experiment](../reports/inference-tta-vascular4) also
completed all four cases with observed child/parent exits0, unchanged inputs and
read-only reports. FP32, tile512/overlap64, threshold.5, one GTX1060, CPU2/inter-op1:

| Mode | Macro Dice | Change vs fresh none [paired 95% CI] | Conditional AV | Invocation seconds |
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
timing. TTA originated on the separate
[`experiment/inference-tta-science` branch](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiment/inference-tta-science)
and is now integrated here. The [high-level prediction fix](../reports/highlevel-predict-tta)
also forwards `--tta` through the public CLI and Python prediction API, recording
the existing exact descriptor. Root91 merged synthetic/CLI checks passed13.81s;
no actual model/VAL/TEST or existing low-level operator changed in that fix.
Default remains `none`.
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

All five original scheduler submissions were accepted with observed submission
exit0. The15:05UTC SSH255/expired-credential observation failure was retained;
it did not establish a job state or trigger a restart. Authentication was later
restored. Raw256 and LC now genuinely finished PBSF/Exit0, with actual audits and
fresh-VAL outcomes described above. At21:57UTC raw384/raw512/CLAHE384 were still
R/E56/E39/E40. No cancellation, replacement submission or own-GPU raw512 fallback
was made. Operator handles, commands, original allocation accounting and the
access-failure evidence remain in the private workflow repository. This document
does not perform a scheduler action or claim those unfinished models complete.

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
