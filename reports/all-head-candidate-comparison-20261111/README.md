# Existing VAL candidates: all heads and actual protocol

Snapshot at research069ef42. These nine rows summarize **existing** completed
770-photo VAL reports, in family/protocol order; they are not a ranking or final
selection. Values below are display-rounded. [comparison.json](comparison.json)
retains full-precision source values, integer TP/FP/FN/valid counts, exact model
roles/configured protocols, coverage provenance and21 byte-SHA input citations.
Macro is the dataset/head aggregation and is **separate from pooled head Dice**.

| Candidate / protocol | Vessel | Artery | Vein | Disc | Cup | Conditional AV | Macro dataset/head |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original reviewed composed reference | .770123 | .689522 | .697286 | .894808 | .435399 | .748566 | .667030 |
| First TPE BEST58 none / .5 | .762882 | .730308 | .792117 | .932671 | .880783 | .824912 | .810895 |
| First TPE BEST58 rot4 / .5 | .766953 | .743989 | .803240 | .937129 | .885207 | .836113 | .818055 |
| First TPE BEST58 rot4 / .35 | .769464 | .759659 | .814793 | .936902 | .884339 | .836113 | .822099 |
| Native V3 trial4 BEST54 rot4 / .5 | .767487 | .754224 | .795406 | .940040 | .880925 | .832472 | .819412 |
| Native V3 trial4 BEST54 rot4 / .4 | .769410 | .764300 | .803133 | .939617 | .878191 | .832472 | .822149 |
| Native TPE6 BEST52 none / .5 | .761709 | .722659 | .779846 | .937703 | .878005 | .806979 | .808853 |
| Curated raw256 BEST52 none / .5 | .759437 | .707847 | .775879 | .933223 | .866570 | .801502 | .801232 |
| Curated LC384 BEST60 none / .5 | .745545 | .736124 | .786222 | .939089 | .859474 | .818850 | .806022 |

The first-TPE none row uses the [matched Ada fresh report](../native-TPE6-matched-fresh-VAL-20261111/README.md),
not the slightly different Old TTA baseline. It and TPE6 use source fingerprint
b269a5e5, FP32/512/64/.5, intra8/inter1. The threshold rows use318e342/198fdd47
on Old, FP32/512/64, intra2/inter1. Their threshold.5 counts reproduce the earlier
same-mode native counters exactly. [Raw256](../curated-raw256-fresh-val-20261111/README.md)
and [LC384](../curated-lc384-best60-fresh-val-20261111/README.md) fresh reports
use1475c264 on Old, FP32/512/64/.5, intra2/inter1; LC applies its saved full-image
`legacy_lc_clahe_v1` profile, all other unified rows use raw. Cross-context numeric
differences are not isolated model/photometry/TTA effects.

Original [182ad346](../frozen-original-val-nearest/README.md) is the reviewed
composed four-model VAL reference after the MATLAB nearest-coordinate correction.
Historical TRAIN exposure, raw/normalized provenance, FOV/preprocessing and
backend qualifications remain unresolved. It is not an independent clinical or
previously blinded reference. All unified rows have original completed60 runs and
reported **own selected**2529-photo/5700-annotation coverage, with evidence per
row. BEST58/54/52/60 roles and epoch-dependent coverage remain distinct; the
original reference has no comparable established own-TRAIN coverage claim.

Existing paired vessel-minus-original intervals, copied without recomputation:

| Exact candidate context | Delta | Existing paired95%CI |
|---|---:|---:|
| First TPE BEST58 none/.5, Ada | −.007241203 | [−.014590512,−.000693086] |
| First TPE BEST58 rot4/.35, Old | −.000659473 | [−.006530228,+.005516923] |
| V3 trial4 BEST54 rot4/.4, Old | −.000713109 | [−.007253184,+.006346865] |

The JSON binds these intervals to their original input pins, including the
first comparison's derived-view SHAs. Other entries are null because no exact
versus182 interval was identified in these inputs; parent/none intervals are not
substituted. These exploratory2000/seed42 source-stratified paired intervals use
612 known groups, with adaptive-VAL/multiplicity and unknown-patient-independence
qualifications. No training-seed uncertainty or all-task superiority is implied.
Adverse source/head outcomes remain in the cited complete reports. Both calibrated
protocols above still have pooled vessel point estimates below original.

## Measured CPU example, kept separate from VAL quality

The [same-run benchmark](../benchmark/tpe72-tta-cpu-drive21/README.md) uses exactly
**four original networks**: dedicated vessel single_net; dedicated raw Disc0
single_net; universal single_net with only cup retained and clipped by the
same-forward Disc0; and one AV classifier that produces both artery and vein
from calibrated vessel/FOV inputs. Five output masks do not mean five networks.
Alternate bundled weights are not additional models in this measured pipeline.
The benchmark's byte-pinned script3a0324e0 is recoverable at public64e834db;
its ML runtime source was separately frozen at1f723ad9.

Measured parameters: **5,943,312 across those four original models**, versus
**1,965,637 in the exact first-TPE20d4caab checkpoint**. The weight hashes match
the original182 components, but the benchmark reference528233fe and raw-photo
geometry/context are distinct from the182 full-VAL evidence. Parameter counts
are not inferred for V3, TPE6 or curated checkpoints.

| Same resident CPU run | Three measured calls, seconds | Median seconds |
|---|---|---:|
| Original four-network pipeline | 59.594548 / 59.801211 / 59.330616 | 59.594548 |
| First TPE none, threshold.5 | 8.806916 / 8.748764 / 8.701024 | 8.748764 |
| First TPE rot4, threshold.5 | 33.199672 / 33.493771 / 33.015030 | 33.199672 |
| First TPE dihedral8, threshold.5 | 66.346643 / 64.175714 / 65.531443 | 65.531443 |

One DRIVE TRAIN RGB/FOV photo, Ryzen5 1600, CPU1/intra1/inter1/OpenCV1, FP32,
first-TPE512/64/.5; one warmup plus three measured calls per group in fixed order.
Resident preprocessing/inference/decoding/restoration are included; file IO,
model loading and CLI startup are excluded. Rot4 was about1.80× faster than this
original pipeline; dihedral8 was about10% slower. No timing is claimed for.35,
other candidates, a representative corpus or a final deployment. Whole-process
RAM and all timing samples/qualifications remain in the source benchmark.

**Unfinished at this snapshot:** raw384, raw512, CLAHE384 and auxiliary Ada trial7.
They have no row, rank or selected status here. Compare their eventual genuine
completed/fresh evidence before an explicit model/protocol decision. This report
creates no model card quality claim, freeze, TEST evaluation, default or promotion.

Validation was metadata-only: nine source metrics copied exactly, common pooled
valid/positive GT support checked from stored counts, three CI input bindings
checked, all21 input byte SHAs retained, executed benchmark script SHA and four
component hashes verified. No ML imports, model/GT decoding, NN/GPU, new statistical
bootstrap or source/result mutation occurred. These projections cannot substitute
for original typed completion or fresh final-selection evidence.
