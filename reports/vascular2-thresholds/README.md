# Vascular2 global-threshold VAL experiment

One forward per photograph evaluated all 770 fixed VAL photos at eight predeclared
scalar thresholds. The primary worst-six-ratio rule selected **0.45**. This is an
exploratory VAL result; **the bundled default remains 0.5**, and no final model,
TEST protocol or clinical claim is selected here.

The immutable BEST E59 checkpoint is 81113e7e, canonical manifest 17a2277c and original
reference 182ad346. Actual inference used frozen model package 0b8762e, GTX 1070,
CPU 2 / interop 1 / workers 0 / nice 15. The run took 301.14 seconds. Probabilities were never
saved; only exact counts were retained. VAL input byte fingerprints matched before
and after. No TEST masks, training, canonical writes or other GPU were used.

## All predeclared results

| Threshold | Worst6 ratio | Macro | Vessel | Artery | Vein | Disc | Cup | Conditional A/V |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|0.25|0.975207|0.809472|0.751029|0.743529|0.799108|0.935802|0.867808|0.810262|
|0.30|0.979517|0.809146|0.754349|0.738634|0.795285|0.935645|0.870216|0.810262|
|0.35|0.982021|0.808359|0.756278|0.733720|0.791574|0.935308|0.872128|0.810262|
|0.40|0.983242|0.807228|0.757218|0.728728|0.787774|0.934836|0.873413|0.810262|
|**0.45**|**0.983329**|0.805748|0.757284|0.723556|0.783703|0.934273|0.874318|0.810262|
|0.50|0.982585|0.803900|0.756712|0.718051|0.779482|0.933591|0.874871|0.810262|
|0.55|0.980741|0.801652|0.755291|0.712027|0.775052|0.932770|0.875009|0.810262|
|0.60|0.977664|0.798990|0.752922|0.705354|0.770032|0.931864|0.875047|0.810262|

Worst6 means the minimum of five pooled Dice / original-reference Dice and
conditional A/V accuracy / original accuracy. Exact ties use closest 0.5. Vessel
is still the bottleneck and remains below original 0.770123 at every threshold.
The highest macro occurs at 0.25, where vessel performance is worse. Thus macro
alone would choose a different and less vascular-balanced setting.

## Same-forward 0.45 versus 0.5

Source-stratified resampling used the same 770 photographs / 612 known split groups,
2,000 draws and seed 42. Groups preserve known identities but are not proven
independent biological patients. These intervals are conditional on this selected
checkpoint/threshold; eight-way VAL selection and source/head multiplicity remain.

| Pooled metric | Delta | Paired 95% interval |
| --- | ---: | --- |
| Vessel Dice |+0.000573|[−0.000080,+0.001289]|
| Artery Dice |+0.005505|[+0.005195,+0.005854]|
| Vein Dice |+0.004221|[+0.003741,+0.004816]|
| Disc Dice |+0.000682|[+0.000403,+0.000902]|
| Cup Dice |−0.000553|[−0.001365,+0.000214]|
| Conditional A/V |0|[0,0]|

Macro increased 0.001848, interval [0.001499,0.002396]. Vessel source-macro instead
decreased 0.000572, interval [−0.001237,+0.000129]. The small pooled vessel gain has
an interval crossing zero and does not establish a robust all-source improvement.
Per-source results and exact counts are retained in the paired report and eight
threshold reports.

Conditional A/V discrimination correct/scored/eligible integer counts were
**identical for every photograph at all eight thresholds**. Artery/vein Dice changes
come from the vessel gate, not better discrimination. The same scalar also changes
disc and cup together, explaining their differing tradeoffs. No decoder API,
per-head threshold, model weight or default was changed.

## Producer checks and numerical drift

Nine synthetic/native-producer tests passed. One real TRAIN photograph and the
first actual VAL photograph passed exact same-forward 0.5 counts against the
unmodified frozen evaluator. The existing independent Ada 0.5 report was compared
separately: all 770 photo/record/source IDs, valid supports and GT-positive counts
matched exactly. Numerical count drift occurred in 284 photographs; maximum
absolute change in any photo/head/count was 7. Aggregate differences included
vessel TP+1/FP+24/FN−1 and conditional A/V correct+13. This yields vessel Dice
−0.00000131 and macro+0.00000210 relative to that earlier producer. No tolerance
was used to pretend these are bitwise-identical references. Hardware/thread/backend
differences are not isolated causally.

Threshold effects above compare the **fresh 0.5 control and other thresholds from
the same forward outputs**, so the older producer drift is not attributed to
calibration. `baseline-0.50-existing-producer-drift.json` preserves each count delta.

## Files and completion scope

- `plan.json`: predeclared thresholds/objective/bounds.
- `threshold-*.json`: all eight 770-photo reports, exact counts, source aggregates.
- `best-vs-0.50-paired.json`: paired source/group intervals and integer totals.
- `summary.json`: full scores, input/source/model pins and actual CUDA identity.
- `native-*-parity.json`: exact native same-forward canary gates.
- `progress.log`, `runtime-and-completion.json`, `artifact-sha256.json`: actual
  progress and completion/file identities. The detached supervisor's numeric exit
  code was not retained; all terminal artifacts were verified and its processes
  were observed absent. No inferred exit 0 claim is substituted.

The new helper is an isolated experimental script; no frozen ML source or older
3157-specific threshold watcher was changed. It requires the declared frozen 0b
package on explicit PYTHONPATH and the pinned checkpoint/manifest/reference;
source/report mismatches fail closed. Existing probabilities are consumed only
one photo at a time. Original observed launch paths remain in the local operator
record outside the repository; no raw photographs, masks or probability arrays
are committed.
