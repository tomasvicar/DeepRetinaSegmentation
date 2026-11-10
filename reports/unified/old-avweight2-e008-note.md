# AV loss-weight screen completed, 2026-11-10 17:53 UTC

The controlled `av_weight_2_lr0003` screen completed eight epochs on old/GTX1060.
Its exact supplementary plan is `afeb05f71945824ea415e2c9aacdcd33d540305dac18ab770f714db63445aa05`,
with immutable source `87cf045`, execution `8172e06b…`, training-source `45e9254a…`
and canonical data `61032be2…`. This is not the original `907713f6…` plan.

Terminal CPU1/nice10/CUDA-hidden QA passed: both weight sets and validation
metrics are finite; resumable last-checkpoint optimizer/state is finite;
65536 samples, 16384 optimizer steps, all 2221 eligible training photos visited;
cosine scheduler horizon remains 60 and scheduler epoch is 8. The best and last
checkpoints are both epoch 8. The original PID 293314 is absent. No training was
resumed and no test data was evaluated.

Independent copies of best/last and scalar/count metadata are preserved at
`~/code/DeepRetinaSegmentation_unified_trials_87cf045/milestones/av_weight_2_lr0003-seed42/epoch-008`
on old. Source SHA before/after copying and destination SHA matched. Best SHA is
`227533b963ed838940430603e65f9e71eb0287158d2588665985b291bfa7497c`;
last SHA is `b8a6e2dcdd73d7fc4de943e7175d265af854f386adcb39948d36e81878a1f5be`.
The original checkpoint format intentionally omits optimizer/RNG state from
`best.pt`; optimizer QA applies to resumable `last.pt`.

| Validation metric | AV weight 1 control E8 | AV weight 2 E8 |
|---|---:|---:|
| Five-head source macro Dice | 0.708936 | 0.714549 |
| Pooled vessel Dice | 0.722513 | 0.714392 |
| Pooled artery Dice | 0.561068 | 0.574194 |
| Pooled vein Dice | 0.615372 | 0.618762 |
| Pooled disc Dice | 0.849109 | 0.865039 |
| Pooled cup Dice | 0.791965 | 0.800265 |
| Conditional AV accuracy | 0.652636 | 0.661485 |

| Source macro Dice | AV weight 1 E8 | AV weight 2 E8 |
|---|---:|---:|
| Vessel | 0.748667 | 0.736253 |
| Artery | 0.542104 | 0.562302 |
| Vein | 0.615946 | 0.619420 |
| Disc | 0.856720 | 0.869492 |
| Cup | 0.781242 | 0.785278 |

`old-avweight2-e008-comparison.json` contains full per-source TP/FP/FN/valid
counts, supported-photo counts, pooled and source-macro changes, conditional and
end-to-end AV accuracy. The saved `old-avweight2-e008-counts.json` comes directly
from the verified checkpoint's validation metadata; no forward was performed.
All 770 photo IDs, selected annotation IDs, channel-source assignments, valid
pixel counts and GT-positive counts match the weight-1 control exactly.

Fourteen annotation sources contribute validation support: REFUGE (400 disc/cup),
G1020 (204 disc/170 cup), AVRDB (20 vessel/AV), RIM-ONE (32 disc/cup), Erlangen
(22 vessel), HRF (9 vessel/AV), ARIA (28 vessel/22 disc), STARE (4 vessel), ONHSD
(17 disc), CHASE (6 vessel), DRHAGIS (8 vessel/3 AV), Drishti (10 disc/cup), IOSTAR
(6 vessel/AV/disc) and RITE (4 vessel/AV). These are supported photo counts per
head, not mutually exclusive whole-photo source totals. RIGA and UoA remain
training-only; DRIVE/RITE shared photos retain their selected annotation source.

These are one-seed eight-epoch validation screens, not final models. The weight-2
screen improves artery/vein point metrics relative to its weight-1 control while
vessel Dice declines. It still trails the calibrated original AV reference's
artery Dice 0.689529, vein Dice 0.697298 and conditional accuracy 0.748573.
Historical exposure and calibrated-reference qualifications remain applicable;
there is no overall superiority claim.

The independent optional cup-containment CPU study remains healthy on Ada:
PID 1278841, 60/612 supported validation photos processed at 453.5 seconds,
all native reproduction gates passed so far, current RSS 820 MiB. This is a
progress observation, not a completed composite result. No process was restarted.
