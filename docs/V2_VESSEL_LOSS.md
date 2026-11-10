# Optional vessel-loss study for v2

This is an isolated implementation and proposed experiment, **not a measured
model improvement or launch plan**. Source3157, the live v1 study, corpus and
shipping defaults remain untouched. The v1 pooled vessel Dice comparison
(0.770123 reference versus0.760819 student, with negative paired interval)
motivates a vessel-specific study; it does not establish loss imbalance as the
cause. Ground-truth completeness, preparation and dataset mix must be audited
separately. D/C gains must not hide a vessel regression in an aggregate score.

## What the current training actually balances

`UnifiedDataset.sampling_weights` sums inverse populations for each photo's
available dataset×task memberships. Tasks are vessel, conditional A/V, disc and
cup. A source with only vessel supervision contributes one membership; a source
with disc and cup contributes two. Overlapping memberships, photo aliases and
optional sampling-group declarations affect the resulting photo probabilities.
This is not a fixed50/50 vascular-versus-disc budget or equal pixel supervision.
The coverage-first sampler also visits each eligible training photo once before
balanced extras. At the v1 counts,2221/8192 delivered samples per epoch are that
coverage pass;5971 are extras. A different v2 corpus requires recalculating this
ratio and retaining epoch_samples>=eligible training photos.

Foreground crops choose among available positive vessel/disc/cup/known-A/V masks,
with probability0.5 by default. Mild/strong augmentation already jointly changes
geometry with nearest labels and invalidates unsupported outside pixels; basic
retains the core rotations/flips. Stronger augmentation is not automatically
better for thin vessels. This patch changes none of these operations.

The old loss averages available V/D/C BCE+Dice components together with weighted
conditional A/V CE. Missing annotations do not contribute. Averaging a newly
weighted vessel term by its *own* weight would cancel its effect in vessel-only
batches, so the optional multiplier deliberately leaves the old denominator
unchanged.

## Two optional training fields

| Field | Default | Effect |
|---|---:|---|
|vessel_loss_weight|1.0|Multiply the vessel BCE+Dice term; retain the original active-task denominator. This also scales vessel-only batches.|
|vessel_pos_weight|1.0|Scale only the positive term of vessel BCE, including soft positive targets. Vessel Dice, other heads and A/V CE are unchanged.|

With active binary heads B and known A/V support A, the scalar objective is
`(wV*LV + LD + LC + wAV*LAV) / (number_of_active_binary_heads + wAV*I(A))`,
omitting unsupported terms. `LV = wBCE*BCE_pos(pV) + wDice*Dice`.
The existing `av_weight`, `bce_weight`, and `dice_weight` retain their meanings.
Reported vessel components omit the whole-vessel multiplier, analogous to the
existing unweighted A/V component. Training-loss values across weighting choices
therefore do not rank model quality; compare actual validation predictions.

Both fields must be finite and strictly positive. Defaults take the original
code path and preserve v1 loss and gradients bitwise on the tested soft/partial,
vascular, structural and empty-support cases. The positive BCE weight can improve
recall at the expense of precision or score calibration; this needs measurement.
No inference threshold or artery/vein decoding change accompanies it.

Example partial training configuration:

```json
{"vessel_loss_weight": 2.0, "vessel_pos_weight": 1.0}
```

Model architecture, parameter names/counts, RGB input and five outputs are
unchanged. Old checkpoint config dictionaries lack the fields and load them as1.
Their model states and odd-sized predictions remain exact in the tested fixture.
New training stores these values through the existing full-config checkpoint
path. Source-fingerprint resume checks remain enforced: this does not authorize
migration of an ongoing3157 training run into new code.

## Proposed bounded matched study, not executable yet

First freeze the v2 corpus, patient/photo split identities, supervision, source,
configuration and cache pins. Audit label/reference changes separately from a
loss claim. Run a matched v2 control using the selected fixed architecture/LR,
source and data; retain default new weights1. Same RGB preprocessing, seed42,
coverage-first8192 (or a shared larger budget if the corpus requires it),
fixed threshold0.5 and60-epoch cosine horizon apply to every candidate.

Use seven predeclared candidates with one changed axis apiece:

1. Control: BCE1/Dice1, vessel weights1/1,384×384 batch4, mild augmentation.
2. Whole-vessel emphasis: vessel_loss_weight2, everything else control.
3. Positive BCE emphasis: vessel_pos_weight2, everything else control.
4. Dice emphasis: BCE0.5/Dice1.5 (sum remains2), otherwise control. This avoids
   simultaneously increasing total binary-loss scale relative to A/V.
5. Geometry choice256×256 batch8, otherwise control. This is one *coupled*
   geometry/batch ablation; its pixel/update compute differs and must be reported.
6. Basic augmentation, otherwise control.
7. Strong augmentation, otherwise control.

Screen all seven through9 epochs of their unchanged60-epoch schedules. Keep the
control always eligible; promote at mosttwo noncontrols through27 and, if still
competitive,60. Maximum budget with all three completing60 is216 candidate-epochs
(`7*9 + 3*(60-9)`); wall-clock cost must be estimated from actual v2 E1 timings.
An optional paired seed43 control/winner confirmation adds120 epochs and must be
budgeted before starting. Do not combine all apparently favorable switches
without a new matched comparison. No broad extra knob inventory is proposed.

Select on validation evidence with an explicitly vascular-aware policy declared
before results: report vessel pooled and dataset-macro Dice/precision/recall,
per-source paired changes, A/V conditional/end-to-end measures, and D/C changes.
Keep a Pareto view instead of letting D/C improvements alone select a vessel
regression. Freeze the final rule/weights before any heldout evaluation. Reused
v1 evaluation evidence is not a newly unseen external test; do not relabel it.
The separate new study/protocol must include these fields and new source/data
pins; the v1 frozen protocol rejects unknown fields and must not be altered.

## Manual baseline and native ASHA rungs

The v1 `Core.import_baseline` creates a COMPLETE reference with intermediate
values but no native `completed_rung_*` system attributes. In installed pinned
Optuna5.0.0, `_get_competing_values` selects trials carrying the relevant native
rung key; intermediate values alone do not count. Thus its curve is a benchmark
reference, not an ASHA competitor. Actual candidate rung values can compete even
if a later infrastructure stage fails; native membership is not filtered by
final trial state in that helper.

For a **new v2 study only**, explicitly choose and document one policy:

- Reference-only baseline, labeled as such, with no claim that ASHA compares it.
- A matched full-budget v2 baseline deliberately participating in ASHA. At study
  creation, use `optuna.trial.create_trial(system_attrs=...)` with native values
  `completed_rung_0=max(csv[:3])`, `completed_rung_1=max(csv[:9])`, and
  `completed_rung_2=max(csv[:27])`, using the actual fixed-objective curve and
  matching source/corpus/protocol. Keep actual final value/CSV provenance too.

With min_resource3 and reduction_factor3, the next native rung is81, beyond the
60-epoch horizon; do not fabricate a60→81 rung. Intermediate-value step labels
remain actual epochs. Native attributes are version-specific, so pin/test
Optuna5.0.0 and re-review before upgrades. Never modify the finished v1 trial0
or retrofit its historic pruning decisions.

`tools/optuna_coordinator/tests/test_baseline_rung_reference.py` demonstrates both
policies in fresh in-memory studies: a low candidate survives its first rung
when the high baseline has only intermediate values, and is pruned when the
baseline has native rung membership. No actual database was opened or changed.

## Verification scope

57 tests passed on Ada CPU1/nice10/CUDA hidden: the new loss/gradient/legacy-load
checks plus the existing unified suite. Its synthetic functionality/resume tests
use tiny generated fixtures; no real-data training or quality experiment ran.
One first test pass lacked a bundled reference weight fixture; the existing
read-only5.6MiB universal checkpoint was then linked into the isolated test
source and the complete suite passed. Two standalone real Optuna5 in-memory
rung-reference tests also passed. These checks establish behavior/compatibility,
not vessel-quality improvement. Real candidate training is still unlaunched.
