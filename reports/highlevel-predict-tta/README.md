# HPO high-level prediction TTA handoff

The research high-level `predict_unified` API and ordinary `deepretina predict`
CLI previously always used the low-level default-none path. They now expose
`none`, `flips2`, `rot4` and `dihedral8` and forward explicit opt-in modes to the
existing implementation. The module prediction CLI supports the same modes.
Custom checkpoints and model aliases enter the same unified route. Non-unified
models reject opt-in TTA rather than silently ignoring it.

`tta="none"` remains the default. Its low-level call and existing provenance
schema remain unchanged (no TTA descriptor means none). Opt-in `prediction.json`
provenance contains the existing complete resolved mode/operator/joint-fusion/
precision descriptor. No ensemble mathematics, model/training/data/photometry
operator, selection gate, alias registry or bundled weight changed.

Example explicit prediction protocol, without changing any default:

```sh
uv run deepretina predict --checkpoint candidate.pt --input photo.png \
  --output prediction --tta rot4 --threshold .35 --tile-size 512 --overlap 64
```

This is a prediction example, not an evaluated release/default or a final
selection command. No frozen selection is automatically inferred from a model
filename or alias; the explicit chosen protocol must accompany a handoff.

## Read-only final-selection readiness finding

The existing first-TPE threshold report is an exploratory
`standalone72_BEST58_scalar_threshold_VAL_ablation`, explicitly
`ordinary_freeze_evidence=false`. Its original model is BEST58 SHA256
`20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`,
trained under source72. It was evaluated under318e342/package198fdd47, not the
newly staged f13 evaluation source. It cannot be renamed into fresh final evidence.

The exact original770-photo VAL reference is
`reports/frozen-original-val-nearest/validation.json`, SHA256
`182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836`.
It is a frozen composed original reference with documented historical exposure,
raw/normalized provenance, preprocessing/FOV and backend qualifications. Those
qualifications remain in force.

The current final APIs already accept `completion_family=standalone_72`,
`checkpoint_role=training_best`, `tta=rot4`, scalar threshold.35 and512/64.
Eligibility binds genuine BEST58 to the separate actual completed LAST60, original
QA/protocol/terminal and CSV/summary, verifying each selected weight's own coverage.
It does not require BEST to be epoch60. Selection pins the fresh protocol and
source/config/data/metrics; the separately reviewed held-out route reads TTA and
threshold from that protocol and forwards them unchanged. No new selection-proof
schema or freeze-gate support is missing for this standalone candidate.

What remains, if this candidate is chosen after pending experiments finish:

1. Actual fresh enriched typed VAL770 evidence under a single reviewed source/
   runtime, using the genuine BEST58, rot4, threshold.35 and512/64, and retaining
   original standalone source/epoch/QA/protocol/terminal provenance. The immutable
   f13 checkout already staged on Old remains unchanged by this feature commit.
2. Review the competing completed models and qualified baseline comparisons, then
   explicitly record the chosen checkpoint/role/protocol and review rationale.
   Only then can existing selection create/check APIs bind actual fresh inputs.
3. Hand off that exact inference protocol with the model. This change closes the
   ordinary-prediction TTA forwarding gap; it does not make a selection decision.

At threshold.35 the recorded vessel Dice.769463926 remains below original
.770123400, with paired interval crossing zero; worst-six ratio.999143678 is below1.
The report retains adverse source/head outcomes and adaptive-VAL qualifications.
Current selection metadata APIs check consistency and provenance; they do not
certify achievement of an all-six superiority objective. No quality equivalence,
all-task improvement, final model choice or TEST result is claimed here.

## Validation

18 focused tests passed in3.54s, followed by47 complementary CLI/TTA regressions
in9.84s (65 distinct tests). Actual tiny synthetic CPU forwards use base_channels2,
depth1, generated13x19 RGB and a generated FOV; no training or optimizer is run.
They verify all four view counts, finite FP32 output, state preservation, default
none arrays/mask bits/RNG, one preprocessing operation, exact opt-in descriptor,
CLI threshold.35 metadata, alias/custom forwarding and invalid/non-unified-mode
rejection. Existing low-level TTA and typed-selection regressions remain passing.

Tests ran CUDA-hidden, CPU1/core6, intra1, nice17/idle IO, with a60s cap per run.
No actual model/checkpoint, real dataset/GT, GPU, VAL/TEST run or frozen selection
was used or created. The staged6d3593f/f13 source and all previous results remain
untouched. This feature changes source fingerprints for future checkouts only;
updating an operational evaluation checkout is a separate action.
