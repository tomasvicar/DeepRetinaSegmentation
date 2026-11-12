# Final accepted handover · 12 November 2026

The user accepted the small measured vessel tradeoff and requested finalization,
without further optimization. Both the original networks and the separately named
**`unified-raw512-selected`** are supplied. The latter is raw512 **BEST50** from a
completed 60-epoch run; its own checkpoint covers **2,529 TRAIN photographs /
5,700 annotation records / 21 sources**. Its frozen inference is **rot4 / scalar
0.35 / tile512 / overlap64**, FP32. Historical previews and pilots are not this
final model. No new weights, defaults or inference mathematics are introduced by
this handover document.

## Install, predict, retrain

Use Python3.12 and `uv`:

```bash
git clone --branch for_Leipzig https://github.com/tomasvicar/DeepRetinaSegmentation.git
cd DeepRetinaSegmentation
uv sync --frozen --dev
uv run deepretina doctor
uv run deepretina models
uv run deepretina predict --model unified-raw512-selected \
  --input fundus.png --fov official-fov.png --dataset drive \
  --output predictions/unified --device cpu
uv run deepretina predict-original \
  --input fundus.png --fov official-fov.png --dataset drive \
  --output predictions/original --device cpu
```

Supply an actual photograph, its matching FOV and correct camera/dataset profile;
use a new or empty output directory. No training-data access or weight download is
needed for prediction. [Original API and input contract](PREDICT_ORIGINAL.md),
[selected unified API](UNIFIED_RAW512_SELECTED.md), [unified preparation/training](UNIFIED_QUICKSTART.md),
[general retraining](RETRAINING.md), and [native original-architecture retraining](NATIVE_LEGACY_RETRAINING.md)
provide the exact recipes. Native functional smoke weights are not released models.

## Final quality and accepted limitation

These are matched, frozen **856-photo TEST** results, after VAL-only selection.
Both executions and the external observer completed0. The first five rows are
pooled Dice; AV rows are accuracies. Dense support is102photos/71known groups,
excluding only positive-only AVRDB vessel annotations.

| Metric | Original four networks | Selected unified |
| --- | ---: | ---: |
| Vessel | 0.787840026 | 0.786564780 |
| Artery | 0.695384358 | 0.784664776 |
| Vein | 0.706571583 | 0.832444103 |
| Disc | 0.892787774 | 0.946856475 |
| Cup | 0.454490768 | 0.885679944 |
| Source/head macro Dice | 0.681857637 | 0.839614248 |
| Conditional AV | 0.751861050 | 0.860869209 |
| End-to-end AV | 0.641227548 | 0.752939771 |
| Dense vessel | 0.780767926 | 0.776547691 |

The accepted dense-vessel difference is **−0.004220235 Dice
(−0.422 percentage points)**, paired95% interval
**[−0.006314644, −0.002161060]**. Strong A/V/disc/cup gains do not establish
all-task superiority. Pooled-vessel interval crosses zero; this is not proof of
equivalence. Keep the original pipeline when measured vessel quality is the
priority. [Full results, all adverse sources, support and provenance](../reports/release/selected-raw512-TEST-20261112/README.md)
retain the limitations: known groups are not proven independent patients,
historical original-weight exposure is unknown, and clinical generalization is
not established. No post-TEST selection or tuning is part of this release.

## Size, speed and numerical scope

The unified network has **1,965,637 parameters**, versus **5,943,312** across the
complete four original networks. Original files total25,068,700B; the measured
selected checkpoint is8,893,810B. The CPU deployment export is8,892,466B.
On the same GTX1060, one warmup and three measured resident-pipeline calls gave:

| Fixed TRAIN geometry | Original median | Unified median |
| --- | ---: | ---: |
| DRIVE21, 584×565 | 2.803334s | 1.081339s |
| HRF04, 2336×3504 | 5.820297s | 3.367838s |

This includes preparation/inference/restoration/decoding, excludes loading,
file I/O, writes and CLI startup, and does not establish population speed.
[Exact timing and memory evidence](../reports/release/selected-raw512-two-TRAIN-GPU-timing-20261112/README.md)
measured original selected **521b…2cea**, not the separate deployment export
**292e…9b29**. Exact selected/export SHA256 values are
`521b8881d1b2c69a7395be6687a8b86581838e3f5dbb004a150333d454132cea` /
`292ea5999a1fc97a767c1c34bb7b4558040267372c5b128f403f2dce50529b29`. The latter has exact tensor/probability/five-mask TRAIN parity;
its serialization difference is not parameter compression.

Original networks preserve verified numerical behavior on prepared inputs.
`predict-original` also matched the literal four-network Python benchmark on one
real TRAIN photo (five mask differences all zero). This does not claim universal
raw MATLAB parity: supplied-FOV calibration and known raw A/V photometry residuals
remain explicit in [MATLAB_AV_PROFILE.md](MATLAB_AV_PROFILE.md) and
[the real composed parity report](../reports/original-composed-real-parity-20261112/README.md).

## Requirements and existing actual evidence

| Requirement | Existing evidence and boundary |
| --- | --- |
| Originals and simpler unified model remain usable | [Model card/identities](UNIFIED_RAW512_SELECTED.md); all12 older files/aliases/defaults preserved beside the13th model. |
| Full suitable TRAIN participation | [Frozen selection](../reports/release/raw512-selection-20261112/README.md): selected BEST50 own2529/5700/21 and genuine full60 completion; not merely a planned epoch count. |
| Independent held-out evaluation after freeze | [Completed TEST](../reports/release/selected-raw512-TEST-20261112/README.md): all856, exact support, original report SHA pairs and exits; no claimed unknown-patient independence. |
| Export and usable CLI/API | [Actual TRAIN parity/export](../reports/release/unified-raw512-selected-20261112/README.md): exact probabilities/five masks, same model tensors, distinct artifact identity. |
| Installation outside checkout | [Final wheel build/install verification](../reports/release/raw512-final-metadata-wheel-20261112/README.md): actual0, all13 hashes, aliases/defaults/module origins; metadata update only, no repeated NN. |
| Software checks documented | [TESTING.md](TESTING.md) distinguishes source-bound full-suite and focused checks; no new suite was run for this docs-only finalization. |
| Retraining is functional | [Native real smoke](../reports/native-legacy-real-smoke/README.md): all5 production presets functional; AV CPU3GiB defer retained before successful GPU one-step check, not full-corpus quality. |
| Quality/size/speed fairly scoped | Full TEST table above and two fixed TRAIN timings; no all-task superiority, equivalence or universal speed claim. |
| Research and compute provenance retained | [Public experiment history](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/experiments/leipzig-hpo/docs/EXPERIMENTS_CURRENT.md); actual PBS profiles, job exits, resource receipts and usage ledger remain in private `metacentrum-workflows`, not required for inference. |

## Closed follow-ups and deferred work

The completed [native scratch LAST60 comparison](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/native-topology-LAST60-20261112)
gave dense VAL0.74091458, below selected0.75131596. Against its matched compact
LAST60 control, dense change+0.002402 had an interval crossing zero while macro
change−0.016561 had a negative interval. The completed [native BEST58 rot4/eight-threshold follow-up](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/leipzig-hpo/reports/native-topology-BEST58-rot4-scalars-20261112) also did not justify replacement: maximum dense0.75040618
at0.40, with all eight dense-versus-original intervals negative. Earlier raw512
threshold and tile-blend negatives remain in the
[research record](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/experiments/leipzig-hpo/docs/EXPERIMENTS_CURRENT.md).
No further inference, training or TEST tuning is required for this handover.

RAW35 is **deferred, not ready**. A real encoded-resource audit failed on a flat
REFUGE mask request because the historical decoder searches nested folders by
basename. The [isolated typed-selector draft](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/raw35-refuge-typed-selectors-20261112/reports/raw35-refuge-selector-draft-20261112)
(`a34c1ba`) preserves frozen membership and has32 synthetic checks; shared audit
adoption, a new reviewed plan and real re-audit/build were deliberately stopped.
It is not a completed higher-resolution dataset or experiment, and is not needed
by the delivered RAW25 model. Interpolating RAW25 would not provide genuine RAW35
information. Remaining vessel superiority and universal raw MATLAB identity are
**not evidenced**; the accepted handover makes neither claim.

Documentation finalization checked82 local link targets and existing JSON
selection/artifact/default/wheel identities; all passed. This was metadata-only:
no new software test suite, model load, inference, dataset read or training.
The immutable freeze summary's earlier `TEST_evaluated:false` describes its
pre-TEST timestamp; the later completed TEST evidence above is authoritative for
the final result. Historical proof bytes were not rewritten to update that field.
