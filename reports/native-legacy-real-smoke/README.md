# Native production-architecture real-data functional smoke

This separate experiment addresses a verification gap: existing native trainer
tests execute small synthetic models, while exact production defaults were
checked primarily by constructor metadata/state shapes. No quality improvement
or completed full-data retraining can be established by this small smoke.

`scripts/prepare_native_real_smoke.py` copies complete original prepared photographs
and annotation bytes from four canonical TRAIN/VAL photos: DRIVE/RITE21 and27,
DRISHTI002 and015. All six records and original photo metadata retain their
original splits. There is no resize/crop/relabel or TEST input. Parent manifest
17a2277c is pinned; every copied input is byte-checked against its unchanged source.
The new four-photo manifest fccc6df3 and content fingerprint560fa031 are distinct
from the full canonical corpus and are never presented as all-data coverage.
`subset-proof.json` records full digests and actual file inventory.

Planned validation uses all five unmodified production presets, fresh constructors,
one epoch, original patch sizes and full-frame A/V, then safe checkpoint reload
and finite outputs. No bundled checkpoint initializes training or gets replaced.
These short-run weights will not be published as useful trained models.

CPU vessel smoke may run separately with one thread/one core, nice17/idleIO,
CUDA hidden, maximum300s, own-process RSS ceiling2GiB and minimum2GiB available
system memory. Only its owned subprocess may be stopped. Larger A/V/full-frame
checks await resource review; no work is added to occupied GPUs. Actual runtime
and completion records must be added before claiming execution or success.

## Actual four single-net CPU checks

All four completed with independently observed child exit0, one CPU core/thread,
no resource stop and unchanged subset content fingerprint560fa031. The actual
four-photo corpus has two TRAIN and two VAL photos; eligibility is task-specific.
Vessel uses one TRAIN photo/two annotation records, disc each one photo/one record,
and universal two TRAIN photos/three records. Do not confuse these with all2529
TRAIN photographs or all5700 annotations in the full corpus.

| Preset | Actual optimizer steps | Wall seconds | Peak RSS KiB |
| --- | ---: | ---: | ---: |
| vessel320 | 2 | 42.56 | 1,318,536 |
| disc192 | 1 | 12.02 | 734,680 |
| disc256 | 1 | 13.02 | 965,620 |
| universal288 | 3 | 44.06 | 1,199,484 |

The real trainer performed complete full-frame VAL through its native tiler,
wrote safe checkpoints and retained finite model/optimizer state. Independent
reload then checked a64×64 real TRAIN crop and all74 trainable parameter tensors
changed versus the same seed42 fresh constructor for each preset. CUDA remained
uninitialized. The checkpoint's actual coverage includes every eligible record
of this small subset. Original architectures, patch sizes and weights in the
model registry were not modified; no original model initialized these runs.

`root-four-native-verification.json` contains full checkpoint digests, actual
numeric exit receipts, architectures, channels and coverage. Per-preset launch
files retain exact commands. The initial vessel launch's TRAIN/VAL counts describe
the **subset**, not the task-eligible training photo count above.

The committed preparation/CPU runner/verification scripts reproduce the bounded
workflow. The vessel launch was initially run by the equivalent inline bounded
watcher; subsequent cases use the stored runner. An initial independent verifier
used the wrong optimizer field name and failed before issuing a success record;
the committed verifier uses the actual `training_state.optimizer` schema and
completed successfully. No training rerun was needed.

The full-frame native A/V check is still pending; this report does not certify it.
