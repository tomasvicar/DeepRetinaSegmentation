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

All five unmodified production presets now have actual bounded real-data
functional checks: four single-net presets on CPU and full-frame A/V on GPU.
Each uses fresh constructors, one epoch and original patch sizes/full-frame A/V,
then safe checkpoint reload and finite outputs. No bundled checkpoint initializes training or gets replaced.
These short-run weights will not be published as useful trained models.

CPU vessel smoke may run separately with one thread/one core, nice17/idleIO,
CUDA hidden, maximum300s, own-process RSS ceiling2GiB and minimum2GiB available
system memory. Only its owned subprocess may be stopped. The separate A/V GPU
run below followed independent resource review and actual preceding-workload
completion; no concurrent work was added to an occupied GPU.

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

## Actual full-frame A/V CPU attempt: controlled memory defer

The separate exact-production A/V CPU attempt used the original full TRAIN
frame1031×1033, batch1, unchanged constructor and one-epoch recipe. Root first
independently ran all five wrapper/process tests (5passed0.62s) and inspected
the corrected full VAL geometry1033×1040. Actual execution used helper413b654,
SHA bf8be02061b8563c1cb7e020140c8c3a7139ca9f35030db826b413e80aa6bf2c.

The group RSS crossed its3GiB limit at3,186,820KiB after12.86s. The observer
stopped only its owned group, observed child−15 and returned75, with explicit
`own_group_RSS_limit`, no exception and no automatic retry. Only run metadata
exists; there is no completed A/V checkpoint, optimizer step or safe-reload proof
claimed. System available memory recovered above6,100MiB after cleanup. See
`av-train-observed-exit.json` and `av-train-launch.json` for original bytes.

This original CPU attempt remains a controlled resource defer, not successful
CPU training. The CPU cap and batch/geometry were not changed. A later GPU run
used a distinct explicitly approved backend profile, as recorded below. The
resource defer is unrelated to segmentation quality or Meta PBS state.

## Actual full-frame A/V GPU check: training and independent safe reload

On 2026-11-11 the separately authorized GTX1060 6GB run completed one fresh
FP32 CUDA optimizer step, exact unchanged production AV constructor/batch1 and
canonical TRAIN frame 1031×1033, followed by full VAL frame 1033×1040. No resize,
patch, architecture override, historical weight transfer or mixed precision was
used. Frozen source was d028314d6a26c6790d82bdc97219027184c051df, execution
3891751d50628db4458c9ac17a71955736901401cc7364f86ca8b480c1243c59.
The source's exact preset changed only epochs=1/device=cuda:0 at runtime.

Both the training child and its external observer reported original numeric 0;
inner duration was 4.753508s. A separately invoked verifier and its external
observer also reported original numeric 0, inner 2.652429s. Both phases attested
unchanged source/subset bytes. Training loss 0.6903879642 and VAL loss
0.6858412623 are functional diagnostics, not quality improvement evidence.

Independent safe-GPU load checked finite weights/optimizer, all 54 trainable
parameter tensors changed versus seed42 fresh initialization, saved
legacy25_raw_av4 normalization and finite CUDA FP32 logits 1×3×1033×1040.
This new recipe still uses the same-observer GT vessel as the A/V training input;
it does not imply an automatically supplied vessel model at deployment.
Actual coverage was exactly one eligible TRAIN photograph and one AV annotation
(rite_21_training_av), both delivered to the successful update. Full-frame VAL
used rite_27_training_av. This is not all 2,529 TRAIN photos or 5,700 annotations.

The verification-only peak CUDA allocation was 1,404,493,824 bytes. No training
peak was measured; success on this photo/allocation is not a general guarantee
that all full-frame photographs fit 6GB. No TEST was decoded or scored. The
one-step checkpoint ba317014b4370f58345af99b17fbad065df39293cf42f74193cd32fd14925f28
is an evidence identity, not a new useful trained release; no weights are added.

[av-GPU-functional-proof.json](av-GPU-functional-proof.json) preserves scientific
configuration, architecture, source/data identities, full shapes, coverage,
metrics, original numeric exit facts and SHA bindings to original private
artifacts, while omitting operational host paths/PIDs/GPU UUID. The existing
CPU defer originals are unchanged. Together with the four CPU presets, all five
native production presets now have real-data functional evidence, without a
quality or completed full-corpus retraining claim.
