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
