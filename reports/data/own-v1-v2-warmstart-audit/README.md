# Own V1 → V2 initialization audit

**PASS with qualifications for a separate development warmstart experiment.**
The own V1 checkpoint's actual training photographs and labels are contained in
V2 TRAIN, its prepared assets match, and the base architecture is compatible.
**It is not an initialization unseen by V2 VAL:** the parent was selected using
that same validation set. No new training/forward/scoring or trainer change was
performed, and no speed or quality gain is established.

## Actual parent, not an inferred teacher

The audited bundled `unified-full-data-baseline` is immutable SHA256
`bfe957460dfed168f2344b1b84ef99b7f502e9dbc8ce1fd4d68632c3234708b1`, selected E56
from the completed own V1 run. Its recorded initialization is scratch,
`historical_exposure=false`, `legacy_initialization=null`; it is not an original
historical network or external pretraining database. Source `3157bb9` and training-source
`dd69012a` identities are retained in `parent-checkpoint.json`.

The checkpoint itself records positive successful-update coverage for all
2,221 TRAIN photo IDs and 5,281 annotation IDs: minimum 64 photo visits and
2 annotation visits at this selected snapshot. These are E56 counters, not the
later terminal E60 coverage copied backward. The actual ID sets were checked,
not reconstructed from source counts or a filename convention.

All 128 state-dictionary tensors match the names, shapes and dtypes of the
current compact U-Net with base channels 12, depth 4, BatchNorm and five outputs. Floating parameters
and buffers are finite. The parameter count is 1,106,597. Shape comparison used a
meta-device model; no existing parent/child/run parameter was modified. No loaded
optimizer or RNG state was applied to another process. This result does not cover
width/depth/normalization/AV-residual changes or a native original architecture.
The saved RGB/255−0.5 and legacy25_raw / 25 pixels per degree contract must be retained.

## Direct containment and encoded-asset identity

The canonical artifacts were independently rehashed from actual files:

- V1 manifest `529edba7` / content `61032be2`.
- V2 manifest `17a2277c` / content `00d9ef19`.

Full hashes are in the JSON reports. The copied V1 manifest inside V2 matches the
actual V1 manifest bytes. All 7,365 complete parent rows are unchanged in V2.
The V2 parent-preservation proof was checked against these actual identities,
including its parent/new content fingerprints; a producer commit is not a data
fingerprint and was not treated as one.

For the checkpoint's actual training IDs, all 2,221 parent photo objects and all
5,281 annotation rows remain in V2 TRAIN. Their 11,972 referenced prepared
image/mask/FOV paths were directly byte-hashed on both artifacts and agree. A
relative-path/SHA pair digest is retained without publishing GT arrays or private
machine paths. Recorded decoded-original RGB photo identities are unchanged;
RAW images were not independently decoded during this audit.

There are zero parent-TRAIN overlaps with child VAL/TEST by photo ID, source ID
or connected split-group ID. The check covers 1,782 source groups and 1,754
connected TRAIN groups; all V2 metadata groups stay in one partition. Known
paired/alias identities are preserved. Opaque IDs do not prove global patient
independence across collections or identify every different photograph of one
person. Existing train-only policies for unresolved RIGA/UoA identities remain
important. This is a known-metadata containment proof, not a universal patient
identity certification.

Encoded fingerprints read file bytes, including heldout bytes for corpus identity;
no RGB or GT arrays were decoded, no masks scored and no TEST inference occurred.
The full fingerprint plus TRAIN asset checks took 7.94 s on own Ada, one numerical
thread, nice15/idle I/O, CUDA hidden. Source inode/size/mtime identity was checked
unchanged across the direct TRAIN asset comparison. Reports contain metadata only.

## Validation and TEST qualifications

The parent's saved validation evidence contains exactly the same 770 photo IDs
as V2 VAL. E56 was chosen using that validation score, and earlier model choices
also used validation. The source `3157bb9` evaluator calls the inference-mode predictor
with `model.eval()`, so these validation passes do not update trainable parameters
or BatchNorm running statistics; the training loop optimizes TRAIN only. They
still create **selection exposure**, which must be inherited by a descendant.
Do not label child VAL as an untouched independent validation of this parent.

Retained parent completion provenance reports no TEST evaluation, and no recorded
parent training photo/annotation is a child TEST member. The child's additional
IDRiD TEST partition is not in parent TRAIN. This audit cannot cryptographically
exclude unlogged external actions; it verifies the retained checkpoint, source,
corpus metadata and encoded assets. Future TEST remains unavailable for choosing
initialization, LR, duration or sampling parameters.

## Minimal proposed experiment — no implementation or launch

1. Use a new source/recipe/run identity with explicit `own_v1_warmstart` lineage.
   Pin the parent SHA/epoch, source, training data and coverage-ID-set hashes;
   retain shared-VAL selection exposure and the absence of a historical teacher.
   Do not call this child initialization scratch or use the historical
   `legacy_initialization` route for this unrelated five-head compact parent.
2. Strictly load only the validated complete model state, including declared
   BatchNorm buffers, into the same architecture. Fresh Adam optimizer and scheduler,
   seeds/RNG streams, sampler and contribution counters: no resume/migration of
   the old run and no parent visit counts credited to child coverage.
3. Declare a full new child budget and fixed schedule before running, with all
   suitable V2 TRAIN data, actual 2,529-photo/5,700-record contribution gates and
   unchanged holdouts. Keep source/profile/camera policy explicit for new sources.
4. Compare to a newly declared from-scratch control with matched child budget and
   source/sampling/loss choices. Report parent and child cost separately: the
   parent run consumed 60 epochs although the copied snapshot was selected at
   E56. A fresh 60-epoch child would therefore consume 120 allocated parent+child
   epochs overall, while its initial weights reflect 56 parent epochs. Any child
   convergence advantage is not a free compute-matched speedup.
5. Inspect validation per head/source, vessel and conditional AV tradeoffs; no
   winner, better accuracy or acceleration follows from compatibility alone.
   Freeze the complete final protocol before any heldout TEST.

No optimizer, scheduler, initialization API, completion gate or production source
was changed by this audit. Checkpoints, data and running jobs remain immutable.
