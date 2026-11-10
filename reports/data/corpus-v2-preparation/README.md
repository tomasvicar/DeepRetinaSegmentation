# Actual proposed corpus v2 conversion

Source snapshot fa1cf03, exact root-reviewed geometry policy2591d846…301e3d.
[Launch](launch.json) pins the actual command/environment/PID and independent
output directory on Ada. CPU2, observed nice10, CUDA hidden. The [bounded supervisor](bounded_launcher.py)
limits its own process group to1800s/4GiB; it cannot restart or kill other jobs.
The code snapshot is read-only. Sources and corpusv1 are read-only inputs.

First actual progress at21:45UTC: process1434222 alive; parent cloning/verification
completed and134 DRIONS annotation variants had been transformed. This is a
progress snapshot, not a successful final audit or training launch. Completion
requires source/parent preservation, full geometry/label audit, schema2 content
fingerprint and frozen3157 generic manifest-reader check. No model forward,
training, validation metric calculation or TEST model evaluation occurs here.

Expected additions are TRAIN DRIONS220, GAMMA100 (99photos), IDRiD54, BIOMISA45,
and official IDRiD TEST27. Corpusv1 rows/images/masks/FOVs and all its memberships
must stay byte-exact. Proposed nominal geometry and BIOMISA role correction remain
explicitly qualified research policies, not measured/certified calibration.


## Completed and independently consumed

Conversion and full mechanical array/integrity audit completed21:51:21UTC, exit0,
401.70seconds, peakRSS1,079,107,584bytes; no watchdog stop. [Terminal](complete/terminal.json).
No process remains running. The proposed combined artifact has7811 variants and
4155photos: TRAIN5700variants/2529photos, VAL993/770, TEST1118/856.
All7365 original rows,3820 original photo records and18853 original pixel-file
paths remain byte/value identical;18866 copied parent files have distinct inodes
from the source parent. The original6103 parent fingerprint was reverified after
construction. No source archive was changed. New826 encoded resources are pinned.
There are0protected-alias conflicts/quarantines in this actual corpus.

Manifest SHA256 `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`.
Schema2 content fingerprint **`00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52`**.
The prepared artifact stays on Ada at the absolute output path in launch.json;
reports contain metadata only, not photographs or target arrays.

The **unchanged frozen3157 consumer**, execution8e0481…cb30b, independently
rehashed schema2 content and opened the TRAIN/VAL manifests under full_data config:
2529/770photos,5700/993annotation variants. Six TRAIN-only input samples cover all
four additions, both GAMMA camera resolutions and BIOMISA's missing-cup case.
Image/target/support are finite and aligned, missing heads invalid and cup nested.
All993 validation rows are exactly the parent rows. [Consumer proof](complete/frozen3157-reader-check.json)
is a data-reader check, not a network forward or training. No model TEST score
was calculated. Full mechanical preparation audit necessarily reads NPZ arrays,
including original heldout records and the27 new IDRiD TEST labels, solely for
schema/geometry/binary/containment checks and inventory—not model decisions.

Final [integrity](complete/integrity.json), [composition](complete/composition.json),
[parent preservation](complete/parent-preservation.json),
[resource hashes](complete/source-resources.json), and
[parent-photo/angle audit](complete/composition-and-parent-qa.json) retain the evidence.
Nominal degree choices remain explicitly qualified, not measured calibration.
Frozen v1 training/HPO/cache4e39 and all models/defaults remain unchanged; v2 needs
a new plan/cache fingerprint before any future training.


After all checks, the proposed artifact's files were set0444 and directories0555.
[Freeze receipt](complete/artifact-freeze.json) records the count/storage view.
This is a read-only research data artifact, not promotion of a trained model.
The freeze changes no bytes, schema fingerprint, parent file modes or current
model/training jobs. Use read-only integrity verification; a new preparation must
write a new directory. Full model inference/checkpoint geometry capture is being
integrated separately; existing models and frozen3157 execution remain unchanged.
