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
