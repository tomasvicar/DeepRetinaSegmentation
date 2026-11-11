# Three TRAIN canaries before any full RAW35 implementation

This experimental helper is restricted to three frozen TRAIN photos and seven
annotation variants: DRIVE37/RITE37, RIM-ONE N-86-R average/expert1/expert2,
and UoA53 vessel/disc. The exact 17-resource pre-decode allowlist is committed
in `predecode-plan.json`. No whole adapter may decode other images to find a
selected ID: names/find/read/bytes are filtered before invoking each adapter.
Archive central-directory/CRC metadata may be inspected without pixel decoding.

The helper independently hashes adapter RGB for EVERY emission, including
repeated observers and cross-source aliases, before UoA watermark suppression
and before geometry. Resource-to-photo memoization cannot hide changed RGB.
Canonical owner/order is frozen: DRIVE with its official FOV owns the shared
DRIVE/RITE geometry; a noncanonical first alias is rejected rather than defining
a new FOV. This tiny ordered canary is not a general enumeration-independent
full-corpus scheduler. RIM retains the entire native 1424×2144 stereo frame as
scale anchor, with inclusive left crop [0,1423,0,1071]. UoA53 is TRAIN-only under
the unchanged unavailable-patient-crosswalk policy.

Before any 35 conversion is accepted, native RAW is replayed at 25 with exact
frozen RGB, target, valid and image-derived/provided FOV array equality for all
seven rows, plus owner/crop/scale/angle/source/group/annotator/resource binding.
Native target float32 values are projected with nearest interpolation without
binarization or quantization. The existing RIM decoder itself uses source_binary
>127; this is preserved, not reinterpreted as new soft average GT. A synthetic
.37 test protects projection of genuine soft values, while actual RIM values
are reported from the actual decoded targets. A missing UoA cup remains invalid.

Only then is 35 prepared directly from the same native RGB, original target and
original FOV policy, never from 25 RGB/labels/cache. Actual native and prepared
valid pixels, valid-positive pixels, positive mass and A/V unknown/known supports
are reported per channel. No new split computation takes place. Frozen
photo/source/group IDs establish known-ID integrity, not biological independence
of unknown opaque patient identities.

`regrid35_to25` is a separate synthetic-tested primitive: require equal original
frame/crop/angle, interpolate joint A/V probabilities once to exact 25 shape,
recover conditional probabilities and apply the fixed 25 FOV. It never resizes
GT. The actual canary runs no NN/probability scoring and reads no VAL/TEST pixels;
future paired evaluation must retain all 770 VAL photos/993 records/612 known
clusters and exact common 25 supports under a new reviewed producer lineage.

Current unified CLI/trainer/cache remain strict 25. This helper does not add
native35 training/inference support or change any running 25 experiment. No full
corpus, GPU, training, TEST scoring or quality improvement is claimed.

CPU scope: Ada core 14 only, nice 19/idle IO, existing Python 3.12.13 runtime,
pinned NumPy/Pillow/SciPy/skimage/OpenCV matching frozen 25 preparation. Owned
180 s timeout plus 15 s kill grace; actual peak RSS measured/limited by the external
owned observer, not RLIMIT_AS (Torch library mappings need larger virtual space).
Generated PNG/NPZ outputs remain outside Git and OwnCloud; only code, pins,
metadata and sanitized proof are committed. Synthetic tests cover pre-enumeration
resource filtering, per-emission RGB checks, first-owner rejection, soft target/
missing-channel preservation and joint A/V regridding.

Documentation correction to the earlier 21-source list: STARE contributes 12 TRAIN
photos/24 TRAIN vessel records. No existing corpus is changed by that correction.
