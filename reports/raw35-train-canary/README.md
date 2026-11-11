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
native 35 training/inference support or change any running 25 experiment. No full
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

## Actual authorized canary completed

The immutable source actually executed was 9b850d80c6b8c09991388e529e6ebdeb06f2823f,
script SHA 40b5dbb40269895a34940436244b860e36dad6d5e6e18f30e736f2192c99e4d0,
packet SHA c6f77ac363e94abea249113c4d4ee18fb1eaa82f63d7a6333ddb06a671f7050f.
Actual CPU-only launch 14:08:56 UTC; owned child 2283237 exited 0 at 14:09:02 UTC,
observed by a separate parent. Producer took 4.436 s (6.413 s including imports/
launch), kernel peak RSS 888,668,160 B; sampled owned-process-tree peak 845,668,352 B.
The 3 GiB RSS watchdog did not trigger. No RLIMIT_AS was used, no retry occurred.
Seven synthetic tests passed in 1.98 s before the real TRAIN run.

Every source RGB emission and all frozen 25 RGB/target/valid/FOV arrays matched
exactly, including DRIVE/RITE shared geometry and RIM observer variants.
Direct35 output shapes were 1443×1454 (DRIVE/RITE), 465×350 (RIM stereo-left crop),
and 1525×1575 (UoA). The source RIM targets actually contain 0/1 after the preserved
>127 decoder; .37 preservation is synthetic evidence, not invented soft real GT.
UoA cup remains entirely invalid at both resolutions. DRIVE and RITE retain
their distinct binary vessel annotations rather than replacing one with the
other. Positive supports below are valid-intersected source labels, not scores:

| Row |25ppd valid-positive counts |35ppd valid-positive counts |
|---|---:|---:|
| DRIVE37 vessel |106615|209718|
| RITE37 vessel/artery/vein |106398 /41627 /56569|209290 /81789 /111461|
| RIM average disc/cup |2121 /427|4162 /844|
| RIM expert1 disc/cup |2095 /368|4105 /732|
| RIM expert2 disc/cup |2158 /488|4227 /962|
| UoA53 vessel |67163|131932|
| UoA53 disc/cup |36315 /0|71168 /0|

RITE unknown vessel support was 8202 at 25 and 16040 at 35; conditional scoring
support (known exclusive artery/vein labels, excluding background/crossings)
was 98196 and 193250 respectively. Different grid counts are expected; these
native 35 supports must NOT be compared as paired VAL denominators against 25.
The canary evaluates no model and demonstrates no quality improvement.

The public `actual-TRAIN-proof.json` is a metadata projection with metrics and
resource SHA preserved; only its scope prose is clarified and source/observer
provenance added. Original report SHA:
`fad593236278670a937f9dc83d233164bd707c5e1672fa7b390f2a5826ce78a1`.
Original report/launch/exit/observer bytes and copy verification are retained in
the private operations repository. PNG/NPZ arrays remain outside Git. The 17
actual archive resources exactly matched the pre-decode allowlist and historical
ZIP CRC/size inventory; actual encoded resource SHAs are recorded. Historical
CRC metadata is not a historical cryptographic hash, and the equality claim is
explicitly the verified decoded/prepared 25 replay for these seven rows.
No VAL/TEST image or GT array, checkpoint, GPU or NN was accessed.

The report's `native_channels` field records the adapter's target/valid arrays
before separately supplied annotation FOV intersection. For example, RIM's raw
valid array spans the stereo canvas while its separate FOV restricts the left
half. Effective supervision/sampling counts are the `grids[25/35].channels`
values AFTER both canonical and annotation FOV intersections. Exact 25 validity
was compared on that effective grid, not inferred from raw target nonzero pixels.
