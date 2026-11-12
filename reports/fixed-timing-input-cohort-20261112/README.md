# Fixed two-geometry TRAIN timing inputs

The cohort is fixed **before final model selection and final timing**: existing
DRIVE TRAIN21 plus lexicographically first distinct eligible canonical TRAIN HRF
`metadata.source_image`. This is HRF `images/04_dr.JPG`; earlier numerical names
are not TRAIN in the pinned split. No model speed was consulted to choose it.
DRIVE21 has historical timing results; reuse is deliberate, not a claim that both
images are previously unmeasured. These are two deterministic geometry controls,
not a representative random population or full-corpus throughput estimate.

Portable `cohort.json` SHA:
**a8332b549a0a339f472e19df4f22101e80b2ba62760d0248d61fa72d46948d87**.
It is byte-exact to the private immutable operator artifact manifest. Image/FOV
binaries and original archive/source paths are kept outside Git/OwnCloud; no
binaries, GT annotations or private paths are published here.

| Fixed input | Native H×W | Nominal FoV | Original RGB entry | Official FOV entry |
|---|---|---|---|---|
| DRIVE TRAIN21 | 584×565 | 45° | `training/images/21_training.tif` | `training/mask/21_training_mask.gif` |
| First TRAIN HRF | 2336×3504 | **60°** | `images/04_dr.JPG` | `mask/04_dr_mask.tif` |

The source adapters explicitly use those official `mask/` resources as FOV,
separately from vessel `manual1/` annotations. No vessel/disc/cup GT was decoded
or substituted for missing FOV. Both selected official masks exist; otherwise
preparation would fail, not invent an FOV. Raw RGB dimensions and the canonical
shape-prefixed decoded identity agree with original17a TRAIN rows/photo metadata:
DRIVE photo77d1a443…d6925 and HRF9fe26b84…e22a7.

| Encoded artifact | SHA256 |
|---|---|
| DRIVE RGB | `e7af55375ba088ac63db120377eaf94b2405fe80b7874d6e1d236cda4f8c4c83` |
| DRIVE official FOV | `acf23eeef91bd7c59624b9c2bb014703bcb37ef65cc8da2c0d221f3ca43dadc7` |
| HRF RGB | `8fcdff8c714cf94b7daac8aa9b3d818d026944dcee6e6ccd7d978719e9008c20` |
| HRF official FOV | `c167c4c82d6feddb35360ea686ff6d7cc4cc38042f24e4cf41ce170fb2f499a8` |

The DRIVE encoded pins match the existing timing control. Raw RGB identity is
SHA256(little-endian int64 H,W,3 + contiguous RGBuint8 bytes), **not** pixel bytes
alone. No orientation, resize, normalization or image re-encoding was applied to
the extracted files. Actual official-FOV calibration used unchanged pure
`matlab_fov.py`, no Torch: DRIVE center(301,279),radius269,diameter538;
HRF center(1166,1762),radius1632,diameter3264 (MATLAB one-based coordinates).
Nominal60° and canonical prepared1000×1397 geometry for HRF are retained verbatim;
calibrated AV diameter and canonical preparation geometry remain distinct fields,
not a new physical-resolution policy. Intended existing raw profile stays25ppd.

## Actual preparation and verification

`build_cohort.py` reads only canonical JSON metadata, archive central directories
and the **four selected** RGB/FOV payloads. Duplicate short/long transport aliases
have matching ZIPCRC/size metadata; short exact entries were selected. Alternate
aliases, annotation files and TEST/VAL assets were not decoded. ZIPCRC checks and
selected encoded SHA/decoded TRAINidentity were verified. Archive size/mtime/ctime/
inode remained unchanged; whole multi-image archives were **not** hashed and no
whole-archive content identity is claimed. Sources were opened read-only.

Actual extraction/calibration exited0 (CPU1/aff4/nice17/idleIO,120s bound). Torch
was not imported; no model/GPU/NN, benchmark, TEST or GT-array access occurred.
New artifacts/files are readonly555/444/nlink1 outside Git. The private source
snapshot contains exact archive paths/stat observations, selected original TRAIN
rows/photos and the same portable cohort. `actual-preparation-proof.json` records
actual exit0, code/cohort/file SHAs and immutable-copy permissions; calibration
preparation is not model latency evidence.

Three synthetic IO/identity tests passed0.005s, actual numeric test exit0, using
the existing runtime without installs. They check shape-prefix identity, exact
FOV alias selection without reading the annotation entry, and rejection of
missing/conflicting official FOV rather than a GT fallback. Test source/command/
exit/stdout/stderr are retained. Ruff check/format pass.

## Later use after separate model/resource review

Resolve each `artifact_relative_path` against the operator's private artifact
root and check its encoded SHA before copying/staging or decoding. Use BOTH same
RGB/FOV byte pairs in BOTH final pipelines. Bind `--dataset drive`/`hrf` and
`--field-of-view-degrees 45`/**60**, respectively; do not silently apply DRIVE's
angle to HRF. Each selected final model keeps its actual saved configuration/
photometry/TTA/threshold/tiling. Use the [final benchmark recipe](../../docs/FINAL_BENCHMARK_RECIPE.md)
for synchronized phases/warmup/repeats and separate IO-inclusive scope.

Report per-image timings/memory and inputs, not merely whichever image is faster.
Do not replace a cohort member after seeing performance or silently downsize on
OOM. No speedup, quality, memory fit, trained weights, defaults, final model or
execution approval follows from this input preparation.

Repeatable builder (new output only; no benchmark):

```sh
python reports/fixed-timing-input-cohort-20261112/build_cohort.py \
  --manifest "$CANONICAL_MANIFEST" --drive-archive "$DRIVE_ARCHIVE" \
  --hrf-archive "$HRF_ARCHIVE" --output "$NEW_PRIVATE_ARTIFACT_DIRECTORY"
```
