# HPO saved photometry preparation

This branch ports the reviewed product input/cache implementation from
`1fb4a0c304578c092f4e0c03c4684e4bb104fd70`, whose operator contract comes from genuine
`fd44cdfa83d1df9ccc42960df6d1d2018db29900`. It adds field-compatible nonraw loading
and the shared prepared-image data path. It does not add a curated completion
family, freeze a source/model, authorize evaluation, promote a model or open TEST.

`UnifiedConfig.photometric_profile` is appended with default `"raw"`:

- `raw`: existing RGB unchanged, with no new FOV reads or RNG calls.
- `clahe_hsv_v1`: float32 RGB/255 → HSV-V scikit-image CLAHE → clipped/truncated uint8 RGB.
- `legacy_lc_clahe_v1`: the existing Python boundary/background correction plus that CLAHE.

The canonical geometry remains raw25ppd. Nonraw transforms run on the entire
prepared photograph before crops, padding, augmentation, TTA views and inference
tiles. Outside-FOV RGB is restored exactly. Corpus FOV comes from `photos.json`
and a standalone image-FOV asset, never targets/GT validity; missing/mismatched FOV
fails. Raw-photo prediction shares the same operator after geometry. The HPO's
current joint A/V TTA math and strict descriptors remain intact.

Old raw checkpoints without the field remain raw. Loading/resume requires the
saved complete nonraw descriptor and exact backend/operator parameters; fields are
not stripped or silently converted. JSON booleans are not integer versions. This
is the existing Python contract, not universal MATLAB parity or cross-library
bitwise compatibility. Use the frozen dependency environment.

## Training and immutable RGB cache

Add the profile to a complete flat unified JSON/TOML config:

```toml
photometric_profile = "legacy_lc_clahe_v1"
```

The existing train route accepts optional `--photometry-cache PATH` and
`--expected-photometry-cache-fingerprint SHA256`, together with the externally
pinned canonical content hash. `python -m deepretina.photometry_cache --help`
describes fresh full TRAIN/VAL cache construction. Existing genuine fd44 caches
retain their original fingerprints, builder/source identity and immutable bytes.
Processed RGB never replaces the raw-derived label cache. Reads verify bytes,
shape, descriptor/operator/backend and coverage, with no raw fallback. Portable
cache identity, not the host path, is saved and checked during resume/relocation.

## Fresh validation protocol

`scripts/unified_validation.py` retains all four existing TTA modes and adds the
same two optional processed-cache arguments. The cache's portable identity is
recorded in evaluation runtime; no host path is put into evidence.

For nonraw input, the fresh inference protocol and metrics both record the full
photometry descriptor. Their canonical JSON types are bound, including NONE-mode
protocol headers, while the existing strict TTA descriptor remains independent.
Changing only a header boolean to an integer cannot preserve admission. Historical
raw configuration identity and NONE protocol/metric schema remain unchanged when
the original saved field is absent; explicitly saved fields are always retained.

**Format compatibility is not completion eligibility.** The existing TTA producer
still requires source72 training identity, and create/check still requires the
original typed `standalone_0b`, `standalone_72`, `pool_6b` or `pool_72_v3` proof.
No fd44/curated family or unsigned fallback is added. Nonraw curated models need
separately reviewed provenance/admission before a future selection route. New
source/runtime identity requires actual fresh evidence; prior reports are not
migrated or retagged. No held-out stage was executed here.

The existing benchmark callers also respect saved nonraw photometry: full-pipeline
operations include it once, while the older primitive tool explicitly reports its
separate preprocessing/model-only scope. No benchmark or speedup measurement was
performed by this port.

[Verification](../reports/hpo-unified-photometry) uses generated RGB/labels, mocked
TTA/selection evidence and the existing permitted tiny CPU train/load fixtures.
There is no actual checkpoint/GT/GPU evaluation or real-corpus cache build on this
branch, and no real LC checkpoint parity is claimed by these tests.
