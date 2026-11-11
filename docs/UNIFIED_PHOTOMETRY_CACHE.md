# Optional immutable photometry RGB cache

This experimental cache stores full prepared `uint8` RGB images for
`clahe_hsv_v1` or `legacy_lc_clahe_v1`. Canonical RGB, image-derived FOVs,
labels, partitions and the existing raw derived cache remain unchanged.
No model quality improvement has been established. The default raw path uses
no photometry cache.

The operator, parameters, outside-FOV restoration and backend versions are
specified in [UNIFIED_PHOTOMETRY.md](UNIFIED_PHOTOMETRY.md). Build and runtime
require the exact descriptor and operator source. A different clip/kernel,
physical scale, dependency version or operator implementation fails admission.
This cache currently supports canonical 25 pixels/degree only.

## Build a fresh cache

Use a reviewed source checkout and a fresh output outside the canonical corpus
and OwnCloud. Obtain the canonical content pin from a verified schema-2
`integrity.json`; independently review the descriptor before recording its hash.
The descriptor hash is SHA256 of sorted compact JSON plus a newline:

```bash
uv run python -c 'from deepretina.photometry import contract; from deepretina.photometry_cache import canonical, sha256; print(sha256(canonical(contract("clahe_hsv_v1"))).hexdigest())'
uv run python -m deepretina.photometry_cache \
  --manifest /data/corpus/manifest.json \
  --output /data/photometry-clahe \
  --profile clahe_hsv_v1 \
  --expected-content-sha256 REVIEWED_CANONICAL_SHA256 \
  --expected-descriptor-sha256 REVIEWED_DESCRIPTOR_SHA256
```

Use `legacy_lc_clahe_v1` and its own descriptor/output for the second profile.
A full build covers every eligible TRAIN and VAL photo; no TEST pixels are
processed. Canonical encoded-file integrity is checked before and after the
build, including the fixed TEST bytes as identity only. No label arrays are
decoded by this builder. Every relevant RGB alias must decode to the same photo.
Each output is checked for shape, dtype, exact save/load round trip and unchanged
RGB outside FOV. Input encoded hashes are checked again after processing.

A unique staging directory and exclusive construction lock protect the fresh
leaf. Completion publishes `index.json`, then `_SUCCESS` and a read-only tree;
failed staging is retained for diagnosis and is never admitted as complete.
Existing outputs are refused. The index pins canonical/manifest/photos metadata,
FOV and RGB resources, descriptor, executable source, file shapes and SHA256.
Read-only permissions reduce accidental writes; hashes enforce identity.

`--proof-photo-id ID` may be repeated for an explicit TRAIN-only subset proof.
It produces `coverage=subset_proof`, not a complete cache. Normal training and
validation reject it. The `allow_subset=True` reader option is for bounded
construction proofs only; it never enables canonical fallback.

## Train, validate and relocate

```bash
uv run deepretina train-unified --manifest /data/corpus/manifest.json --output /runs/clahe \
  --config configs/clahe.json \
  --expected-content-sha256 REVIEWED_CANONICAL_SHA256 \
  --photometry-cache /data/photometry-clahe \
  --expected-photometry-cache-fingerprint REVIEWED_CACHE_FINGERPRINT
```

The configuration must explicitly select the matching `photometric_profile`.
Check `uv run deepretina train-unified --help` for the complete argument list.
Cache paths are runtime arguments, not model configuration. Checkpoints and run
metadata record the portable cache identity, not its host path. Resume permits
relocation of an identical cache and rejects changing or dropping its identity.
Fresh validation accepts the same two cache arguments through
`scripts/unified_validation.py`. HPO plans declare the exact portable identity
under `photometry_caches[profile]`; `scripts/unified_hpo.py run
--photometry-cache PATH` verifies it before dispatch.

`VerifiedPhotometryCache(root, manifest, profile, expected_fingerprint=...,
expected_canonical_fingerprint=...)` checks the external index/completion pin,
current descriptor/operator and complete TRAIN/VAL inventory. Its
`verify_all_bytes()` streams every processed file hash and validates NPY headers
without decoding image arrays. `require_coverage(rows)` verifies an eligible
subset of that inventory. `rgb(photo_id)` verifies file size and SHA256 on every
read before decoding; corruption never falls back to canonical RGB.

The reader retains metadata only: no open memmaps or unbounded array dictionary.
A transient `uint8` array is copied into the existing float Tensor LRU, which
remains limited to 128 MiB per Dataset/worker. Transient encoded bytes and RGB
arrays are released after conversion. The cache does not make worker RSS or
storage reads free; actual worker throughput, RAM and GPU idle time still need
a complete-epoch measurement. Eight workers require an explicit resource budget.

Raw-photo inference always performs the same full-image transform, never a
training cache lookup. `benchmark_strong_pipeline.py` includes that transform
once inside every measured complete operation. For a nonraw model,
`benchmark_latency.py` measures preparation once separately, then times
model/restoration/decoding with saved tile settings; these timings cannot be
compared as equal full-pipeline scopes. Neither benchmark silently feeds raw RGB
to a photometric checkpoint.

## Bounded evidence and capacity

[The actual three-TRAIN proof](../reports/unified-photometry-cache/actual-3-TRAIN-proof.json)
uses DRIONS, G1020 and HRF prepared images at 0.375, 1.018 and 1.397 MP.
Both profiles reproduce uncached Dataset tensors exactly across five reads per
photo. These are RGB-input checks, not NN/quality measurements or a full build.
The small synthetic suite additionally checks target/crop/RNG identity,
checkpoint resume and relocation, mismatched pins/backend/physical scale,
corruption, alias differences, and bounded array lifetime.

All 3,299 TRAIN/VAL photos require 8,586,029,301 RGB payload bytes per profile,
plus about 422 kB of NPY headers and an index of several MB. Reserve at least
22 GB for both completed profiles and construction/metadata margin; failed
staging may require extra space. Current single-core operator measurements
suggest roughly 34 CPU-minutes for CLAHE and 80–84 for LC+CLAHE before
hashing/decoding/I/O, with at least 25–50% planning margin. These estimates are
hardware dependent and are not measured full-build times. No whole-corpus
photometry cache has been built or admitted yet. Whole-epoch/GPU-hour estimates
remain pending a real warm-cache training measurement.
