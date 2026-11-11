# Immutable profile-specific RGB cache

A processed cache stores full prepared uint8 RGB for `clahe_hsv_v1` or
`legacy_lc_clahe_v1`. It does not change canonical RGB, standalone image FOVs,
labels, splits or the raw-derived cache. The [saved operator contract](UNIFIED_PHOTOMETRY.md)
applies identically with and without this cache.

Build into a new directory outside the canonical corpus. Obtain the externally
reviewed schema2 canonical content hash and descriptor hash first; the descriptor
hash is SHA256 of sorted compact JSON plus one newline.

```bash
uv run python -c 'from deepretina.photometry import contract; from deepretina.photometry_cache import canonical,sha256; print(sha256(canonical(contract("clahe_hsv_v1"))).hexdigest())'
uv run python -m deepretina.photometry_cache \
  --manifest /data/corpus/manifest.json --output /data/cache-clahe \
  --profile clahe_hsv_v1 \
  --expected-content-sha256 REVIEWED_CANONICAL_CONTENT_SHA256 \
  --expected-descriptor-sha256 REVIEWED_DESCRIPTOR_SHA256
```

The normal builder covers all eligible TRAIN/VAL photographs. It verifies encoded
canonical identity before/after, including unchanged partition metadata; it does
not decode GT or process TEST images. Each RGB alias must agree. Construction
checks shape/dtype, encoded identity, exact NPY round trip and outside-FOV raw RGB.
A fresh staging directory, exclusive construction lock, immutable index and
`_SUCCESS` marker prevent incomplete caches from being admitted. Existing outputs
are refused. An explicit TRAIN-only subset proof cannot be used as full coverage.

```bash
uv run deepretina train-unified \
  --manifest /data/corpus/manifest.json --output runs/new-photometry \
  --config configs/my-photometry.toml \
  --expected-content-sha256 REVIEWED_CANONICAL_CONTENT_SHA256 \
  --photometry-cache /data/cache-clahe \
  --expected-photometry-cache-fingerprint REVIEWED_CACHE_FINGERPRINT
```

The config must save the matching nonraw profile. Runtime requires the external
cache fingerprint, canonical identity, exact descriptor/backend/operator and full
TRAIN/VAL inventory. Every RGB read verifies bytes and shape before decode;
corruption or missing coverage fails without fallback. No unbounded RGB-array or
memmap dictionary is retained. `verify_all_bytes()` streams file hashes and NPY
headers without decoding arrays; `require_coverage(rows)` checks the requested
eligible subset. Raw checkpoints cannot consume processed caches.

The checkpoint stores portable cache identity, not the host path. Resume permits
an identical relocated cache and rejects replacing or dropping its identity.
Raw-photo prediction never uses this training cache and applies the saved operator
to the actual prepared image instead. Old genuine `fd44cdf` caches remain readable
without rebuilding, rewriting descriptors or changing their externally pinned
fingerprints when the exact backend contract matches.

Synthetic cross-source verification is in the [port report](../reports/product-unified-photometry).
It is not a full-cache build or measured worker/epoch throughput for this product
branch. Budget storage, CPU, worker memory and full-frame transform cost for the
actual corpus; no loader or model speedup is inferred from input equality.
