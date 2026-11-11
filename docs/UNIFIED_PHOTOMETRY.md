# Saved unified full-image photometry

Unified checkpoints can save one of three input profiles. Prediction reads the
saved choice automatically; no extra approval or prediction-only profile switch
is needed. All twelve existing bundled checkpoints and aliases remain unchanged.
Old unified checkpoints without `photometric_profile` resolve to `raw`.

| `UnifiedConfig.photometric_profile` | Prepared full-image operation |
| --- | --- |
| `raw` (default) | Existing RGB unchanged; no extra FOV reads or RNG calls |
| `clahe_hsv_v1` | Float32 RGB/255 → HSV, scikit-image CLAHE on V, RGB, clipped/truncated uint8 |
| `legacy_lc_clahe_v1` | Existing Python boundary extension and Gaussian background division, then the same HSV-V CLAHE |

Geometry is still `legacy25_raw` at **25 pixels per degree**. Each profile runs on
the **entire prepared photograph**, after image geometry and before any crop,
padding, augmentation, TTA view or inference tile. TRAIN, VAL and raw-photo
prediction share this input contract. Targets, valid domains, camera/FOV geometry,
normalization `RGB/255−0.5`, joint A/V blending, decoder and thresholds retain their
existing behavior. TTA continues to average inverse-aligned joint vessel/A/V,
disc and cup probabilities in FP32 before recovering conditional A/V.

Both nonraw profiles restore original prepared RGB outside the image-derived or
official FOV exactly. Corpus FOV comes from `photos.json` and its standalone FOV
asset; missing, empty or inconsistent FOV fails. There is no fallback to target
masks, target validity or GT statistics. Prediction uses the FOV from the existing
raw-image preparation path. The historical original-model preprocessing modes are
separate and retain their existing defaults and limitations.

## Retrain with a saved profile

Add one field to an otherwise complete flat unified JSON/TOML configuration:

```toml
photometric_profile = "clahe_hsv_v1"
```

Use `"legacy_lc_clahe_v1"` for the background-corrected profile. The default is
`"raw"`; profile names and physical scale are validated. Continue to choose the
architecture, sampling, patch size, training precision and other training settings
explicitly for the intended experiment. The supported product architectures remain
unchanged; this port does not add the research A/V-residual architecture.

```bash
uv sync --frozen --dev
uv run deepretina train-unified \
  --manifest /data/corpus/manifest.json \
  --config configs/my-photometry.toml --output runs/new-photometry \
  --expected-content-sha256 REVIEWED_CANONICAL_CONTENT_SHA256
```

Training without a processed cache computes a full-image transform on bounded LRU
misses, before cropping. Another worker or an eviction can compute it again; this
is not a once-per-run cache. The image Tensor LRU remains bounded to128MiB per
Dataset/worker. Existing raw-derived caches still contain raw RGB and labels.

An optional separately pinned immutable processed-RGB cache avoids repeated
photometry computation. See [cache construction and admission](UNIFIED_PHOTOMETRY_CACHE.md).
Cache paths are runtime arguments, not saved model configuration; portable cache
identity is preserved across relocation and resume. Dropping or changing a saved
training-cache identity is rejected. New training records its own actual source
identity; old checkpoints and frozen source identities are not retagged.

## Predict from the saved recipe

```bash
uv run deepretina predict --model /models/photometry-best.pt \
  --input /images/fundus.jpg --dataset drive \
  --output predictions/new-photo --device cpu --tta rot4
```

Use the photograph's actual camera/dataset profile or supply the appropriate FOV
and field angle. Inference always applies saved full-image photometry once before
TTA and tiling; it never hides that cost with a training-cache lookup. `--tta none`
is the default. No current bundled checkpoint is claimed to have been trained with
CLAHE or LC+CLAHE by this port. Retagging existing weights does not create such a
model: nonraw loading requires its original complete saved descriptor.

`read_unified_sample(..., config=...)` explicitly opts into the saved profile;
without `config` it retains the previous raw behavior. Low-level
`predict_probabilities` receives already processed RGB01 and does not preprocess
individual views or tiles. `evaluate_unified_model` accepts the same optional
verified cache object as the shared Dataset.

## Exact contract and limits

At25ppd, requested CLAHE kernel is113.63636363636363 pixels, truncated by
scikit-image to113×113; clip limit0.005 and256 histogram bins. The LC Gaussian sigma
is36.36363636363636 pixels, with the original Python boundary/remap and Gaussian
semantics. Output clips to[0,255] and truncates toward zero. Final outside-FOV
restoration is an explicit wrapper. No per-patch or per-dataset histogram fitting
is added.

Checkpoints retain the versioned descriptor, operator parameters, FOV policy,
quantization and NumPy/SciPy/skimage/OpenCV versions. Loading/resume rejects a
missing or changed nonraw recipe/backend, including JSON boolean/integer type
substitution. Use the locked environment. The operator bytes and fingerprint remain
those of the frozen `fd44cdf` authority; existing genuine caches keep their exact
fingerprints. Stricter descriptor/schema admission does not rewrite them.

This is the documented **Python** preprocessing contract. It is not universal
MATLAB parity, cross-version bitwise compatibility, or measured quality improvement.
The historical raw A/V source profiles retain their separately documented
photometric limitations. Nonraw processing can amplify noise or change colors;
training and evaluation must establish its suitability.

The existing primitive latency benchmark recognizes saved nonraw profiles and
reports preprocessing separately from model timing. Its nonraw model-only scope
cannot establish full-pipeline speedup over an original complete pipeline.

[Port verification](../reports/product-unified-photometry) covers synthetic arrays,
cache interoperability, default raw RNG/masks and joint TTA integration. It does
not train or evaluate a real model, decode real GT, or establish real-image parity.

## Actual checkpoint compatibility check

[Two real TRAIN CPU parity checks](../reports/product-photometry-real-train-parity)
matched probabilities and masks bitwise for the existing TPE preview and original
fd44 raw256 checkpoint. This verifies the scoped raw compatibility paths;
nonraw operational model parity and quality are separate measurements.
