# Experimental full-image photometry

This branch adds opt-in preprocessing for a **new** unified training run. Existing
models, frozen source 72 / 0b, live TPE runs and raw canonical/cache data are
unchanged. No model was trained or scored to establish a benefit here. The
implementation is ready for review; measured CPU cost means the nonraw GPU arms
should wait for a reviewed precomputed RGB cache and warm-load admission.

`UnifiedConfig.photometric_profile` is appended after all existing fields and
defaults to `"raw"`. The canonical geometry remains `legacy25_raw` at 25 pixels per
degree. The field changes full-image intensities after geometry and before any
crop, padding, augmentation, fallback or inference tile. Targets, validity,
physical scale, FOV geometry, observer choices and model normalization
`RGB/255 - 0.5` are untouched.

| Profile | Operation inside the prepared FOV |
|---|---|
| `raw` | Exact existing camera RGB; no extra FOV reads or RNG calls |
| `clahe_hsv_v1` | Float32 RGB/255 → HSV; scikit-image CLAHE on V → RGB; clip and truncate to uint8 |
| `legacy_lc_clahe_v1` | Existing Python boundary extension, Gaussian background division, then its HSV-V CLAHE and uint8 export |

For **both** opt-ins, the final RGB outside the prepared FOV is restored from the
original raw prepared RGB exactly. This avoids introducing a background-masking
confound against raw. The local-contrast helper internally extrapolates/zeros as
its original Python implementation does; that internal zeroing is not the final
outside-FOV output. FOV must come from the canonical `photos.json` image-FOV asset,
not target masks, target validity, disc contours or GT statistics. Missing,
empty or mismatched FOV fails instead of inventing a fallback.

## Exact versioned contracts

At 25 pixels/degree, requested CLAHE kernel is `250 / (55 / 25)` =
113.63636363636363 pixels in both axes. Scikit-image truncates to integer **113 ×
113**; values below one after truncation are rejected, not silently clamped.
The clip limit is **0.005** and histogram bins **256**. These are scikit-image
semantics, not OpenCV's differently scaled clip parameter. HSV conversion operates
on float32/255; RGB export clips to `[0,255]` and truncates toward zero to uint8.
No per-patch, per-source or per-dataset histogram or parameter fitting is used.

The background profile directly calls the existing
`legacy.preprocessing.local_contrast_and_clahe`. Its sigma is `80 / (55 / 25)` =
36.36363636363636 pixels, with the original eroded-FOV EDT/Sobel/OpenCV remap,
scikit-image Gaussian (nearest boundary mode, truncate 4, channel axis 2),
`clip((RGB-background)/background+0.5,0,1)`, then the same HSV CLAHE. A zero/invalid
illumination denominator raises a clear error. The function is reused, not an
approximate rewrite. Final outside-FOV restoration is the explicit new wrapper.

The checkpoint/run/inference descriptor records the named version, actual kernel
and rounding, sigma, clip/bins, quantization, FOV policy and NumPy/SciPy/skimage/
OpenCV package versions. Safe loading and resume reject a missing or changed
nonraw descriptor/backend. This is a Python contract, **not MATLAB parity or a
claim of bitwise equivalence across library versions or machines**. Use the
locked `uv` environment. The new module is included in executable training-source
fingerprints, and profile/backend changes cannot silently resume another recipe.

The old `normalized=false` / `legacy25_raw` metadata describes canonical geometry
and image storage. The explicit `photometric_profile` plus its descriptor states
the final network-input intensity contract. Old checkpoints missing the new field
still resolve to raw. Adding a tag to existing weights does not create a trained
photometric model and is rejected without its matching descriptor.

## Shared data and inference path

`UnifiedDataset._image` applies a profile on the **entire prepared image** and
stores its final float32 tensor in the existing bounded **128 MiB** LRU. Opt-in
keys include photo ID/profile/physical scale; no second unbounded image or FOV
array cache is added. The small FOV metadata index is retained, but FOV images
are decoded on actual misses. “Before crop” does not mean once for the whole run:
eviction or another worker can recompute a full image. Fallback uses the same
processed native image; it cannot silently revert to raw.

TRAIN and VAL use this loader. `predict_unified` prepares raw geometry once,
applies the same profile once to full RGB/FOV, then tiles. Low-level
`predict_probabilities` expects **already profile-processed** prepared RGB01; it
must not preprocess each tile again. The independent `read_unified_sample`
remains raw by default and accepts an explicit `config` to opt in.

Schema 2 already hashes `photos.json` and every referenced standalone FOV byte
alongside the RGB, labels and manifest. Full-data training verifies that content
identity before using the dataset. A regression confirms FOV mutation changes
the canonical fingerprint. Raw manifests, images, masks and raw-derived cache
identities are never relabelled or overwritten by this feature.

For a future reviewed arm, add just this field to its otherwise frozen JSON/TOML
config; the existing `uv run deepretina train-unified --config ...` route accepts
it without a new CLI flag:

```toml
photometric_profile = "clahe_hsv_v1"
```

Raw inference reads the saved choice automatically; no prediction-only override
is added. No bundled weights currently claim either new profile.

## Measured scope and next gate

[The actual TRAIN input/performance report](../reports/unified-photometry/actual-TRAIN-sanity-perf.json)
verifies image-only FOV availability/headers for **2,529 TRAIN + 770 VAL** photos,
without decoding GT or TEST data. DRIONS TRAIN image 001's raw-inference geometry,
prepared RGB/FOV and final profile tensors match its canonical dataset inputs
exactly. Its 30° angle is a documented nominal research assumption, not measured
acquisition. The private contact sheet was visually inspected; it is not bundled.

One-core actual TRAIN timings span DRIONS 0.375 MP, G1020 1.018 MP and HRF 1.397 MP.
CLAHE operator medians are about **0.151 / 0.730 / 1.001 seconds**; LC+CLAHE about
**0.483 / 1.786 / 2.356 seconds**. The report includes seconds and MPix/s, real
cold RGB-load/tensor costs for raw and opt-ins, and three actual LRU misses plus
three hits per profile. Cached/uncached tensors match exactly. This tiny bounded
sequence is **not** an epoch cache-hit estimate, GPU benchmark or speedup claim.
CPU priority/architecture and process warm state qualify the timings.

These costs can dominate the available CPU loader budget. The background profile
also visibly amplifies dark noise and changes colors in parts included by the
automatic FOV on the canary. Neither intensity change nor contrast proves better
segmentation or A/V classification. Before GPU arms, review an immutable
precomputed profile-specific uint8 RGB cache, exact cached/uncached parity,
capacity and warm-load throughput. No full corpus photometry cache has been built.
The metadata RGB payload for all 3,299 TRAIN/VAL photos is 8,586,029,301 bytes per
profile before headers/index; this is capacity arithmetic, not a completed build.

The separate [five-arm patch/photometry proposal](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/experiments/patch-photometry-study-plan/reports/patch-photometry-study)
keeps batch 4, 8,192 samples and 2,048 updates per epoch / 60-epoch cosine for
raw patches 256, 384, 512 and both photometric profiles at 384. Nominal contexts
are 10.24°, 15.36°, 20.48°; presented pixel budgets are respectively 4/9, 1 and
16/9 of raw 384. Same updates do not mean same pixels, context, compute or crop
RNG path. Keep inference tile 512 / overlap 64 and threshold 0.5 fixed. Current
TPE patch 384/raw is a reference candidate; neither patch nor photometry was
optimized by its six-parameter search. Any GPU-hour estimate remains provisional
until the precomputed-cache pipeline is admitted and a complete epoch is measured.

The two older raw-photo latency protocols (`benchmark_latency.py` and
`benchmark_strong_pipeline.py`) now explicitly refuse nonraw profiles. They must
not silently benchmark raw inputs for a photometric model or omit transformation
cost. A future cache-aware timing protocol needs separate review; existing raw
model timing results are unaffected.
