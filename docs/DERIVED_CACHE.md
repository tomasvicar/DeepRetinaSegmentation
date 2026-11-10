# Optional lossless data cache

This experimental branch adds a derived storage format for future unified training.
It does not change the canonical corpus or activate in existing runs. The default
loader still reads canonical PNG/NPZ files. Existing immutable `87cf045` runs and
plans stay unchanged; using this code requires a reviewed new source revision and
new frozen plan, with training started from scratch after model selection.

RGB derivatives are uint8 HWC NPY files. Labels remain compressed NPZ: exactly
binary targets become bool, while soft targets retain their original array dtype
and values. Negative floating zero is retained as float. Validity and optional FOV
keep their boolean meaning; missing FOV remains missing, and other NPZ arrays are
preserved. Construction compares restored RGB bytes, float32 target bits (the
loader's representation), validity/FOV and auxiliary arrays before publishing a
completion marker. There is no resize, photometric processing or augmentation.

The builder reads one source file at a time and refuses existing nonempty output
or any cache base inside the canonical directory. It checks canonical content
before and after construction. Files are keyed by the original source-file SHA256
so byte-identical annotation/image aliases share derivatives. Canonical hardlinks
and file bytes are never changed.

```bash
nice -n 5 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run python scripts/build_derived_cache.py \
  --manifest data/unified25/manifest.json --cache-base data/derived-cache
```

By default only train and validation are decoded. The canonical schema2 fingerprint
still audits all referenced bytes, including test bytes, without evaluating test
predictions. Output resides under `<base>/<canonical fingerprint>/v1-train-val/`.
The JSON result supplies its independent cache fingerprint. The portable index
contains relative source-to-derived mappings, source and derived file SHA256,
array encoding, file sizes, explicit photograph/split coverage and equality counts.
No absolute runtime path participates in the cache fingerprint.

For a bounded initial experiment use `--splits train --max-photos 64`. Selection
round-robins source datasets with deterministic SHA ordering, includes every selected
photo's annotation variants, and labels partial coverage `subset`. Such a cache
cannot be used for a full training plan. A low-level benchmark may explicitly
construct `VerifiedCache(..., allow_canonical_fallback=True)`; only absent mappings
fall back. Corrupt or missing mapped files always fail. Fallback is disabled by
default, and training/plan creation additionally require every train and validation
image and annotation reference regardless of an index's claimed coverage count.

At opening, `VerifiedCache` requires both externally supplied canonical and cache
fingerprints, rehashes canonical inputs, and verifies every indexed derived file.
Each subsequent derivative read hashes the exact bytes that will be decoded from
memory; file timestamps are never accepted as evidence of identity. A disk bitflip
is detected even when size and mtime are restored. A previously verified image
already held in the loader's in-memory RGB cache does not read the disk again.
This is detection, not an adversarial filesystem security boundary.

Create a future plan explicitly:

```bash
uv run python scripts/unified_hpo.py plan --manifest data/unified25/manifest.json \
  --derived-cache data/derived-cache/CANONICAL_FP/v1-train-val \
  --expected-cache-fingerprint CACHE_FP --output reports/unified/new-plan.json
uv run python scripts/unified_hpo.py run --plan reports/unified/new-plan.json \
  --trial baseline --output runs/new-baseline \
  --derived-cache data/derived-cache/CANONICAL_FP/v1-train-val
```

The plan pins canonical content and cache identity separately. The runtime cache
path is outside `UnifiedConfig`; it may relocate alongside an unchanged portable
corpus. Resume checks identical cache identity, canonical content, executable source
and training configuration. Enabling, disabling or replacing the cache on resume
fails. No automatic cache discovery, old-plan migration or relaxed source guard is
provided. Inference checkpoints contain cache identity as provenance, not a path
needed for prediction.

The public experimental training CLI accepts the same explicit pins:

```bash
uv run deepretina train-unified --manifest data/unified25/manifest.json \
  --purpose full_data --epochs 60 --epoch-samples 8192 \
  --expected-content-sha256 CANONICAL_FP \
  --derived-cache /path/to/CANONICAL_FP/v1-train-val \
  --expected-cache-fingerprint CACHE_FP --output runs/new-cached-run \
  --stop-after-epoch 1
uv run deepretina train-unified --manifest data/unified25/manifest.json \
  --resume runs/new-cached-run/last.pt --output runs/new-cached-run \
  --derived-cache /relocated/path/to/CANONICAL_FP/v1-train-val
```

For full-data model selection, put the reviewed hyperparameters and
`selection_metric = "macro_dataset_mean_dice"` in a flat TOML/JSON `--config`.
On resume, the checkpoint supplies both trusted pins; the relocated runtime cache
path is still explicit. A fresh cached run requires both external pins. Exact
public CLI tests compare canonical training with cached training resumed after
cache relocation, including all saved model weights.

CPU throughput measurements belong in `reports/unified/derived-cache/`; they do
not establish GPU epoch acceleration or model-quality improvement. A complete cache
build followed the 64-photo tensor/RNG/workers gate and capacity review. The
verified complete train/val cache contains 2991 photos and 6274 annotation variants,
uses 8.17GB allocated disk and took 4m34s to build on Ada. Its exact pins and
resource observations are in [HANDOFF_DERIVED_CACHE.md](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/d83f17a83e873f840751493b1e0457b4c386a07d/docs/HANDOFF_DERIVED_CACHE.md).
