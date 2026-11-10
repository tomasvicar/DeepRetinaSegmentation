# Original-model versus unified-model comparison

No independent original-versus-new score is established yet. The original
single-net checkpoint indices contain 3,803/646 indexed images, not that many
training images: their `split` values must be inspected. IDs such as
`hrf_na_dr_0000` come from unsorted original filesystem enumeration. A clean
filename or freshly sorted archive listing cannot recover the corresponding
photograph. The original HDF5 `original_filename` attributes would provide the
missing crosswalk. A scoped read-only MetaCentrum home search found none of the
named original HDF5 files within four directory levels on 2026-11-10.

Original optimization also used historical test scores to select settings.
A mapped old `test` image can be unseen by gradient updates while remaining
exposed to model selection. Original A/V split membership is unrecoverable with
the available artifacts; absence from a universal single-net index cannot prove
A/V independence.

## Conservative eligibility audit

`scripts/benchmark_manifest.py` requires explicit canonical `photo_id` fields
and an independently reviewed crosswalk. `source_id` may identify a patient
rather than one photograph. Canonical photo grouping must unite duplicate
observers/tasks, DRIVE/RITE aliases, RIGA/MESSIDOR aliases and exact decoded-image
hashes; patient/paired-eye grouping belongs to the split builder too. The full
manifest's `photos.json` preserves source-path aliases for this review.

Example crosswalk schema (placeholder IDs are not evidence):

```json
{
  "schema_version": 1,
  "mappings": [
    {
      "photo_id": "decoded-image-hash",
      "family": "universal",
      "legacy_image_ids": ["verified_original_index_id"],
      "verified": true,
      "evidence": "Pinned original HDF5 path/hash and original_filename attribute"
    }
  ]
}
```

A photograph proven outside a model's entire source corpus may instead use
`legacy_image_ids: []` and `verified_absent_from_corpus: true`, with explicit
reviewed evidence covering every dataset alias. Empty IDs alone never prove
absence. Unknown identity, invalid legacy IDs, missing family mapping or a
photo spanning splits fails closed. Only clean `test` photographs can count
as eligible; a new training/validation photograph is not an untouched benchmark.

```bash
uv run python scripts/benchmark_manifest.py \
  --manifest data/unified/manifest.json \
  --crosswalk reports/benchmark/reviewed-crosswalk.json \
  --legacy-index universal=../DeepRetinaSegmentation_legacy/reports/legacy/universal_training_index.json \
  --legacy-index av=reports/benchmark/verified-av-index.json \
  --output reports/benchmark/eligibility.json
```

The unavailable A/V index in this example is an explicit prerequisite, not a
file supplied by this repository. Do not run a comparison omitting A/V exposure
and call the resulting five-head benchmark independent. Audit output records
input SHA256 values, per-photo reasons, gradient exposure and historical tuning
exposure separately. The tool reads only metadata; it never opens reference
masks, changes partitions or trains models.

## Evaluation after selection

Freeze architecture, source/checkpoint hashes, preprocessing, manifest and
validation-selected thresholds before opening held-out results. Compare both
pipelines on the same eligible raw photographs with the same valid annotation
support and original-coordinate restoration. Report vessel/disc/cup confusion
counts and Dice separately; report conditional A/V classification on known
reference vessel pixels and end-to-end A/V with missed vessels counted as
errors. Unknown labels contribute no loss or score. Keep per-dataset and
per-photo results and paired uncertainty, rather than only pooled pixels.

Time both pipelines on the same device and inputs, including the original
vessel-to-A/V chain, with identical synchronization/warmup policy. Report model
parameters, peak memory, preprocessing time and prediction time separately.
A smaller model, more training data or longer training is a named ablation;
none by itself establishes a quality improvement. If independent eligibility
cannot be demonstrated, label results as an exposure-qualified reference
comparison and retain that limitation beside every headline score.

## Reproducible latency measurement

`scripts/benchmark_latency.py` measures a complete raw-photograph pipeline,
including CPU preprocessing, inference and restoration, while excluding file IO
and model loading. CUDA is synchronized before and after every measured call;
all samples and Torch peak working allocation are retained. Both paths use the
same photograph and device. The original pipeline is universal single_net plus
Ves_Class; the latter uses the explicitly uncertified raw Python proxy.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python scripts/benchmark_latency.py \
  --input /path/to/raw.png --dataset drive --fov /path/to/official-fov.png \
  --original-universal /path/to/legacy-single-net-universal.pt \
  --original-av /path/to/legacy-av-v9.pt \
  --unified /path/to/unified/best.pt --device cuda \
  --warmup 1 --repeats 3 --output reports/benchmark/latency.json
```

The initial DRIVE21 functionality measurement on GTX1070 used a **two-step,
non-quality** unified checkpoint. Median complete latency was 3.573s for the
original chain versus 0.529s for the compact unified prototype. Parameter counts
are 3,334,734 versus 1,106,597. Torch peak working allocations were 1,882MB versus
97MB (decimal units). Exact hashes, all timing samples, software and scope are in
`reports/benchmark/latency-functionality-drive21-pascal.json`. This establishes
that the measurement script runs and indicates the architectural cost difference;
it does not establish general speed, segmentation quality, or a final trained
model advantage. Repeat on multiple representative photos and selected final
checkpoints with controlled system load before making a final comparison.

## Exposure-qualified original reference scores

`scripts/evaluate_original.py` evaluates the original universal model on exactly
those prepared photos, first sorted annotation variants and valid pixels used by
the unified evaluator. It preserves original strict logit decoding, leaves the
original disc/cup outputs independent, and records per-photo and per-dataset
confusion counts. Raw and normalized profiles can both be assessed **on validation**:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python scripts/evaluate_original.py \
  --manifest data/unified25/manifest.json --split val --device cuda \
  --universal /path/to/legacy-single-net-universal.pt \
  --av-raw-proxy /path/to/legacy-av-v9.pt --data-root /path/to/fundus_databases \
  --output reports/benchmark/original-validation-raw.json
```

Add `--normalized` for the alternative original Python local-contrast profile.
Its FOV comes from the same prepared annotation, retaining the same evaluation
geometry. Omitting `--av-raw-proxy` excludes AV from scoring rather than assigning
fake zeros. With the explicit proxy, raw RGB identity is verified against the
preparation hash before classification. A/V preserves the original binary vessel
support and tie behavior. Undefined empty/full-vessel inputs yield no detections;
end-to-end scores count the misses and conditional scores disclose excluded pixels.
Test evaluation requires a frozen `--selection-record` whose SHA256 is recorded.
This guard does not establish historical independence; unresolved exposure and
preprocessing limitations remain beside the scores.

The first all-source preview evaluated 21 validation photos. It is only a
pipeline diagnostic: `reports/benchmark/original-validation-all17-preview.json`
explicitly has `preview_only=true`, and the absence of an original image crosswalk
prevents an independent improvement claim. Low scores on individual sources
trigger geometry/annotation inspection before full-data evaluation. Dataset
photographs and diagnostic overlays remain outside Git.

The same 21 preview validation photos were also evaluated with the optional
original Python local-contrast profile. It performed substantially worse than
raw input in this diagnostic (pooled vessel Dice0.027 versus0.701; disc0.023
versus0.522). Full reports preserve all channels and counts. This supports
keeping the source-inferred raw default for subsequent validation; it does not
prove which historical HDF5 filename was used. Neither preview result is a final
quality benchmark. Low raw scores on RIM-ONE, Erlangen, UoA-DR and IOSTAR disc are
being checked against source geometry and annotations before interpretation.

The v2 raw preview report adds macro dataset scores and attributes each channel
to its selected annotation's source (for example, RITE A/V on a shared DRIVE
photo). Its pooled confusion counts match the initial raw report exactly.
Earlier reports remain immutable evidence of the original measurement version.
