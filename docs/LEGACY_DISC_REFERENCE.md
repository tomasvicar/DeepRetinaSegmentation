# Strong original optic-disc reference — completed validation

The dedicated original networks outperform the universal disc head on the common
validation support. The predeclared macro-dataset metric selects **disc0/raw**;
**disc1/raw** has the highest pooled disc Dice and is retained as an alternative.
Neither normalization profile nor checkpoint is selected using test data.

| Original variant | Macro-dataset disc Dice | Pooled disc Dice |
| --- | ---: | ---: |
| Universal/raw | 0.579029 | 0.807155 |
| Universal/Python-normalized | 0.010423 | 0.009490 |
| **Disc0/raw — primary** | **0.729530** | 0.894808 |
| Disc0/Python-normalized | 0.261445 | 0.235025 |
| **Disc1/raw — pooled winner** | 0.727019 | **0.918482** |
| Disc1/Python-normalized | 0.092065 | 0.087074 |

Macro scores average pooled Dice over the seven datasets with disc support;
pooled Dice weights all valid pixels together. Different rankings are reported
openly. The primary metric was documented before full evaluation completed.
The primary checkpoint is packaged as `legacy-disc-0` (SHA
`7d6ddc3bf144e970cc7f5e011e834e241b8d3762464aeacf08d6a87287cdc51f`);
the pooled alternative is `legacy-disc-1` (SHA
`8ebf60971813bbaecc9602663034d75cfe7f5c3c9587b7b2a27d056c1cae3605`).

## Same annotation support and original decoding

This isolated branch starts at `0d6d9087a92f0f07cf75fd1083c590e8af6bb18a`.
`scripts/evaluate_legacy_disc.py` reuses `UnifiedDataset`'s deterministic first
sorted primary annotation, FOV-restricted support and annotation-source mapping
from `evaluate_original.py`. It skips photos without disc GT before forward.
There are691 supported photos out of770 validation photos,583880677 valid disc
pixels and13478446 positive GT pixels. All six variants have exactly matching
photo IDs, record lists, source attribution and per-photo valid/positive counts.
[Complete support audit](../reports/disc-reference/comparison.json).

Tiling uses each original checkpoint's crop and border. Decode strict
`logit > 0` (zero is background), without modern containment or hierarchy.
Only the original disc head is scored. Raw input is canonical legacy25 RGB/255
minus0.5. The alternative normalization uses the same original Python
`local_contrast_and_clahe` and first-row FOV as the existing evaluator, including
its uint8 roundtrip and25pixels/degree. This is not a MATLAB normalization claim.
The empirical raw preference does not recover missing historical provenance.

All dedicated weights and selected logits were finite. The complete four-variant
CPU run finished exit0 in3950.638seconds on Ada, nice5, Torch4/interop1 and OpenCV4
(OMP/MKL/OPENBLAS4), with CUDA disabled. Exact source/helper/checkpoint/config,
command and data hashes are in the native report. The script's `source_commit`
names the pinned core baseline; `script_sha256` identifies the additive helper.
Three focused evaluator/gating tests and the actual three-photo universal/raw
canary passed, including exact TP/FP/FN/support/source agreement after the final
thread cap. No A/V or vessel forward was run by this head-only comparison.

## Strong five-head original reference

`scripts/compose_original_disc_reference.py` copies the calibrated A/V baseline
and replaces **only** disc counts on the691 supported photographs. All770 photo
IDs, GT support/source/record lists and every vessel/artery/vein/cup per-photo
value are preserved. Pooled/per-dataset/macro headers are reaggregated from
counts. Non-disc aggregate headers and calibrated A/V accuracies/profile remain
exactly equal too. No inference, hierarchy, containment or test evaluation occurs.

| Reference | Five-head macro-dataset Dice |
| --- | ---: |
| Calibrated A/V + weak universal disc | 0.624398124 |
| **Calibrated A/V + disc0/raw — primary** | **0.654498285** |
| Calibrated A/V + disc1/raw — pooled-disc alternative | 0.653996118 |

Use the stronger primary reference when comparing a new unified model; do not
claim improved disc performance only against the weak universal head. Both
references retain original cup predictions even where independently predicted
disc masks differ. [Compact comparison](../reports/disc-reference/original-calibrated-disc-comparison.md).

The two complete reference JSON files record actual disc checkpoint SHA,
calibrated-baseline SHA, native report/helper/runtime proof and composition
helper SHA. Six focused tests passed, including source/positive-GT rejection,
unsupported-photo preservation and no added cup containment. Ruff passed without
using its filesystem cache. [Composition proof](../reports/disc-reference/composition-verification.json).

```bash
uv run python scripts/compose_original_disc_reference.py \
  --baseline reports/disc-reference/calibrated-av-baseline-val.json \
  --native-disc reports/disc-reference/dedicated-disc-val.json \
  --variant disc0/raw --output /tmp/original-disc0-reference.json
```

Use `disc1/raw` and a fresh output for the pooled alternative. Output refuses to
overwrite earlier evidence. Full native forward reproduction accepts repeated
`--checkpoint disc0=PATH --checkpoint disc1=PATH --profiles raw normalized`
and the canonical manifest; it is validation-only and CPU-limited.

## Qualifications

Original photograph-to-training-index mappings remain unavailable. These old
models may have historical exposure to current validation photographs, and the
historical training preprocessing profile remains uncertain. They are therefore
**exposure-qualified original references**, not independent publication/test
scores. Calibrated A/V qualifications are preserved from the source report.
Selecting a stronger original baseline is not evidence of a new model's
improvement. Freeze settings before any future held-out evaluation.
