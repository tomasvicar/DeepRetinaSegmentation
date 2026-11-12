# Bundled unified prediction defaults

A reviewed bundled unified model may declare an optional `inference_defaults`
object in its registry entry. Its only permitted keys are `tta`, `threshold`,
`tile_size` and `overlap`. These defaults apply to `deepretina predict --model`
and the public `predict_bundled_unified` API, **not** evaluation, selection/TEST,
custom checkpoint paths, original models or `predict-original`.

No current registry entry declares this object. All12 existing models and their
aliases retain `tta=none`, `threshold=0.5` and their existing saved/native tiling.
No model was selected/promoted or registry/default/weight bytes changed here.
The existing TPE preview remains an experimental preview, not a newly chosen final.

For a future reviewed model, the entry could contain (illustrative only):

```json
"inference_defaults": {"tta": "rot4", "threshold": 0.35}
```

Omitted keys retain historical defaults; omitted/None tile/overlap use the saved
checkpoint values. Each explicitly supplied CLI option wins independently,
including `--tta none` and `--threshold 0.5`. A malformed stored declaration fails
before weights load even if a user supplies overriding values. Unknown keys,
invalid modes, nonfinite/bool/out-of-range thresholds, invalid integer tiles and
incompatible tile/overlap pairs fail. Defaults on an original/pilot entry are
rejected. No second named-preset registry or extra user-facing preset flag exists.

```bash
# Current preview: ordinary none/.5, saved tiling (unchanged).
uv run deepretina predict --model unified-v3-tpe-preview \
  --input fundus.png --dataset drive --output predictions/preview --device cpu
# Explicit protocol always wins, including future models with other defaults.
uv run deepretina predict --model unified-v3-tpe-preview \
  --input fundus.png --dataset drive --output predictions/preview-explicit \
  --device cpu --tta none --threshold 0.5 --tile-size 512 --overlap 64
```

Use the actual camera/source/FOV and a new/empty output directory as described in
[PREDICT_ORIGINAL.md](PREDICT_ORIGINAL.md) and the relevant model card. A prediction
records the resolved actual threshold/TTA/tiling in the existing `prediction.json`
fields; this feature adds no new fields to the default-none output schema. The
new registry declaration itself does not certify quality or resource cost. A final
model's advertised measurements must match its reviewed resolved protocol.

## Python API

```python
from deepretina import predict_bundled_unified

files = predict_bundled_unified(
    "unified-v3-tpe-preview", "fundus.png", "predictions/preview-api",
    dataset="drive", device="cpu", tta="none", threshold=0.5,
)
```

This lazy wrapper resolves the same bundled entry/alias defaults, verifies the
bundled checksum using the existing loader, and delegates to unchanged unified
prediction. `fov`, when supplied, is a binary array as in the underlying API.
Input is raw unprocessed RGB; saved preparation/photometry remains automatic.
The return value is the same output-path dictionary as `predict_unified`.

For metadata-only inspection, import
`deepretina.bundled_inference.resolve_prediction_options(model, ...)`; it returns
the four resolved options without opening weights or importing Torch. `None`
means inherit, so pass explicit `"none"`/`0.5` to override corresponding defaults.
Direct checkpoint-level `predict_unified` retains its own historical defaults;
it does not consult model names or this registry. Use the bundled wrapper/CLI
when the named model's reviewed defaults are intended.

The routing is tested with synthetic registry entries and mocked predictors only.
It does not rerun model quality, NN inference, training or TEST. See
[verification](../reports/release/bundled-inference-defaults-20261112/verification.json)
and [the complete test catalog](TESTING.md).
