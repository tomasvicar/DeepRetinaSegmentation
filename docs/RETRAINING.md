# Retraining scope and a colleague's starting point

Use [UNIFIED_QUICKSTART.md](UNIFIED_QUICKSTART.md) for the complete RAW → 21-source
manifest → first epoch → resume workflow. Only the repository and shared RAW
archives are required. `uv sync --frozen` supplies the Python dependencies;
Linux preparation currently also uses GNU `cp` for independent CoW-capable copies.
Neither MATLAB nor the unavailable large pretraining database is used.

| Route | Trainable now | What it reproduces |
| --- | --- | --- |
| `deepretina train-unified`, `configs/unified.toml` | All five outputs, partial labels, all suitable training sources | New compact RGB network and explicit modern training protocol |
| Same command, `configs/unified_legacy_topology.toml` | All five outputs from scratch | Original universal backbone topology, expanded output layer, modern objective and padding |
| `deepretina train --task ...` | Separate task models | Earlier small modern models; not the recommended all-source five-head route |
| `deepretina train-legacy --task vessel/disc/universal` | Fresh one-/three-output native models | Exact original constructor and native tiling, explicit new canonical raw training recipe |
| `deepretina train-legacy --task av` | Fresh four-input, three-logit native AV model | Exact original constructor/internal normalization, explicit canonical raw AV recipe |

Native training uses freshly initialized original architectures with a new explicit
training recipe. It preserves the shipped original weights and their historical
preprocessing paths. The AV recipe uses full canonical RGB plus its paired vessel
support; it does not relabel canonical crops as historical MATLAB/DICOM inputs.
See [native retraining and prediction](NATIVE_LEGACY_RETRAINING.md) for all four
presets, coverage, partial labels and supplied-vessel inference.

```bash
uv run deepretina train-legacy --manifest data/rebuilt-training21/manifest.json \
  --config configs/native-legacy-vessel.toml --device cuda:0 \
  --output runs/native-vessel
```

For a fresh original-topology experiment:

```bash
uv run deepretina train-unified \
  --manifest data/rebuilt-training21/manifest.json \
  --config configs/unified_legacy_topology.toml \
  --output runs/universal-topology-scratch --stop-after-epoch 1
uv run deepretina train-unified \
  --manifest data/rebuilt-training21/manifest.json \
  --output runs/universal-topology-scratch \
  --resume runs/universal-topology-scratch/last.pt
```

The first command retains a 60-epoch scheduler and stops after epoch one for
inspection. Resume retains checkpoint configuration. The preset uses no old
weights. This topology has not been shown to improve quality over the compact
model. Do not combine an existing run with changed source, data, or configuration;
start a new output directory instead. CPU works for a functionality check; a full
all-source run requires substantial compute. Do not copy timing expectations
from a different machine or worker configuration.
