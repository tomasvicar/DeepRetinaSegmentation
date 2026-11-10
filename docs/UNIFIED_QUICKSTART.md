# Train and predict with one RGB network

The pipeline combines vessel segmentation, conditional artery/vein classification
and disc/cup segmentation in one model. It trains only annotated objectives;
missing cup or A/V annotations do not become negative labels. Alternate observers
remain usable. The bundled original networks are ready for inference; new-model
optimization and final quality comparison are still in progress.

Run commands from the repository root after `uv sync --frozen`. All application
steps use Python. Models need no missing large pretraining database. Set the data
root to your ownCloud `fundus_databases` folder; original archives are read only.
Choose new output directories for preparation and training.

## Use the bundled completed baseline

```bash
uv run deepretina predict --model unified-full-data-baseline \
  --input /path/to/fundus.jpg --dataset drive --fov /path/to/fov.png \
  --output predictions/unified-baseline --device cpu
```

Use your photograph's actual dataset/camera geometry; `drive` above is an example.
This single RGB model produces vessel, artery, vein, disc and cup masks,
probabilities, overlay and original-frame metadata. Defaults retain threshold0.5.
It completed60 epochs on all2221 eligible training photos /5281 annotation records
from17 sources, without pretraining; selected checkpoint E56 has training VAL
source-macro Dice0.800123. It has **no held-out TEST results yet**; HPO and measured
comparison continue. [Training CSV and terminal QA](../reports/release/control60-baseline/)
document provenance. Original models and their aliases remain available.

## Prepare the full suitable annotated corpus

```bash
uv run python scripts/prepare_unified_data.py \
  --data-root /path/to/ownCloud/fundus_databases --output data/unified25
uv run python scripts/prepare_unified_data.py \
  --data-root /path/to/ownCloud/fundus_databases --output data/unified25 \
  --regroup-patients --train-only-datasets riga uoadr
uv run python scripts/prepare_unified_data.py \
  --output data/unified25 --audit-existing
uv run python -m deepretina.unified inspect data/unified25/manifest.json
```

The default preparation includes17 suitable supervised sources and preserves the
original physical25-pixel/degree raw geometry, without a modern maximum-size cap.
The metadata regrouping step checks known G1020 patients, ARIA identity prefixes
and shared photograph aliases before training. RIGA and UoA-DR stay train-only
without a verified patient crosswalk. Opaque source IDs are not proof of patient
independence. Read [data rules](UNIFIED_DATA.md) and the
[patient split audit](PATIENT_SPLIT_AUDIT.md). A `--limit` preparation is a marked
preview and is rejected for full-data training. Exact inventories may vary with
archive versions; retain the generated inventory and integrity reports.

## Train, check the first epoch, then resume

```bash
uv run deepretina train-unified --manifest data/unified25/manifest.json \
  --config configs/unified.toml --output runs/unified --stop-after-epoch 1
uv run deepretina train-unified --manifest data/unified25/manifest.json \
  --output runs/unified --resume runs/unified/last.pt
```

The preset is a reproducible starting point, not a selected optimum. It keeps a
60-epoch cosine horizon even when an invocation stops after one epoch. Check
`coverage.json`, `validation.json` and `metrics.csv` before continuing. Every
training photo must contribute a supervised crop in a full epoch, followed by
source/task-balanced extra crops. The8192-sample preset fits the current2221-photo
training corpus; increase `epoch_samples` to at least the number of eligible
photos for a larger corpus. One epoch need not select every alternative observer.

`best.pt` contains the validation-selected model; `last.pt` also retains optimizer,
scheduler and RNG state. Resume in the same run directory with the same source,
config and data. The source/config/content guards reject silent changes. Keep the
code checkout used for training until it finishes; use another checkout for code
updates. `--device cpu` is available; GPU time varies considerably by hardware.
`--max-steps` is a functionality check and is not a complete training budget.

## Validate and predict

```bash
uv run deepretina evaluate --checkpoint runs/unified/best.pt \
  --manifest data/unified25/manifest.json --split val \
  --output runs/unified/validation-selected.json
uv run deepretina predict --checkpoint runs/unified/best.pt \
  --input /path/to/fundus.jpg --dataset drive --fov /path/to/fov.png \
  --output predictions/unified
```

Use the correct dataset/camera geometry; for an unknown source supply
`--field-of-view-degrees`. FOV is optional for unified prediction, with provenance
recording the automatic fallback. A raw photograph or folder is accepted. A single
provided FOV applies to a single photograph. Checkpoint tiling defaults are used;
explicit `--tile-size`/`--overlap` overrides are recorded. Whole-view checkpoints
reject irrelevant tiling overrides. Original-frame outputs include named masks,
probabilities, overlay and metadata. The five probability planes are
Pv,Qartery,Qvein,Pdisc,Pcup; artery/vein probabilities are conditional on vessel.
Decoding gates A/V by predicted vessels and constrains cup inside disc.

Validation verifies the checkpoint's manifest/content pins and preserves existing
report files. Test evaluation for unified checkpoints remains deliberately blocked
pending reviewed frozen final selection. Use validation for optimization; no new
model test results are claimed here. Historical original-model exposure is not
fully recoverable, so comparisons with originals carry that qualification.

## Optional derived cache

The canonical prepared corpus remains authoritative. A derived train+validation
cache can reduce repeated decoding while verifying its own fingerprint and
canonical content pin. Read [DERIVED_CACHE.md](DERIVED_CACHE.md) before building it;
pass `--derived-cache` and `--expected-cache-fingerprint` explicitly at training.
Keep the same cache identity on resume. Cache construction is CPU work; it does
not train a model or establish quality. Full objectives, sampling, augmentations,
partial-label rules and checkpoint contracts are in [UNIFIED.md](UNIFIED.md).
