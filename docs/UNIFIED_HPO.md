# Full-data unified model search

This is the experimental branch, separate from the original models shipped on
`for_Leipzig`. No full-data unified quality improvement has been established yet.
The three single-dataset pilots and their old LR searches are preliminary only.

Prepare all 17 supervised sources with `scripts/prepare_unified_data.py`, then
run its `--audit-existing` integrity check. A completed preparation marker,
matching manifest hash, complete source inventory and positive/valid train and
validation support for all five outputs are required before planning a search.
See [data and geometry](UNIFIED_DATA.md) and [the shared model](UNIFIED.md).

```bash
uv run python scripts/unified_hpo.py plan \
  --manifest /path/to/unified25/manifest.json \
  --output /path/to/search/plan.json --screen-epochs 8 --total-epochs 60
uv run python scripts/unified_hpo.py run \
  --plan /path/to/search/plan.json --trial baseline \
  --output /path/to/runs/baseline --device cuda --stop-after-epoch 1
```

The first full epoch provides an actual time/memory/coverage check. Resume that
same run to the eight-epoch screen, then resume shortlisted configurations to
60 epochs without changing their cosine scheduler horizon:

```bash
uv run python scripts/unified_hpo.py run \
  --plan /path/to/search/plan.json --trial baseline \
  --output /path/to/runs/baseline --device cuda --resume
uv run python scripts/unified_hpo.py run \
  --plan /path/to/search/plan.json --trial baseline \
  --output /path/to/runs/baseline --device cuda --resume --final
```

Each call runs one selected trial; there is no automatic scheduler submission.
Candidate names are `baseline`, `lr_0003`, `lr_003`, `augmentation_strong`,
`width_8`, `width_16` and `legacy_topology_scratch`. The baseline uses RGB legacy25 geometry, width 12,
patch size 384, AdamW learning rate 0.001, mild augmentation and a 60-epoch cosine schedule.
Each epoch visits every training photo once, followed by balanced extras to
8192 samples or the larger training-photo count. Actual successful-update photo
and contributing-annotator coverage are saved, excluding worker prefetch.

Selection uses **macro dataset Dice**: average supported dataset Dice per output,
then average the five outputs. Large FOVs and large datasets therefore do not
win solely by supplying more pixels. Pooled Dice remains reported separately.
Per-head/per-dataset regressions must be assessed alongside this selection score;
a higher average alone does not establish improvement in every task.
The original-topology candidate is trained from scratch. Historical teacher
initialization is excluded from this independent search, even with its separate
exploratory opt-in flag.

The plan freezes execution-source, manifest and prepared-content fingerprints,
all configurations and the original Git revision. Use a fixed checkout for a
running search. Code changes require a new reviewed plan; they must not silently
change a resumed trial. `--manifest` can relocate identical prepared data to
another host, while its SHA256 must remain unchanged. Device changes are allowed
on resume; settings and scheduler horizon remain fixed. A sibling
`<run>.search_trial.json` is written before training so provenance survives a
worker interruption.

No test metrics are used by this script. Freeze selected settings before held-out
evaluation. A single-seed early screen is not a robustness result. Compare the
selected full-budget model with original models under the limitations in
[BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md), including unresolved historical
exposure and the uncertified raw A/V preprocessing proxy.

Prepared-data identity is rechecked from the actual file bytes at planning and
trial startup/resume. Schema2 covers manifest, photo geometry/provenance, every
referenced RGB/annotation file and standalone FOVs. Matching filenames, sizes or
timestamps alone are insufficient. A regenerated audit after an annotation
change does not authorize silently replacing data in an existing plan. Core
full-data training stores this fingerprint in checkpoints and rejects changed
inputs before optimization. Copied preparations remain portable across hosts.
