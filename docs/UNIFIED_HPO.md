# Full-data unified model search

This page preserves the **historical V1 search on 17 sources**. Its commands,
candidate list and progress statements describe that study, not the current V2/V3
search. Start with [the current experiment overview](EXPERIMENTS_CURRENT.md) for
completed full-data results, frozen scientific sources and the next TPE stage.
Single-dataset pilots remain preliminary. No new model has established an
improvement over the original models in every task.

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

## First full-data epoch and normalization refinement

The first baseline epoch used every one of the 2,221 training photographs and
3,264 of 5,281 eligible observer/task variants, selected stochastically. There
were 8,192 delivered crops and 2,048 optimizer updates. On GTX 1070, training took
897.53 seconds and full validation took 310.11 seconds, excluding startup data
checks. Macro validation Dice was 0.3762; this is an undertrained diagnostic,
not a selected final model or a quality improvement claim. The learning-rate 0.0003 trial
also covered every training photo and reached macro 0.4408 after one epoch;
early rankings are not final convergence evidence. Both retain a 60-epoch
cosine horizon and are continuing to the 8-epoch screen.

A bounded paired probe on 12 predeclared validation photos found heterogeneous
BatchNorm-statistics sensitivity, without changing the checkpoint or using test
images. Temporary tile statistics improved some cup/vessel predictions and
worsened others; they are not deployed as an inference change. The frozen
supplemental `reports/unified/plan-groupnorm-87cf045.json` therefore adds one
from-scratch GroupNorm candidate with all other baseline settings unchanged.
Its parent plan hash, execution source, dataset identity, seed and training
budget are recorded. Separate plan provenance remains distinct from the
original seven-candidate plan; no existing run is silently reassigned to it.
