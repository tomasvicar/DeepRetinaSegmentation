# 2026-11-10 experiment evidence

JSON environment reports, validation/test metrics, training CSVs and run metadata are
retained here without images, labels or weights. `manifest.snapshot.json` is a provenance
snapshot, **not a standalone runnable dataset**: its paths require the prepared images/masks
under `data/leipzig`. Check its SHA256 against run metadata to verify the source split.
The same snapshot was used on all three machines. See ../../docs/EXPERIMENTS.md for
commands, task support, score interpretation and limits.

Weights and prepared images are ignored by Git and retained in `runs/` and `data/` locally.
