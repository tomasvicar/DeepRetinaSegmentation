# Derived-cache handoff — complete, 2026-11-10 14:09 UTC

Agent data's task is complete. No existing training process or immutable87 file was
changed. Do not edit HPO staging or silently activate this cache in an old run.

## Source and documentation

- Worktree `/var/home/tom/code/DeepRetinaSegmentation_derived_cache`, branch
  `experiments/unified-derived-cache`, based on `15672d6`.
- Implementation `ef5413b`; reviewed gate/capacity + actual full-build source
  `c141493a778a25158e2334eb3dfea5c40766af33`. Later commits only add notes/reports.
- Ada clone `~/code/DeepRetinaSegmentation_derived_cache` stays at build source c141.
- Read `docs/DERIVED_CACHE.md` and `reports/unified/derived-cache/README.md`.
- 240 full tests PASS, Ruff PASS, independent read-only engine audit found no blocker.

## Full artifact — verified ready for a future explicit plan

Canonical manifest on Ada:
`~/code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json`.

Full cache on Ada:
`~/code/DeepRetinaSegmentation_derived_storage/full/61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e/v1-train-val`.

- Canonical schema2 fingerprint:
  `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`.
- Manifest SHA:
  `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`.
- **Full cache fingerprint**:
  `4e39e40def03d0f2859dd147107a3736a62b4fc0249dd440a1280a9803aa8591`.
- Immutable87 source `87cf045ab89425c4994df577fd510d210a1a0eab`, tracked clean,
  execution SHA `8172e06bfa2c61497d3a34d443027f2f0193834090e95530322e3a6f795254fe`.
- Complete train+val: 2991 photos /6274 annotation variants /all 17 sources. No test
  arrays decoded. All 3019 source RGB references and 6274 target/valid/FOV bundles
  exactly checked during construction; all full targets binary, encoded bool NPZ.
- Actual files: 2991 RGB NPY +6268 label NPZ (byte-identical sources deduplicated;
  all variant mappings retained). Pinned reopening rehashed all derived files and
  canonical content, then `require_coverage` verified every train+val reference.
- Sample's204 source mappings and encoded-file hashes match the full cache exactly.
- Files 8,147,071,511B; directory logical 8,152,309,238B; allocated 8,172,400,640B.
  Disk output only grows, so final allocation is its peak excluding tiny reports.
- Build 14:03:02UTC, nice 5/max 2 CPU/CUDAhidden; completed exit 0, wall 4:34.45,
  user 257.68s/system 10.86s, maxRSS 177440 KiB (~173MiB), no swaps.
- Historical Python PID 1053957/time-wrapper1053955 are TERMINAL. No build job remains.
- Reports: `ada-full-build-launch.json`, `ada-full-build-summary.json`,
  `ada-full-build-resource.txt` under `reports/unified/derived-cache/`.
  Full raw result (including photo list) remains only on Ada as `ada-full-build.json`.

## Equality/performance gate and limits

Sample: 64 train photos/140 variants/all 17 sources/all 24 dataset-task pairs, 152 MB,
explicit SUBSET; sample FP `d73bb541ea0eb3ee422095b5a12f071906ff6f03e0f667c3f36ce65dae196d9a`
is NOT the full-cache identity. Source mappings retain every annotator variant.

Actual workers 2/batch 4:384 paired samples across six basic/mild/strong × patch/whole
settings, exact tensors/observer traces/worker global+private RNG/parent RNG.
Additional 1280 timed paired samples matched tensors/traces/parent RNG. Incremental
CPU delivery vs optimized 503: 38.8841→27.1419s, aggregate 1.433×, median 1.383×. Reused
64-photo warm working set, randomized four reference-first/one candidate-first
pair; no GPU epoch or quality claim. Details/provenance are in the report README.

## Exact next steps

1. Root reviews/merges this isolated branch when ready; do not mutate old87 sources.
2. Future final60 training intentionally starts from scratch AFTER HPO selection
   with a NEW immutable source and NEW frozen plan. Root chooses model/settings.
3. Explicitly provide cache directory + external cache pin when creating/running
   the new plan; canonical pin is separately frozen. Full training must verify all
   train+val image/mask references and disable canonical fallback. No autodiscovery.
4. Runtime cache path is outside UnifiedConfig. It may relocate; cache/content/source
   identity and all training settings must remain identical on resume. Never migrate
   an old87 checkpoint/plan or weaken its source guard to adopt the cache.
5. Data remain outside Git and ownCloud; transfer portable derived directory only
   when root schedules the future job. No new GPU trial or test evaluation here.
6. Canonical fallback is only explicit for the partial benchmark. Canonical files
   are checked at open rather than each fallback read; keep them read-only. Full
   training has no fallback. Current training startup rehashes canonical twice.

Read-only training snapshot around 14:00UTC (root monitors newer progress): Pascal
baseline PID 3910854 E7 macro .6739773048; old LR.0003 PID 287975 E7 macro .7063917974;
Ada augmentation_strong PID1041876 E4 macro .5844663763. GN finished E8 (best E6
.6279426914), width16 paused E1. Do not restart, overlap or extend these runs.
These unfinished validation screens do not establish superiority across all heads;
original A/V raw preprocessing reference remains an uncertified MATLAB proxy.
