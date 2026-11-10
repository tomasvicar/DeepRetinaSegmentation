# Derived-cache handoff — 2026-11-10, 14:00 UTC

This is agent data's compact continuation note. Do not edit HPO staging or any
running immutable87 checkout. No training restart or old-plan migration is needed.

## Source and contract

- Local worktree: `/var/home/tom/code/DeepRetinaSegmentation_derived_cache`.
- Branch: `experiments/unified-derived-cache`, based on `15672d6`.
- Cache implementation: `ef5413b`; subsequent gate/capacity commit adds reports,
  estimator and one extra test without changing the benchmarked production code.
- Ada isolated clone: `~/code/DeepRetinaSegmentation_derived_cache`.
- Read `docs/DERIVED_CACHE.md`, `src/deepretina/derived_cache.py`, and
  `reports/unified/derived-cache/README.md` for implementation/protocol details.
- Runtime requires explicit canonical AND cache fingerprint pins. Portable relative
  references, exact decoded-byte SHA checks on every derivative disk read, no cache
  autodiscovery. Cache path stays outside UnifiedConfig; checkpoint/resume identity
  includes cache fingerprint, never host path. Canonical fallback is explicit and
  only used by the partial-cache benchmark; full training requires all train+val
  image/annotation references and disables fallback.
- Exactly binary target arrays become compressed bool NPZ; soft and negative-zero
  floats remain unchanged. RGB is uint8 NPY. Missing FOV/unknown validity/extras
  preserve current loader semantics. Canonical bytes are never written.

## Pins and completed gates

Canonical manifest on Ada:
`~/code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json`.

- Canonical schema2 FP:
  `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`.
- Manifest SHA:
  `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`.
- Immutable87 source `87cf045ab89425c4994df577fd510d210a1a0eab` remains clean,
  execution SHA `8172e06bfa2c61497d3a34d443027f2f0193834090e95530322e3a6f795254fe`.
- Sample cache is a SUBSET: 64 train photos /140 annotation variants, all17 sources,
  all24 dataset/task pairs, 151,949,829 bytes. All construction arrays exact.
- Sample root on Ada:
  `~/code/DeepRetinaSegmentation_derived_storage/sample64/61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e/v1-subset-b64aa6882dd89fe6`.
- Sample cache FP:
  `d73bb541ea0eb3ee422095b5a12f071906ff6f03e0f667c3f36ce65dae196d9a`.
- 240 full tests PASS; Ruff PASS. Read-only engine review: no semantic blocker.
  Caveats: double canonical startup scan; explicit partial fallback canonical bytes
  checked at opening, not every fallback read. Canonical stays read-only.
- Actual workers2/batch4 gates:384 paired samples (64×six view/augmentation modes),
  exact tensors/observer traces/worker global+private RNG/parent RNG. Timed1280
  additional paired samples exactly matched tensors/traces/parent RNG.
- Incremental CPU delivery vs optimized503:38.8841→27.1419 seconds, aggregate1.433×,
  median1.383×, all five repeats positive. This is a reused64-photo working set,
  warm OS caches, four reference-first/one candidate-first randomized pairs. No GPU
  epoch or model-quality claim. Benchmark PID1049448 and builder1048941 completed.

## Authorized full cache build

Parent authorized the full TRAIN+VAL build after committed gates and capacity
recheck. Default builder excludes test decoding. Expected coverage:2991 unique
photos /6274 annotation variants; image aliases3019 references before dedup.
Estimated full output8.202GB (~7.64GiB), suggested free space10.663GB. Exact RGB
sizing before source-byte dedup, label ratio estimated per dataset/task. At14:00:29
UTC Ada had1,023,318,482,944 bytes free and the fresh full output did not exist.

Target base: `~/code/DeepRetinaSegmentation_derived_storage/full` on Ada, outside all
canonical/raw trees. Exact output will be `<base>/<canonicalFP>/v1-train-val`.
Launch only nice5/max2CPU with CUDA hidden. Use `/usr/bin/time -v` for wall/CPU/RSS;
record wrapper+actual Python PIDs, command, source commit, result/log/resource paths.
**Full build is now RUNNING:** Python PID1053957, `/usr/bin/time` wrapper
PID1053955, launched2026-11-10T14:03:02.159464Z from committed source
`c141493a778a25158e2334eb3dfea5c40766af33`. See the committed launch JSON. Scheduling budget10–30minutes, not a measured
promise. A monitor may only read progress, never launch or restart training.

After launch/completion:

1. PIDs are saved here and sent to root. Continue with read-only build monitoring.
2. Inspect own build PID/log periodically; leave all87 training processes alone.
3. On terminal completion require a complete JSON result and `_SUCCESS`; inspect
   `/usr/bin/time` exit status, wall/CPU/maxRSS and final logical/allocated disk usage.
   Builder writes only final derivative files (no disk staging), so output growth is
   monotonic; final disk consumption is its peak, apart from small report files.
4. Open full `VerifiedCache` with both returned pins and `allow_canonical_fallback=False`;
   call `require_coverage` on every train+val row, verify2991 photos/6274 variants,
   construction equality counts and actual indexed-file SHA checks. Recheck canonical
   and immutable87 source pins. Record full cache FP and byte counts in reports/docs.
5. Commit only this isolated branch's code/reports/notes. Send root the SHA, exact
   full cache path/pins/coverage/resources. Do not silently merge into existing plans.
6. A future final60 plan requires reviewed NEW immutable source/plan and starts from
   scratch after HPO selection. Root chooses architecture/hyperparameters. No87
   source-guard weakening, checkpoint migration, new GPU launch or test evaluation.

## Running training, read-only snapshot around14:00 UTC

- Pascal baseline: PID3910854, E7 macro validation Dice .6739773048, still running.
- old LR.0003: PID287975, E7 macro .7063917974, still running.
- Ada augmentation_strong: PID1041876, E4 macro .5844663763; one Ada training GPU job.
  Source87/original plan9077/data6103; screen8 only,60-epoch scheduler. Do not overlap
  another GPU trial. Its run is in
  `~/code/DeepRetinaSegmentation_unified_trials_87cf045/runs/augmentation_strong-seed42`.
- Ada GroupNorm completed E8: last .5980181932, best E6 .6279426914, all2221 coverage;
  milestone `.../milestones/group_norm-seed42/epoch-008`, best checkpoint SHA
  `96502b9584a974df507e70c2277fccb277137e13d961c619d20c31a4c1aa3586`.
- Width16 paused at E1 by design. Do not resume without root's scheduling decision.
- Original normalized CPU-reference PID1023129 has ended (not killed by us).
- These are unfinished validation screens. Do not claim final superiority across
  tasks; original A/V raw preprocessing comparison remains an uncertified proxy.

## Actual full-build launch, 14:03 UTC

- Python PID: **1053957**, time-wrapper PID: **1053955** on Ada.
- Exact source: `c141493a778a25158e2334eb3dfea5c40766af33`, tracked-clean at launch.
- CUDA hidden, nice5, OMP/MKL/OpenBLAS2; no GPU and no training restart.
- Canonical6103 reverified immediately before launch; free disk1,023,329,628,160B.
- Actual command and runtime provenance:
  `reports/unified/derived-cache/ada-full-build-launch.json`.
- Completion result on Ada clone:
  `reports/unified/derived-cache/ada-full-build.json`.
- stderr: same directory `ada-full-build.log`; resource report:
  `ada-full-build-resource.txt` (time wall/CPU/maxRSS and exit status).
- Fresh output:
  `~/code/DeepRetinaSegmentation_derived_storage/full/61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e/v1-train-val`.
- No full cache fingerprint yet: derive only after successful terminal output and
  pinned reopening/full-reference coverage checks. Never present the partial sample
  fingerprint as the full cache identity.
- Later handoff/report commits do not update the active Ada checkout or source files.
