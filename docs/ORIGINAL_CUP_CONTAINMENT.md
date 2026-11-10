# Optional original cup containment experiment

This isolated validation experiment tests the native universal/raw cup prediction
intersected with either dedicated disc-0/raw or disc-1/raw prediction. It does not
change original defaults, checkpoints, reference reports, thresholds or geometry.
It is motivated by the unified model's existing predicted-disc containment rule.
No test evaluation or GPU inference is involved.

The input is canonical manifest fingerprint
`61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`.
The two strong five-head reference reports are pinned by byte hashes in
`scripts/evaluate_cup_containment.py` (base commit `84f899b`). Each native cup,
disc-0 and disc-1 prediction must reproduce the corresponding reference's integer
confusion counts exactly before its cup intersection is accepted. Annotation
selection uses `UnifiedDataset` validation's deterministic first observer, matching
the references. Cup validity and GT positives remain unchanged; invalid pixels
are ignored. Unsupported cup photos require no forward.

The complete proposed evaluation covers the 612 cup-supported validation photos
(REFUGE 400, G1020 170, RIM-ONE 32, Drishti 10). Its composite reports retain all
770 validation photos and preserve every vascular/disc count and AV accuracy
field from the corresponding strong reference. Only cup counts and aggregates
are recomputed. Empty valid cup GT remains a legitimate false-positive check;
absent cup supervision does not become a perfect score.

## Five-photo canary, 2026-11-10

Immutable execution commit: `b413416`. Ada PID `1277634`, completed successfully.
Six focused CPU tests passed. CPU threads 2, interop 1, OpenCV 2, OMP/MKL/OpenBLAS
2, niceness 10, CUDA hidden. All five native cup and dedicated disc predictions
reproduced the existing reports' per-photo counts exactly. The canary contains
two REFUGE photos and one each from the other three cup sources, selected by
fixed positions in the canonical photo ordering without inspecting results.

Elapsed time was 34.72 seconds, maximum RSS 838120 KiB (818 MiB). The full-data
tile-weighted inference/data estimate is 4752 seconds (79.2 minutes), or 5465
seconds (91.1 minutes) with a 15% margin. This exceeds the authorized autonomous
approximately-one-hour initial budget. The parent subsequently approved a bounded
100-minute full validation run. There are no complete composite scores or overall
quality claims from this five-photo canary.

All canary Dice values increased after intersection, but true positives were
also removed on RIM-ONE with disc-0 (129 to 88) and Drishti with disc-0 (10293 to
10091). This small sample cannot establish an aggregate benefit or select a
postprocessing policy. Original historical-exposure and calibrated-AV caveats
remain applicable. Detailed counts, hashes, parameter counts, tile counts,
runtime and policies are saved in
`reports/cup-containment-20261110/canary/report.json` and `per-photo.jsonl`.

## Full validation launch

Started 2026-11-10 17:44:59 UTC on Ada, child PID `1278841`, supervisor PID
`1278838`. Execution remains immutable `b413416`; the full launch command and
environment are recorded in `reports/cup-containment-20261110/full/launch.json`.
The isolated supervisor enforces 6000 seconds wall time and a 2097152 KiB RSS
limit, sampled every five seconds. It can signal only its newly created child
process group. The original training process is unaffected. Partial count JSONL
is preserved on timeout, memory violation or count-gate failure.

Remote output is outside ownCloud at
`~/code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z/`.
The canary estimate gives approximately 19:04–19:16 UTC completion; the hard
deadline is 19:24:59 UTC. This launch note does not establish full completion.

## Reproduction

Use a fresh output directory and the matching frozen dependencies via uv:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 10 uv run python scripts/evaluate_cup_containment.py \
  --manifest /path/to/unified25/manifest.json \
  --reference-dir reports/disc-reference \
  --mode canary --output /path/to/new-canary
```

Only after reviewing resource cost, use `--mode full` in a new output directory.
The process checks canonical/reference/model hashes and all native count gates
again. Per-photo JSONL is flushed as it progresses. Full reports are written only
after every supported photo passes. A count mismatch must be investigated rather
than accommodated by relaxed tolerances or changed GT.
