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

## Full run aborted at the historical reproduction gate

The run stopped at 18:05:09 UTC after 1210 seconds, having completed 157 of 612
photos. Peak observed RSS was 965652 KiB (943 MiB). Photo158 is G1020
`g1020_image_532_disc_cup`, ID
`sha256:402f3347e4c847e0beade05125d2c98d73751d2ddd0e8352da5d2002afec23b2`.
Fresh native cup counts were TP4125/FP19778/FN0 versus historical
TP4125/FP19777/FN0, with identical valid885195 and GT positives4125. The exact
count gate refused the one-FP difference. No complete composite/summary exists;
157 partial records and terminal evidence are retained under `full/`. The
post-success watcher correctly skipped composition. No full restart occurred.

### Bounded one-photo diagnosis

The historical universal cup was produced from source87 with CPU8/nice5; later
dedicated disc references used CPU4. Fresh CPU1/2/4/8 forwards on the failed photo
all gave FP19778. CPU2/4/8 logits are bitwise identical. CPU1's logits differ by
up to6.87e-5 but its valid binary mask is identical. Float64 and float32 strict
sigmoid thresholds also give the same counts as strict logit>0 on this photo.
An additional replay of the **actual original87** `segmentation_masks` wrapper,
loading the original universal, dedicated vessel and AV-v9 models, CPU8 and
default interop20, also gives FP19778 and identical logits to CPU2/4/8.

The image input tensor, selected cup GT and validity bytes match between current
b413 and original87 loaders; the legacy forward source is unchanged. Thus the
simple CPU2-versus8 and sigmoid-rounding explanations are **not supported** by
these bounded tests. The underlying historical numerical-state difference remains
unresolved. Native reference counts and all previous files remain untouched.

The exact historical mismatching coordinate cannot be recovered from aggregate
counts because historical logits/masks were not saved. This limitation is explicit
in `forensic/evidence.json`. The nearest-zero current valid pixel is (y0,x855),
GT-negative, with CPU2/4/8/original-wrapper logit+5.95598672e-6 and sigmoid
0.500001488997; CPU1 gives+2.55783290e-6. This is a diagnostic candidate, **not an
identified historical mismatch**. Per-thread full logits stay on Ada; only compact
evidence/hashes are committed. Four thread probes and the original-wrapper replay
took seconds per photo; there was no GPU or test inference.

This gate failure does not establish model incorrectness or a postprocessing
benefit. A scientifically useful follow-up must compare fresh native and clipped
cup counts from the same forwards, log differences from historical cached counts
separately, and retain the untouched historical references. A new reviewed plan is
required before continuation.

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

After a successful full run, `scripts/summarize_cup_containment.py --result
/path/to/result --reference-dir reports/disc-reference --output /new/summary.json`
reports native and clipped cup counts, TP removals and FP reductions for each
source and pooled support, plus all five pooled/source-macro metrics and the
five-head macro. It requires complete 612-photo results, rechecks reference byte
hashes, and verifies all 770 photos and non-cup counts remain unchanged.
