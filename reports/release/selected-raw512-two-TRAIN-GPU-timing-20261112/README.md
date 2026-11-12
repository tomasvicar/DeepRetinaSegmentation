# Selected raw512: two fixed TRAIN GPU timing points

**Measured:** original selected BEST50 checkpoint
`521b8881d1b2c69a7395be6687a8b86581838e3f5dbb004a150333d454132cea`,
FP32 **rot4 / threshold .35 / tile512 / overlap64**. Both pipelines output five
masks. These are paired timings on two predeclared TRAIN geometries, not a
population speed estimate, quality result or TEST evaluation.

| Fixed TRAIN input | Original four networks, median | Selected unified, median | Original / unified |
|---|---:|---:|---:|
| DRIVE21, 45° | 2.803334 s | 1.081339 s | 2.59× |
| HRF04, 60° | 5.820297 s | 3.367838 s | 1.73× |

[Exact calls, load times, memory counters and output hashes](timing.json).
Each pipeline has one warmup followed by three measured calls, on the same
NVIDIA GTX1060 6GB, CPU2/inter1/OpenCV1, FP32/no autocast. Each pair uses the
same RGB and official FOV. The original is the reviewed four-network recipe:
raw vessel + calibrated provided-FOV AV + dedicated disc0 + universal cup
intersected with the **same-forward** disc0. All original weights are unchanged.

Resident pipeline timing includes preparation, model inference, restoration and
decoding. It **excludes** model loading, input-file I/O, output-file writes and
CLI startup. Separately measured same-process load times (filesystem warmed by
byte hashes) were original/unified **.296436/.181997 s** on DRIVE and
**.320174/.178727 s** on HRF. These are not cold-start times. Final five-mask
shapes and hashes were recorded from the last timed call without extra forwards.

| Size accounting | Original four networks | Selected unified |
|---|---:|---:|
| Complete-network parameters | 5,943,312 | 1,965,637 |
| Measured checkpoint bytes | 25,068,700 total | 8,893,810 |
| Unified state-tensor bytes | — | 7,874,468 |

Original parameters include each complete network even where universal heads
are discarded. The measured unified BEST already has no optimizer/training
state. Artifact size alone is not a compression or speed claim.

**The deployment export is a different artifact:**
`292ea5999a1fc97a767c1c34bb7b4558040267372c5b128f403f2dce50529b29`,
8,892,466 bytes. It was **not timed here**. Separate CPU TRAIN21 checks show
exact model-state/probability/five-mask parity versus the original checkpoint;
that evidence is separately byte-pinned in the JSON. Frozen selection continues
to identify the original checkpoint, not the export.

## Preserved failure and official FOV representation

The first two-case attempt completed DRIVE (actual child0), then rejected HRF's
official TIFF because it decoded as three channels; this occurred before HRF
model loading/forward. Its whole observer therefore exited **1**, not success.
DRIVE was not repeated. After explicit review, a new HRF-only attempt used a
single-channel PNG view made by the cohort's existing rule: **PIL convert L >127,
then binary×255**. Shape2336×3504, foreground6,913,398 and binary-pixel SHA
`9fa8f8a5f63cda5c5bc332a283bfbca5371b6a313f57bb9327feaa751cd9f087`
match the original official FOV. Original TIFF/cohort bytes are unchanged.

The corrected HRF child and outer observer both actually exited0. No raw-TIFF
I/O speed is claimed; no input I/O was timed at all. Original checkpoint,
source and input hashes/stats were unchanged after each attempt. Both cases
ran under exclusive GPU admission and bounded owned cleanup, without concurrent
TEST work. Peak per-pipeline CUDA allocated/working bytes remain in JSON and
are distinct from whole-process CPU RSS or GPU reserved memory.

Training-image timing does not establish better segmentation, clinical benefit,
cold-start latency or general speedup. The selected model's dense-vessel deficit
versus the original and V3's higher artery Dice remain separate quality caveats;
this is not an all-task replacement claim. No weights, aliases, defaults,
frozen evaluation source or registry are changed by this report.
