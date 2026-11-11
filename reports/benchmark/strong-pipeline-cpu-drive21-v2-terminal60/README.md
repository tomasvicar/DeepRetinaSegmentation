# Actual terminal E60 CPU resident timing

The fully annotation-exposed terminal E60 checkpoint (`10670c…fab2`) was measured
against the original four-network vessel/disc0/universal-cup/AV pipeline. Both use
raw DRIVE TRAIN photo 21 and its official FOV, with no segmentation annotation or
TEST scoring. Actual process returned zero at 2026-11-11 04:02:42 UTC.

| Resident pipeline | Median seconds | Parameters | Serialized checkpoint bytes |
| --- | ---: | ---: | ---: |
| Original four networks | 60.577181 | 5,943,312 | 25,068,700 |
| Unified terminal E60 | 5.442199 | 1,106,597 | 14,385,264 |

On this single input the unified model was 11.13 times faster. These are CPU
preprocessing, model inference and original-frame mask restoration timings,
excluding file IO and model loading. One warmup and three measured calls per
pipeline used one numerical thread, CPU affinity 0 and nice 15 on an AMD Ryzen 5
1600. CUDA was hidden. Original component hashes and unified checkpoint bytes were
verified unchanged by the benchmark. The E60 file retains optimizer/RNG state,
so serialized file size is not a count of inference parameter bytes alone.

`timing.json` is the original output; `terminal.json` records actual completion.
`verification.json` binds runtime source and measured resource settings. Benchmark
helper is `scripts/benchmark_strong_pipeline.py`; its report records its byte hash.
Runtime imports used the clean stable `42aa3e3` package, including corrected
nearest resize. The prior composed quality report selects the same original four
weight files and inside-disc cup policy; it is not relabelled as the fresh d14 flat
VAL report. The official raw FOV used here also differs from full validation's
restored canonical FOV. See the verification record and report qualifications.

One photograph and one CPU do not establish population latency or GPU speed.
This is the released control baseline, not the final optimized selection. Its
[actual VAL comparison](../../v2-control60-terminal-val/README.md) still has a
vessel regression despite disc/cup and aggregate gains. Speed is not an accuracy
improvement, and no final selection or TEST authorization follows from this run.
