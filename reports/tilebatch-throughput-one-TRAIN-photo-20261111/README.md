# Tile batching: no substantial speedup demonstrated on one TRAIN photograph

Keep the default tile batch 1. On this bounded GTX1060 6GB experiment, batch 2/4 changed full-pipeline median latency by only 0.4–1.4%, while peak allocated memory increased roughly 2×/4×. Three repeats and one image do not establish a substantial or reproducible throughput improvement. No product/default promotion follows.

The actual run on 2026-11-11 used first native TPE BEST58 checkpoint `20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4`, completed 60 training epochs, evaluation source `4c14a044b524bfd9bcedcd60ebbeeaed59739dcc` (execution `200cbf1fba7f4addd137cb9cef789b01bf1d9ab72c2b30082c73a24ce8a24255`). Original checkpoint training source remains `3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28`; the research evaluator was not used for training or resume.

Input was exactly RAW DRIVE TRAIN `21_training.tif` (584×565) plus its official image-only FOV, matching the earlier CPU tradeoff. Raw 25 preparation, FP32 inference, tile 512/overlap 64, joint AV/FOV restoration and 0.5 decoding remained fixed. No annotation, GT, VAL or TEST image was used or scored. CPU 2/inter-op 1, Torch 2.6.0+cu124, Python 3.12.14, CUDA 12.4; actual backend flags and all input/prepared-byte hashes are in the report.

Each of six cases had one warmup and three measured calls, 24 operations total. Measured order per mode alternated 1/2/4,4/2/1,1/2/4. Full latency includes raw 25 preparation, prediction, inverse geometry/FOV and decoding; synchronized core latency measures prediction only. Model load and image IO are outside both clocks. Root forward-hook overhead is present in all cases.

|Mode|Batch|Full median [range] s|Core median [range] s|Full ratio vs1|Peak allocated MiB|
|---|---:|---:|---:|---:|---:|
|none|1|0.44072 [0.44051, 0.44364]|0.18502 [0.18361, 0.18525]|1.0000|129.58|
|none|2|0.43799 [0.43778, 0.45488]|0.18238 [0.18162, 0.18447]|1.0062|251.58|
|none|4|0.43740 [0.43367, 0.43952]|0.17974 [0.17891, 0.18153]|1.0076|495.58|
|rot4|1|1.06360 [1.06193, 1.06724]|0.80739 [0.80547, 0.80866]|1.0000|129.58|
|rot4|2|1.04882 [1.04126, 1.04981]|0.79067 [0.78201, 0.79192]|1.0141|251.58|
|rot4|4|1.05908 [1.05719, 1.06058]|0.80092 [0.80002, 0.80429]|1.0043|495.58|

All 24 operations completed, with observed external supervisor and child return codes 0; no OOM or timeout. Source and input bytes remained unchanged. Model weights/BN buffers and CPU/CUDA/NumPy/Python RNG state hashes remained exactly equal. Peak reserved memory 806MiB reflects resident allocator/order history, not independent cold-case footprints. Actual forward shapes confirmed 9/5/3 calls per view for batch 1/2/4.

Maximum probability difference for every batch 2/4 case versus the same-mode GPU batch 1 was 2.384185791015625e-7, with zero decoded-mask or conditional-argmax changes on this photograph. All per-channel differences, probability/mask hashes, warmup/measured latencies and forward shapes are preserved for every operation. This numerical observation is neither a quality evaluation nor evidence of equivalence over all 770 VAL photographs. Numerical changes would require a separate full VAL quality check, not automatic quality rejection.

The original generic per-case `experimental_speedup_only_no_product_promotion` labels mean only that the observed median ratio exceeded 1; they imply no significance or robust benefit. This negative practical result supports retaining batch 1. It must not be extrapolated to CPU timings, other images/models/GPUs or full-VAL runtime. No retry or further forward was performed for this export.

`measurements.json` is a scientific derivative of original whole-report SHA256 `508adaaae86fcfe6ee4f884b2da78708058f52763601cbc7ce2a71e8a164a96f`; only host input paths and physical GPU UUID are omitted. The 24 original rows remain exactly equal, and summaries only gain exact ranges/counts/memory fields. Actual operational launch/admission/observer artifacts remain private. No standalone certification or scientific receipt was retagged.
