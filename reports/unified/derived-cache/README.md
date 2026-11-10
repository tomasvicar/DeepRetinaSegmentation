# Lossless derived-cache experiment, 2026-11-10

Candidate implementation `ef5413b` is isolated on
`experiments/unified-derived-cache`, based on loader-fast `15672d6`.
The paired reference is the optimized `5031792` loader, so this measures the
incremental storage-format effect. It does not compare cache speed directly to
immutable `87cf045`, and ratios must not be multiplied across different protocols.

A bounded Ada CPU build selected 64 train photographs from all 17 sources and all 24
represented dataset/task combinations, preserving all 140 annotation variants.
Its 151,949,829-byte output is explicitly a subset, with cache fingerprint
`d73bb541ea0eb3ee422095b5a12f071906ff6f03e0f667c3f36ce65dae196d9a`.
Construction restored and compared all 64 RGB arrays and all 140 target/valid/FOV
bundles exactly; every sampled target was binary and could become bool. Soft and
negative-zero arrays are separately regression-tested without quantization.
The recorded launch-to-output-file completion interval was approximately 16seconds;
this is not an instrumented CPU-time measurement.

The real DataLoader used workers 2, batch 4, parent Torch threads 2, worker threads 1,
default prefetch 2, no pinned memory and no persistent workers, under nice 5 with
CUDA hidden. The six basic/mild/strong × patch/whole gates compared 384 samples
exactly, including observer traces, worker global/private augmentation RNG and
parent sampler RNG. Five randomized paired timing repeats delivered 1280 additional
paired samples with equal tensors/observer traces/parent sampler state. Eight
warmup batches were excluded before 64 measured batches per method/repeat.

Aggregate warmed delivery was 38.8841seconds with canonical PNG/float-NPZ and
27.1419seconds with verified NPY/bool-NPZ: **1.433× CPU delivery throughput**.
Median paired ratio was 1.383×; all five ratios were 1.316–1.675×. Excluding the first
pair gives 1.371×. Each measured pair included all 17 sources. Peak parent RSS was
3.10 GiB, including both retained batch sequences for exact comparison.

This is a reused 64-photo working set with warm OS caches. Four randomized pairs
ran reference first and one ran candidate first. Existing 128 MiB per-worker RGB
memory caches were reset before each iterator and operated identically afterward.
The Ada host also ran the existing augmentation_strong GPU trial. Startup cache
verification/support scans are excluded; partial-cache canonical fallback was
explicitly enabled for the full-population support scan, while all gate/timed
photo/variant references were required to be cached. These measurements establish
neither GPU epoch speed nor model quality.

The capacity estimator reads full train/validation metadata and scales label
compression per dataset/task from the sample. For 2991 photos/6274 variants it estimates
8,089,039,433 RGB NPY bytes before source-file alias dedup plus 113,172,873 compressed
label bytes, total 8.202 GB (7.64GiB). Suggested free space with 30% reserve is 10.663 GB
(9.93GiB). Label sizing is an extrapolation, especially for unseen soft arrays; it
is not an exact upper bound. No full cache was built for this report. A future full
build would use nice 5/max 2 threads with a 10–30 minute scheduling budget and publish
only after exact construction/coverage/content checks.

The independent read-only engine review found no semantic blocker. It recorded
that fallback canonical files are checked at opening rather than each fallback
read, and that training currently performs two canonical preflight scans. Neither
changes training values; canonical inputs remain read-only. Full training requires
complete train+validation reference coverage and disables fallback. See
[DERIVED_CACHE.md](../../../docs/DERIVED_CACHE.md) for the explicit pin/resume API.

After the benchmark, canonical and derived bytes were reverified. Canonical
fingerprint remains `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`,
manifest SHA remains `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`.
Immutable87 tracked sources stayed clean and retained execution SHA
`8172e06bfa2c61497d3a34d443027f2f0193834090e95530322e3a6f795254fe`.
No canonical, raw-source or running-training files were changed.

Raw machine reports, launch commands and the post-gate proof are alongside this
note. The capacity estimator is reproducible with
`scripts/estimate_derived_cache.py`; the worker protocol is
`scripts/benchmark_unified_workers.py` at `ef5413b`.
