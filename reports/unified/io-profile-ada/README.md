# Ada CPU data pipeline profile (source87cf045)

Measured64distinct training photographs, stratified across all17sources (3–4per
source), including142native annotation variants. Source and canonical data stayed
unchanged. Execution used `nice5`,2PyTorch/OMP/MKLthreads, and an empty
`CUDA_VISIBLE_DEVICES`; no GPU or optimizer was used. GN training and the
8thread original-model reference evaluation continued independently. The process
peaked near1GBRSS. Temporary arrays/copies were outside the corpus on Ada's local
NVMe and were removed afterwards.

[profile.json](profile.json) contains observations, line attribution, config,
manifest/source hashes and identity checks. [protocol.py.txt](protocol.py.txt)
is the exact temporary measurement protocol, retained as a report artifact rather
than a production loader. It imports the immutable source snapshot directly.

## Main result

| Complete64sample CPU pipeline | Total | Median/sample |
|---|---:|---:|
| Original PNG+float32targetNPZ |2.879s|35.21ms|
| Temporary RGBNPYmmap+booltargetNPZ, cast back |1.793s|21.47ms|

The prototype reduced total sample-processing time by37.7% (1.61×throughput) on
this stratified sample. Complete-pipeline method order was randomized per photo.
Both returned bitwise-identical image/target/valid tensors, sample indices and
chosen annotation traces under the same seeds, including existing crop and mild
augmentation. This is not a measured epoch-speed improvement: IPC, GPU work,
validation, worker contention and preflight are excluded. The sample is roughly
source-balanced, not the exact full-epoch sampling distribution.

The original source's traced time was2.981s, about3.5% above its untraced time.
Inclusive sample-loading totals in the JSON overlap detailed substeps and must
not be added twice. Approximate exclusive shares of traced `__getitem__`:

| Stage | Seconds/64samples | Share |
|---|---:|---:|
| PNGdecode plus RGBtensor conversion |0.892|29.9%|
| Native NPZread/decompress/cast/FOV |0.725|24.3%|
| `target[support] = mapped[support]` |0.659|22.1%|
| Paired augmentation |0.274|9.2%|
| Numpy five-channel mapping |0.108|3.6%|
| Padding/whole-view branch |0.090|3.0%|
| Foreground mask/coordinate selection |0.069|2.3%|
| Five-channel initial allocation |0.059|2.0%|

Foreground selection is a minor cost in this measurement. The large boolean
indexed target copy is an important separate CPU hotspot. The original source
also decompresses `data['fov']` twice per selected annotation (shape check, then
intersection); its two lines took0.136s combined here. These are observations,
not implemented optimizations or measured alternative algorithms.

## Serialization-only controlled comparison

Methods read the full arrays, including a real copy/touch of mmap pixels; lazy
mapping without accessing pixels was not counted as a read. Method order was
randomized. The first pass used fsync plus POSIX_FADV_DONTNEED **only on private
scratch copies**, avoiding eviction of files used by active jobs. This is
cache-advised/cold-ish, not a guarantee of physical disk cache state. Two further
warm passes agreed closely. Median warm numbers below use the final pass.

| Full array read | Cache-advised median | Warm median |
|---|---:|---:|
| RGBPNG |15.26ms|10.92ms|
| RGBNPY |1.40ms|0.180ms|
| RGBNPYmmap with actual pixel access |1.80ms|0.262ms|
| Float32target NPZ, target/valid/FOV |7.62ms|6.22ms|
| Booltarget NPZ, cast target back tofloat32 |3.74ms|3.26ms|

Sample storage: RGBPNG45.6MB versus NPY148.1MB (3.25×RGBstorage); existing
float-target NPZ5.39MB versus bool-target NPZ4.30MB (20.3%smaller). These sizes
are measured on the stratified sample, not a full-corpus capacity estimate.
Ordinary NPY reading was a little faster than mmap when reading every pixel;
mmap is not automatically preferable for this full-frame conversion pipeline.

Identity checks passed for all576RGB reads, all142native target/valid/FOV sets
and their five-channel mappings, and all64complete sample pipelines. Binary
encoding is applicable only to audited0/1ground truth and explicitly casts back
tofloat32. It must not silently quantize soft labels or future teacher probability
outputs. Validity and FOV semantics remain unchanged.

The bounded dataset RGBcache was cleared per measured sample to represent a
large working set; the operating system's original-corpus cache was not evicted.
Original alias-verification checks were completed during staging. Thus the main
pipeline comparison is a warm/steady-state preparation probe, not the full cost
of each worker's first-ever alias validation. Both methods had been warmed by
serialization checks before complete-pipeline timing. One initial fixed-order
exploratory pass agreed (2.880→1.780s); the retained JSON uses the randomized-order
confirmation reported above.

## Recommendation for a future frozen pipeline

1. Keep all current runs, checkpoints and canonical files unchanged. Stage
   canonical data on job-local scratch independently, as already being handled
   by the operational workflow.
2. First evaluate a derived lossless RGBNPY cache and booltargetNPZ cache outside
   the canonical corpus, keyed by its original manifest/content fingerprint and
   cache schema. Validate tensors on construction and preserve canonical
   provenance. Ordinary NPY is the simplest promising RGB variant here; mmap
   offers no demonstrated advantage for full-frame reads. Budget its extra disk.
3. Independently test removal of repeated FOV decompression and a numerically
   equivalent replacement for the boolean indexed target copy. The profile
   identifies opportunity but does not establish the speed or correctness of an
   unimplemented replacement. Do not spend the first optimization effort on
   foreground selection or weaken augmentation on the basis of these timings.
4. Only after a new implementation passes exact-pipeline equality and an actual
   worker/GPU throughput benchmark should it enter a new frozen search plan.
   Do not migrate an active source87checkpoint to changed executable sources.

The11minute Meta preflight was not profiled here and cannot be attributed to one
cause by these results. Metadata latency, content verification, all-variant
support scans and network storage can dominate there. Stage-level times from
64localNVMe samples should not be extrapolated into a preflight guarantee.
