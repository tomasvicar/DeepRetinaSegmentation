# Lossless annotation loading optimization

This isolated branch keeps all current source87cf045 runs and canonical data
unchanged. It replaces the CPU boolean-indexed target assignment with dense
`torch.where(..., out=target)` and decompresses each selected FOV only once.
Supported target values are copied exactly; unsupported pixels retain their
previous value. It retains float32 soft targets, task selection, annotation
traces, augmentation and every random-number draw.

The paired benchmark compares this checkout against the actual immutable
source87cf045 `unified.py`, loaded separately in the same CPU process. Both
implementations decode the same original PNG/NPZ files. Method order is shuffled
per photo; bounded RGB caches are cleared before each method; alias verification
is retained. No GPU, optimizer, canonical-file writes, OS cache eviction or derived
serialization cache is used. Both versions run with two PyTorch threads.

## Numerical checks

- `pascal-cpu.json`: 459 paired samples across 68 photos and all17 sources;
  basic/mild/strong augmentation and patch/whole views, three seeded passes.
- `pascal-confirmation.json`: 204 further paired samples, mild patch configuration.
- All image/target/valid tensor values, photo indices, successful annotation
  traces, global PyTorch RNG states and augmentation generator states matched
  exactly for every pair.
- The full suite passed: 227 tests, including epoch resume with workers0 and2.
  Ruff checks and formatting passed.

This does not exhaust every annotation variant or arbitrary future corpus. The
existing annotation range and geometry checks remain active.

## Timing

The confirmation was run after the unit tests finished. Each row processes the
same 68 source-stratified photos with the same seeds in paired randomized order:

| Pass | Original seconds | Candidate seconds | Sample throughput ratio |
|---|---:|---:|---:|
| 1 |10.501|7.335|1.432×|
| 2 |10.065|7.113|1.415×|
| 3 |11.231|7.768|1.446×|

This reduces CPU sample-pipeline time by29.3–30.8% on this sample. The host also
runs the existing GTX1070 training job; it is not an idle benchmark machine.
The first all-configuration equality run overlapped unit tests briefly and is
retained primarily as numerical evidence, not the timing headline.

These are CPU `__getitem__` timings, **not GPU epoch speedups**. Worker IPC, optimizer
work, validation and full-data preflight are outside this measurement. Before
adoption into a new frozen training plan, measure actual workers2 and GPU epoch
throughput. Existing checkpoints must not be silently resumed under changed
source fingerprints. The optional larger NPY/bool cache suggested by the earlier
I/O profile is not implemented here.

Reproduce using an environment installed with `uv sync --frozen`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
uv run --frozen python scripts/benchmark_unified_loader.py \
  --reference-module /path/to/frozen87cf045/src/deepretina/unified.py \
  --manifest /path/to/unified25/manifest.json \
  --output reports/unified/loader-fast/replay.json
```

Add `--only-mild-patch` for the confirmation timing protocol. The manifest and
both loader hashes are stored in each JSON; the confirmation also records its
benchmark script hash. The initial exploratory JSON predates that extra field.

## Actual two-worker DataLoader gate on Ada

`benchmark_unified_workers.py` compared the unchanged `87cf045` reference module
with candidate `5031792` using the frozen baseline configuration, batch size 4,
two DataLoader workers, default prefetch factor 2, no persistent workers and
two parent CPU threads. CUDA was hidden; the process and its workers ran at
niceness 5. No model, optimizer, canonical file or active training source was
changed. Both module SHA256 values were checked against their Git objects.

All six patch/whole-view and basic/mild/strong augmentation combinations passed:
**216 paired samples across all 17 sources**, with bitwise-equal RGB, targets,
validity, sample indices and observer traces. The gate additionally compared
worker CPU RNG, private augmentation RNG and parent sampler-generator states.

Five randomized-order timing pairs used the real full-population coverage
sampler with seeds 42–46. Dataset/support initialization and eight warmup batches
were excluded. Each timed window consumed 64 batches / 256 distinct photographs
from the shuffled coverage prefix. Tensor and observer-trace equality was checked
after timing on all **1,280 additional paired samples**.

| Pair | Order | Reference seconds | Candidate seconds | Throughput ratio |
|---|---|---:|---:|---:|
| 1 | Reference, candidate | 16.718 | 10.809 | 1.547× |
| 2 | Candidate, reference | 15.630 | 9.610 | 1.626× |
| 3 | Reference, candidate | 16.286 | 10.268 | 1.586× |
| 4 | Reference, candidate | 14.343 | 8.466 | 1.694× |
| 5 | Reference, candidate | 12.500 | 8.417 | 1.485× |

Equal delivery work took **75.477 → 47.571 seconds**, an aggregate **1.5866×**
throughput ratio; median paired ratio 1.5860×. Peak parent RSS was 3.10 GiB;
retaining both sides for exact comparison accounted for 2.79 GB of tensors.

This measures a warmed CPU DataLoader prefix. Ada was shared with the running
GPU experiment and eight-thread original-model reference. Randomized orders
happened to contain four reference-first pairs and one candidate-first pair;
all five improved, including the candidate-first pair. The result establishes
CPU delivery improvement under this measured workload. Full-epoch and GPU
training speed require separate measurement. The balanced extra-sampling tail
was not timed.

Evidence: `ada-workers2.json`, `ada-workers2-summary.json`,
`ada-workers2-launch.json`; protocol: `scripts/benchmark_unified_workers.py`.
