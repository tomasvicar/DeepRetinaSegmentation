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
