# Actual full product CPU suite — 12 November 2026

**PASS: 2,851 passed, 0 failed, 0 skipped, 83 warnings in 41.27 seconds.**
This execution tested product source `b36a5ee11be4fc3ba0edfe73a20cd1561a444074`,
not every later repository revision. All 63 test modules appear in the JUnit
results; the source catalog contains 456 AST test definitions, a different count
from expanded parametrized cases. No cases were excluded.

[Verification](verification.json), [JUnit](junit.xml), [console](final-full.stdout)
and [systemd exit/resource summary](final-full.stderr) retain the actual outcome.
The actual pytest return, main process `exited/0` and external client return code 0 agree.
The runner's planned exit is not used as an independent witness.

## Runtime and custody

- Existing Python 3.12.14, pytest 9.2, Torch 2.6.0+cu124, NumPy 2.5.4 and locked-compatible
  dependencies; no installation or mutation of an existing environment.
- CPU-only: CUDA hidden and uninitialized afterward; one CPU core, intra 1/inter 1,
  nice 17/idle I/O, CPUQuota100%, MemoryMax 3 GiB, MemorySwapMax 0, RuntimeMax 1800 s.
- Actual service wall 43.293 s, cgroup CPU 42.625 s, charged memory peak 793,190,400 bytes,
  swap 0. Main-process RSS peak 1,021,324 KiB is a **different** accounting scope:
  RSS includes shared mappings, while cgroup accounting charges physical pages.
  Neither is a real model-training memory estimate or an inference benchmark.
- The isolated checkout's 588 tracked file hashes remained identical before/after;
  all 12 bundled weight checksums and aliases match. Existing frozen Git stores,
  environments and original source files were preserved. See
  [source manifest](source-files.json) and [bundled hashes](bundled-models.json).
- The authorized owned original-v9 checkpoint, known SHA 75c5048a…c8cb1c0,
  was supplied read-only. **Both** actual conversion/geometry and forward-versus-
  original-source cases passed (not skipped). All other model inputs are the
  repository's safe bundled weights or test-created fixtures.

Tests include tiny CPU forwards/training, serialization and synthetic adapters;
this is not a real-corpus training, quality evaluation or held-out TEST result.
No GPU computation, raw archive scan or OwnCloud write was performed. Historical
prepared-input parity and full-model quality have their own documented scopes.

## Earlier failures are preserved, not retagged

1. A full source 275 run reached about 92% and stalled in the first workers=2 resume
   case. It was explicitly stopped after a verified idle period, not completed.
   Its systemd client returned 0 for the manual stop, but the main process was
   killed by TERM: **that was not pytest PASS** and no completed JUnit existed.
2. Separate capped diagnostics on original 275, a guarded original 275 and an
   unadopted spawn candidate reproduced the wait. Parent stacks waited for
   DataLoader results. This did not establish a production fork defect.
3. The harness had chosen a deep HOME-backed `TMPDIR`. A stdlib Listener directly
   failed on a 112-byte Unix socket address with `AF_UNIX path too long`; a short
   HOME prefix produced 56 bytes and succeeded. The tensor-sharing resource-sharer
   mechanism explains the waiting queues; the original hidden feeder stderr was
   not recovered. **The same original 275 case, same guarded bootstrap and limits,
   changing only TMPDIR, passed in 1.22 s.** No production loader change was adopted.
4. The complete 275 run with that corrected environment then produced
   **2,850 passed /1 failed**: a historical config-order assertion omitted the
   subsequently appended `photometric_profile`. The exact failure is preserved in
   [the prior console](corrected-full.stdout).
5. Test-only commit b36 protects the historical positional prefix, explicitly
   checks raw photometry and old BF16 positional construction, and allows future
   fields appended afterward. Production code/model bytes did not change.
   Its focused case passed before the final full 2,851-case run above.

The original byte-complete operator receipts, failed staging preflights,
source/runtime identities and all diagnostic traces remain in the private
operations evidence. Public copies redact only operator paths/PIDs/hostname;
[export provenance](export-provenance.json) binds original and exported hashes.
The historical static catalog was not rewritten. The
[new execution inventory](actual-test-inventory.json) has the actual b36 label and
exact test-file hashes.

## Reproduce from the reviewed checkout

After installing the development dependencies once with `uv sync --frozen --dev`,
choose a **short**, owned IPC temporary prefix before interpreter startup:

```bash
DEEPRETINA_TEST_TMP=$(mktemp -d "$HOME/.drt-XXXXXX")
export TMPDIR="$DEEPRETINA_TEST_TMP"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python reports/release/product-full-CPU-suite-20261112/run_tests.py -q -ra
```

The small guarded reproduction helper sets intra 1/inter 1 and invokes unchanged
pytest. The actual execution additionally enforced the resource limits above.
On a checkout lacking this later report helper, the standard command is
`uv run --no-sync pytest -q -ra` with the same environment; actual source/version
and resource scope still need recording. With no owned original-v9 fixture,
its two optional tests honestly skip; see [external inputs](../../../docs/TESTING.md#external-inputs-and-skip-boundaries).

Keep `TMPDIR` short: multiprocessing appends `pymp-*`/`listener-*` for Unix sockets.
Test artifact/results directories may be longer; they are not the IPC prefix.
Remove only the owned temporary directory after its tests and children exit.
No new quality claim, model selection, default or bundled weight follows from
this software verification.
