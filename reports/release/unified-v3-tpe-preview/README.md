# First completed TPE preview release checks

The bundled model is unchanged original BEST58 (`20d4caab…f474d4`), not the later
TPE proposal. The original run completed 60 epochs; both terminal and selected
BEST have full photo/annotation coverage. This release adds no quality evaluation
and does not perform TEST or claim final model selection.

- `model-qa.json`: actual safe CPU checkpoint inspection, finite FP32 state,
  selected-role coverage and actual configuration; no optimizer state to strip.
- `provenance.json` and `original-completion-summary.json`: original completion
  QA, terminal and fresh BEST validation pins. Earlier training scores and fresh
  measured scores are kept distinct.
- `TTA-VAL-results.json`: byte copy of the completed matched Old-GTX1060 TTA
  study; separate from Ada fresh NONE. No new VAL forward was run for packaging.
- `resident-CPU-cost-results.json`: byte copy of the completed matched one-TRAIN-
  photograph latency evidence, one warmup/three calls; mode costs stay separate.
- `preserved-models.json`: all 12 model hashes; all previous eleven registry
  entries and aliases match product base `00d7a94` exactly.
- `real-cli-smoke.json`: actual public model-name CLI, CPU1/interop1/nice17/idle
  I/O, one RAW DRIONS TRAIN photograph, explicit nominal 30-degree geometry,
  `rot4`, observed exit0, finite five-channel original-size outputs and decoding
  constraints. No label arrays, TEST or GPU were used.
- `wheel-validation.json`: wheel-only build using existing cached backend and
  HOME-backed scratch; no installation. All twelve packaged checkpoints and
  registry match exact SHA256 values. No source archive was created.

Focused registry, recipe and CLI tests: **32 passed in 13.89 seconds**, with four
existing skimage morphology deprecation warnings. The recipe instantiates the
actual saved configuration with only portable `device=auto` substituted; the
CPU/Pascal `--mixed-precision none` override is checked without training.
Ruff and `git diff --check` passed. The 148 previously verified TTA tests were not
repeated because TTA arithmetic was unchanged in this release.

The raw smoke initially stopped before inference on an ambiguous duplicate ZIP
member lookup; it then used the exact `images/image_001.jpg` member, verifying its
encoded SHA and TRAIN membership before the sole actual CLI inference.
