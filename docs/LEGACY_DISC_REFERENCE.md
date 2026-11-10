# Dedicated original optic-disc validation

This isolated branch starts at `0d6d9087a92f0f07cf75fd1083c590e8af6bb18a`.
`scripts/evaluate_legacy_disc.py` compares the original dedicated `disc-0` and
`disc-1` checkpoints under both available original Python input profiles. It
never evaluates test data and never calls A/V or vessel models. Photos without
a disc annotation are skipped before image/vascular-label decoding; photos
whose selected disc support is empty are skipped before any model forward.

The canonical `UnifiedDataset` selects exactly the same first sorted annotation
per task, FOV-restricted valid pixels and source attribution as
`evaluate_original.py`. Original logit tiling/crop/border come from each safe
checkpoint. Only strict `logit > 0` is decoded; zero is background. There is no
cup/disc hierarchy or modern postprocessing. The normalized profile reproduces
the existing evaluator's RGB roundtrip and `local_contrast_and_clahe` with its
first-row FOV and 25 pixels/degree. It is the original **Python** profile;
historical choice of raw versus normalized training inputs remains uncertain.

Every weight/output is checked finite. Reports contain baseline source commit,
evaluator SHA, checkpoint SHA/architecture/tiling, canonical content and manifest
SHA, exact command, CPU thread limit, timing, pooled and per-dataset metrics,
and per-photo source/counts. The report's `source_commit` names the pinned core
baseline; the additive evaluator is independently identified by `script_sha256`.
Primary variant selection should use macro-dataset **disc** Dice on validation;
report pooled Dice too. The universal model is a candidate, not an assumed winner.

## Completed validation of the helper

Three focused tests pass: strict-zero/unknown-support/finite-output behavior and
exact canary count/source checks, and deferred resource/reference gating. Ruff passes. On Ada with GPU visibility disabled,
nice5 and four CPU threads, a three-photo universal/raw canary exactly matched
all per-photo TP/FP/FN/valid-pixel counts and disc annotation source in the full
strong-raw validation report. Total time 15.09 seconds (forward time 10.37 s).
The canary independently verified full prepared fingerprint `61032be218c6…` and
manifest `529edba7743e…`. [Machine-readable proof](../reports/disc-reference/universal-raw-canary.json).
A three-photo canary is a correctness check, not a quality estimate.

## Full run: prepared, not started

The normalized universal reference currently runs as PID1023129 on Ada. Do not
start the following dedicated-disc job until that process has completed and
CPU/RAM capacity is available. This avoids delaying the active unified training.
No existing process should be killed or changed. The checkpoint/data paths are
read-only; output goes to the separate checkout.

```bash
# On Ada, only after the running normalized reference completes:
RETINA_DISC_DIR=/var/home/tom/code/DeepRetinaSegmentation_disc_reference_0d6d908
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH="$RETINA_DISC_DIR/src" nice -n 5 \
/var/home/tom/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python -u \
"$RETINA_DISC_DIR/scripts/evaluate_legacy_disc.py" \
--manifest /var/home/tom/code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json \
--checkpoint "disc0=$RETINA_DISC_DIR/src/deepretina/pretrained/legacy-single-net-disc-0.pt" \
--checkpoint "disc1=$RETINA_DISC_DIR/src/deepretina/pretrained/legacy-single-net-disc-1.pt" \
--profiles raw normalized --threads 4 \
--output "$RETINA_DISC_DIR/reports/disc-reference/dedicated-disc-val.json"
```

Progress snapshots are atomically saved every25 eligible photos beside the
output. They contain each variant's last-photo counts and cumulative forward
time. Final output refuses to overwrite an existing result. No finite/final
quality claim is made before the complete result exists.

After completion, compare all four dedicated variants with universal/raw and
universal/normalized on matching per-photo disc support. Historical exposure
mapping is unresolved, so even the strongest validation result remains an
exposure-qualified reference, not an independent benchmark. Freeze the selected
disc reference before any future held-out evaluation. No test results are used.

## Deferred execution status

After the successful canary, a lightweight stdlib waiter was started on Ada as
PID1035760, niceness5, with a 60-second polling interval. The heavy evaluation
has **not** started. It requires the existing normalized-reference PID1023129
to exit, a complete matching val770 normalized report, at least10GiB available
RAM and load1 no greater than CPU-count minus8. It then runs exactly one CPU4
full dedicated comparison, never retries a failed run, and keeps stdout plus
launch/finish metadata beside the result. It does no CUDA work or model import
while waiting.

The guard is `scripts/launch_disc_reference_when_idle.py`. To keep it deferred
even after resources become available, create the file
`reports/disc-reference/HOLD` in the isolated Ada checkout; removing that file
allows the same guards to be checked again. This changes no existing process.
The [launch snapshot](../reports/disc-reference/deferred-launch.json) records
the exact command and currently waiting status. Full results and reference
selection remain pending, and the canary is not presented as a full evaluation.
