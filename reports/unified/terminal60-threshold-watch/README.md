# Authorized one-shot terminal60 → six-VAL controller

This controller implements root's explicit instruction to watch Ada PID1257093,
then execute the already-reviewed terminal verifier and six existing validation
producer calls. It never resumes training, opens TEST scores, produces a selection
artifact or automatically promotes a model. Source3157 and evaluation8c remain
unchanged. Its own program has only standard-library imports.

`scripts/watch_control_e060_thresholds.py` polls PID existence every15s, bounded
until2026-11-10 20:00 UTC. If training is still alive at the cutoff, it records
that state and exits without action. All newly started QA/VAL children are also
bounded by this absolute deadline. A timed-out child can receive TERM/KILL only
in its own newly isolated process group; training and unrelated processes are
never signalled. No failed phase is retried automatically.

After actual termination, the controller rechecks verifier/threshold-plan pins
and runs exact helper2221 (`e88c2c17…`) with source3157, plan6be48, canonical6103,
cache4e39 and CPU1/nice10/CUDAhidden. The actual remote plan filename is
`plan-av-residual-cached-approved.json` (no3157 suffix); its byte SHA still6be48.
The verifier refuses existing milestones/reports and creates independent copies.
A `qa_pass` event and notification JSON are flushed immediately. Copied milestone
files become0444 and the directory0555; originals remain untouched.

Before GPU use, it verifies clean8c commit, package3779e1b5 and producer824bab34.
An actual CPU-only probe from the existing old87 uv Python verifies every observed
ML dependency version against the frozen8c lock. The preliminary read-only probe
matched: Torch2.6.0+cu124, NumPy2.5.4, Pillow13.0.0, SciPy1.18.1,
scikit-image0.26.0, headless OpenCV4.14.0.94, h5py3.16.0 and pydicom3.0.2.
These observations are repeated after terminal QA, not substituted for that check.

The threshold addendum binds the independently copied best SHA/selected epoch,
last60 SHA, checked60-epoch scheduler proof, package/runtime, canonical geometry
metadata and parent threshold planbf238040. It is explicitly **not a selection
artifact**. Hardware observation must show physical
`GPU-d0675605-e6c2-1c85-9799-08b279aab0ff`, driver615.78.08 and no compute process
on that GPU before each call. An unrelated/residual compute process stops the
controller without launching or killing anything.

The exact reviewed existing CLI runs sequentially at thresholds
`[0.5,0.35,0.4,0.45,0.55,0.6]`, one fresh report each, same copied weights and
checkpoint tiling, with CPU2/OMP2/MKL2/OpenBLAS2, CUDA_VISIBLE_DEVICES set to the
physical UUID and the frozen8c PYTHONPATH. No geometry override is passed.
Command/environment/readiness observations are preserved for every invocation.
Fresh producer output must bind actual checkpoint/source/runtime/data pins and
all770 photo/993 record inventories. Cross-threshold GT/source/observer support
and per-photo conditional AV counts must match exactly. Actual installed runtime,
CUDA UUID/driver and all protocol fields except threshold must remain unchanged.
Individual full reports and a descriptive per-head/source summary are retained;
there is no automatic threshold winner/selection/freeze artifact.

Seventeen synthetic CPU fixture tests passed in0.05s with no model, checkpoint,
GPU or data operations. Cases cover terminal/deadline behavior, wrong60-source/
sample/optimizer/cache/coverage evidence, GPU occupancy/driver/UUID refusal,
cross-threshold source/weights/geometry/GT/conditional/runtime/inventory drift,
and timeout signals restricted to an owned isolated child instead of training.
The controller was tested separately from the immutable e88 verifier.

Watcher started19:04:00 UTC, PID1312198, niceness10 and approximately22MiB RSS.
Staged source SHA6681f0fced6097516aef9eaa1f2127ea2b992bf8c9ecf805d4e2236f53a9770b,
verifier e88 and threshold planbf238040 were checked before launch and staged0444.
The exact command/hashes are in `launch.json`. At launch training was still alive
with E56 completed, macro0.800123; no actual terminal QA or GPU VAL was started.
Live files are `~/code/DeepRetinaSegmentation_terminal60_watch_20261110/watcher.log`
and `result/events.jsonl`. The existing cup continuation PID1303824 is independent
and is neither restarted nor signalled by this watcher.
