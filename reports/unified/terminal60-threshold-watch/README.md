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

## Actual terminal outcome

Training terminated normally. The watcher observed this at19:18:30 UTC and the
reviewed terminal verifier passed at19:18:33 UTC. All122880 optimizer steps,
491520 delivered samples,2221 photos/minimum60 visits, finite states, scheduler60,
CSV selection and source/data/cache pins passed. Best is **epoch56**, score
0.8001230874055754. Independent best SHA is
`bfe957460dfed168f2344b1b84ef99b7f502e9dbc8ce1fd4d68632c3234708b1`;
last60 SHA is `60f6d66845a0f453d78b50679815a1123a874a2725fd3897b0eb62fa09d77796`.
`e060-verified.json` has SHA
`f4ebab64f77fc574a112bcd487950cfd8d7df4c3f607363010e3bfa722cfcf20`.
Milestone copies were made0444, directory0555. No original checkpoint changed.

At19:18:34 the controller stopped before **any** GPU forward: the GPU safety query
listed the established desktop's GNOME remote desktop/portal, terminal, file
manager and Chrome as C+G contexts. The initial guard treated every entry from
`query-compute-apps` as blocking. The full process table confirms these are the
known desktop clients; no Python/MATLAB/ML task remains, with observed idle1–4%
GPU/912MiB. No process was killed and no automatic retry occurred. Source8c and
actual locked dependency checks had already passed. A separately reviewed narrow
desktop-context classification is needed before continuing the authorized six
VAL calls; the E60 QA/milestone must not be repeated or overwritten.

## Explicitly authorized GUI-aware continuation

Root independently confirmed the same established desktop clients and explicitly
authorized their narrow identity-pinned allowance. New control program
`scripts/resume_six_e060_validation.py`, SHA
`000e582514c2b755a62ff0f181e785778ddd5f496cc366b0475db297150b1d12`,
uses the unchanged6681 reviewed helper functions and the unchanged824bab existing
VAL producer. It does **not** rerun checkpoint QA, migrate anything, overwrite the
stopped watcher's evidence or modify training/evaluation packages.

Seven existing C+G desktop contexts are pinned by PID, UID, actual `/proc/exe`
and kernel start-time ticks. The approved identity artifact has SHA
`90dd9b3a43d439193adc85f3be6d953d6bb391e28fd67369c53efde87c78c21a`.
Before every producer call, physical UUID/driver and current XML process types
are checked. New/unknown compute or changed UID/executable/start time is refused;
known desktop entries must remain C+G. Pure graphics entries do not claim compute
exclusivity. Full Chrome or other command lines are deliberately not recorded.
Desktop activity remains present, so this is **not a background-free latency
benchmark**. No process is killed or GPU memory explicitly cleaned up.

Seven new tiny synthetic tests passed in0.01s: known GUI accepted, unknown PID,
UID/executable/start-time/type changes rejected, target-GPU XML parsing excludes
process command lines. Existing17 report/timeout fixtures still apply to reused
helper functions. No real model operation occurred during these tests.

The continuation started at19:29:14 UTC, controller PID1321793, with threshold0.50
as the first actual fresh GPU forward. Path:
`~/code/DeepRetinaSegmentation_e060_VAL_gui_20261110/{continuation.log,result/}`.
The original QA reportf4ebab, immutable bestbfe957/last60f6d, source3157, clean8c,
package3779, producer824, actual dependency lock and0444/0555 permissions were
rechecked. `gui-aware-continuation/threshold-addendum.json` records those actual
pins, selected epoch56, geometry metadata and qualified GUI background. Exact
executed argv/environment/readiness for the first call are retained beside it.
The same20:00 UTC deadline applies, six calls maximum, no retry/tuning/TEST or
selection artifact. Completion and cross-threshold evidence checks are pending.
