# Independent bounded review

Reviewed `b5a9af72737504143e9ea6eb3759b74872dcc6f3` without new neural-network
execution, GT decoding or model changes. `git diff de8cf4f..b5a9af7 -- src configs`
is empty: production implementation, recipes and bundled weights are unchanged
from deliverable `de8cf4f6779eb399bbab3832949bc5e1bf7638ce`.

The committed actual four-case proof SHA256 is
`b4ce3cc532f09743c9e11ebb05b638071bd100b99604295343e2be77ea1bc544`.
It reports the genuine filters32/depth4/in3/do0 single-net architecture for all
four recipes, out1 vessel/disc/disc256 and out3 universal(vessel/disc/cup).
Observed child exits are0, steps2/1/1/3, wall42.56/12.02/13.02/44.06seconds.
The independent verifier checks finite stored model/optimizer, same-seed fresh
initialization vs74 changed parameter tensors each, selected task-specific small
subset coverage and actual reloaded64×64 TRAIN-crop outputs. Its code and report
explicitly separate trainer full-frame VAL from the later crop reload check.

This proves short functional updates/reload for four production architectures and
recipes on a tiny unchanged TRAIN/VAL subset. It does not prove full-corpus
retraining, historical optimizer reproduction, convergence or quality gains.
Native A/V full-frame functionality remains pending; no A/V success is inferred.
No meaningful scope/model-metadata blocker was identified in this bounded review.

## Actual non-NN cleanup regression

`tests/test_native_real_smoke_cleanup.py` runs the monitor in an independent test
process. Its Popen shim replaces only the training command with an owned
`/bin/sleep60` in the requested new session; git metadata commands remain real.
The test synchronizes from inside the protected monitor loop, sends SIGTERM to
the monitor, observes numeric exit143, confirms the child returnedSIGTERM and was
reaped (`/proc/PID` absent), and confirms an unrelated owned sleep survives.
No trainer, model, manifest or GT was opened. The interrupted monitor does not
forge a normal smoke-exit receipt. Test fallback cleanup targets only recorded
owned processes with matching start ticks.

**1 actual synthetic process test passed in0.09s**, CUDA hidden, CPU1/nice17/idle
I/O; Ruff/whitespace checks passed. The first test version patched the shared
`time.sleep` module and inadvertently delayed subprocess.wait cleanup itself;
that fixture was corrected to isolate the monitor's clock wrapper. No runner
bug or neural rerun was involved. Scope is SIGTERM after the guarded monitor loop
has been entered, not proof of every possible process-spawn race or signal timing.
