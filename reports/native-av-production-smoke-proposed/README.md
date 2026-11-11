# Pending native AV production functional smoke

No AV training, real-data decode, checkpoint load or forward was performed by this preparation. Root must explicitly authorize execution after its four-model verification completes. This branch is based on d4a1f935838489abd9cdc226155c31e3686dac7d and leaves root's live worktree unchanged.

Helper: `scripts/run_native_av_real_smoke_cpu.py`. Two manually invoked phases `--phase train` and, only after actual original successful observed training exit, `--phase verify`; both require `--execute`. The train phase reads the unchanged native-legacy-av.toml and changes only epochs=1/device=cpu. Architecture remains the original production AV constructor, batch1 and full canonical frame, with no crop/resize/letterbox or architecture override. The saved canonical raw AV4 input retains RGB/255−0.5 plus same-observer valid binary vessel; the model retains its historical internal per-image/channel normalization. This is a new functional training recipe, not historical optimizer or quality reproduction.

Both phases run CPU1/inter1, affinity4, nice17, idle IO and CUDA hidden. Each checks MemAvailable≥5500MiB before spawning; runtime group RSS≤3072MiB and system MemAvailable≥3072MiB. Work is stopped after293s, reserving7s for owned TERM/KILL cleanup inside300s. Limits cause controlled defer/nonzero with no automatic retry or resizing. SIGTERM/SIGINT and post-spawn metadata-write errors enter the same finally cleanup of only the newly created process group. All original numeric child return codes are saved in av-{phase}-observed-exit.json. Existing outputs reject; no foreign process is signaled.

After successful training, separate bounded verify uses safe weights-only AV loader, finite model/optimizer state, exact fresh original-architecture lineage and actual trainable parameter changes against seed42 direct constructor. It then reads an eligible real VAL observer through NativeDataset.sample(crop=False), requires 4×1033×1040, and checks finite full-frame 1×3×1033×1040 reload inference. It checks one-epoch functional subset coverage and no TEST/CUDA initialization, and saves av-safe-verification.json. Native training's own validation already uses full frames. No quality or full5700-annotation training claim follows this six-record subset smoke.

Root command shape, using its existing Torch environment and the NEW helper path:

```
<python> scripts/run_native_av_real_smoke_cpu.py --execute --phase train --data /var/home/tom/code/DeepRetinaSegmentation_native_real_smoke_data_20261111 --runs /var/home/tom/code/DeepRetinaSegmentation_native_real_smoke_runs_20261111
# Root verifies original actual train exit0 before explicitly invoking:
<python> scripts/run_native_av_real_smoke_cpu.py --execute --phase verify --data /var/home/tom/code/DeepRetinaSegmentation_native_real_smoke_data_20261111 --runs /var/home/tom/code/DeepRetinaSegmentation_native_real_smoke_runs_20261111
```

Five focused tests exercised only tiny stdlib CPU processes: nonzero return propagation, timeout cleanup, injected post-spawn ENOSPC cleanup while an unrelated process survives, pending CLI rejection before imports/data/output, and the exact two bounded cleanup waits (maximum6.1s, no third wait). Ruff check/format pass; five tests pass in0.66s. No automatic invocation exists.
