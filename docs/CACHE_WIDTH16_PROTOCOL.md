# Cached width16: fresh first-epoch preservation gate

This is a new, independently initialized run, not a migration of an old
checkpoint. The frozen execution source is
`d83f17a83e873f840751493b1e0457b4c386a07d`, reviewed after 530 tests passed.
Execution-code SHA256 is
`6fd905fc9c065226553beb37e9d8daa801a5fe3e9903fbbcd56b2321d29f5c69`.

[The plan](../reports/unified/plan-width16-cached-d83.json) has SHA256
`c8ab9c4a70335287af8e65ce673c0734a3cd9538d0541d5a0fa0dcac1ac046c0`.
It pins the canonical6103 corpus and full train/val cache
`4e39e40def03d0f2859dd147107a3736a62b4fc0249dd440a1280a9803aa8591`.
Its sole candidate has exactly the original source87 width16 configuration:
seed42, LR0.001, mild augmentation, patch384/batch4,8192samples,8-screen/60-horizon.
Runtime device is the same Ada GPU. The source87 run remains paused at E1.

After the active augmentation screen completes and its E8 checkpoint/milestone
is verified, run this candidate from scratch with `--stop-after-epoch 1`.
Require all2221 training photos, finite weights/optimizer and matching pins.
Compare model tensors/buffers, optimizer/scheduler, RNG, successful annotation
coverage, loss and shared validation counts against the original width16 E1
`last.pt`, SHA256
`33ea51e1d0fa34066532290d5408d64fd54c394ccf6cbb2fd0277c1cfe33d22f`.
Source/cache provenance and new per-photo conditional-count metadata are expected
to differ; they must not be mistaken for a changed model state. Any numerical
differences require explicit investigation, not a hidden tolerance or source-guard
bypass. Original CUDA deterministic mode is warn-only, which is a retained caveat.

Record actual epoch times as an operational comparison. The older E1 overlaps
different CPU reference work, so a raw epoch-time ratio is not an isolated causal
cache benchmark. The paired CPU loader gates provide separate bounded evidence.
Only after this gate may this new cached run continue to its eight-epoch screen.
No final model quality, test score or universal epoch speedup is established by
the plan. All old source87 trials and original ownCloud files remain unchanged.

## Actual launch

The preceding strong-augmentation screen completed E8 and passed CPU QA with an
independent milestone. The new cached run started on Ada at 14:48:35 UTC,
Python PID1067139, only the known desktop GPU processes present. It stops after
E1 for the preservation gate. The launcher inherits the original no-OMP/MKL
override policy; current Torch parent thread/inter-op counts are20/20, whereas
historical counts were not recorded. No old checkpoint was resumed.

[Launch metadata](../reports/unified/width16-cached-e1.launch.json) records the
exact command, source, plan/cache/canonical pins and GPU UUID. A successful launch
is not a completed gate or a final-quality claim.

## Completed E1 gate and continuation

The real first-epoch gate PASSED: all model tensors/buffers, optimizer/scheduler,
RNG, successful annotation coverage, shared validation counts and training loss
compare exactly. Source/cache provenance intentionally differs. Both runs used
all 2221 training photos,2048 successful updates,8192 samples and the60-epoch
scheduler. The cached E1 milestone is independently copied and checksum-verified.

Observed train/validation times were164.826/65.539s versus421.653/193.456s.
Combined3m50s versus10m15s is an operational observation; different background CPU
reference work prevents assigning the full ratio causally to caching alone.
The earlier paired worker benchmarks give separate controlled CPU evidence.

[Full preservation report](../reports/unified/width16-cached-e1-preservation.json)
and [checkpoint QA](../reports/unified/width16-cached-e1-verification.json) retain
all hashes and exceptions. The same new cached run resumed to E8 at15:02:54UTC,
PID1074202, with no source/plan/config/checkpoint migration.
[Resume command](../reports/unified/width16-cached-resume-e008.launch.json).

Pascal width8 was separately SIGSTOP-protected at15:03:51UTC because its local
Btrfs failed checkpoint-sized atomic replacements. Its state remains live in
RAM; it was not cancelled or restarted. Resume requires repeated filesystem
write/replace and Git index checks after administrator repair. Its paused epoch
must not enter speed comparisons. Other hosts' trials continue.
[Pause record](../reports/unified/width8-storage-pause.json).
