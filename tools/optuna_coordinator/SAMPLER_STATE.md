# Sampler state across coordinator restarts

This change affects the public standalone Optuna controller, not the frozen3157
training source, ML configuration, protocol, worker profiles or checkpoints.
Optuna is pinned to5.0.0. Its TPE sampler has **two** RandomState streams:
`TPESampler._rng.rng` and `TPESampler._random_sampler._rng.rng`. Both are saved
and restored. Restoring only the TPE stream would still repeat startup sampling.

The `sampler_state` study attribute is one atomic SQLite-backed JSON value with
both MT19937 key arrays, positions and Gaussian cache values; schema/version,
protocol/binding pins, parameter-history watermark and a checksum accompany it.
No pickle or new dependency is used. Unexpected version, schema or corruption
fails closed. New empty studies retain the original seed42 behavior.

A write-ahead snapshot is stored immediately before a new ask and a committed
snapshot after its pending task is durable. Returning an existing pending task,
reporting a rung, pruning/telling, and importing a baseline consume no sampler
randomness. The watermark contains trial numbers and parameters, excluding
rung/status/session changes. Baseline insertion refreshes that watermark while
preserving both streams. Future ordinary restarts restore the exact streams.

For a pre-existing study that has no saved sampler state, its original streams
cannot be recovered honestly. A one-time migration derives separate stream seeds
from canonical stored parameter history, study/protocol/binding identity and the
preceding stored session. The complete derivation context, digest, seeds, reason
and actual migration UTC are recorded in `sampler_state_migrations`; initial
seed42 is excluded for this migration. This avoids restarting the original
sampling prefix while preserving all existing pending tasks and trial history.
The same pre-migration database yields the same derived states.

An interrupted ask that changes parameter history after its write-ahead snapshot
is detected and uses a separately recorded `unclean_ask_history_mismatch`
migration. This does not claim exact crash replay or repair/finish partially
created trials. Malformed/corrupt state does not silently trigger a migration.

`status` exposes restoration mode, saved-state digest and migration records.
The session metadata still disclaims complete replay: asynchronous completion
order and changed study observations affect TPE independently of RNG persistence.
Neither the old accidental duplicate nor a deliberate infrastructure retry is
relabeled as an independent model improvement.

Verification uses real pinned Optuna5 and local SQLite fixtures: uninterrupted
versus restart sequences match in both startup and TPE phases; existing pending
and promoted tasks leave the streams unchanged; baseline import retains seed42;
legacy migration is deterministic and recorded once; corruption fails closed;
interrupted-ask mismatch is explicit. Original writer-lock and receipt-idempotence
tests remain included. No actual study database, training process, GPU or
validation/test dataset was accessed by these tests.

Operational recovery remains explicit: stop the controller before replacing
its source; preserve study/WAL/transport-state snapshots; deploy the reviewed
controller; restart once and retain its migration record. The duplicated old
trial5 should remain an infrastructure FAIL with its operator-cancellation
receipt and the original pruned trial2 preserved. Do not resurrect, overwrite or
relabel either trial. Worker registration/log archival follows the already
reviewed guarded procedure; source/config/profile/GPU remain unchanged.
