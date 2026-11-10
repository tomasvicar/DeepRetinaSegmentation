# Unified checkpoint interruption recovery

`last.pt` is the committed optimizer/scheduler/RNG/coverage epoch boundary.
A process can stop after publishing a stronger `best.pt` or appending a CSV row
but before committing `last.pt`. Atomic replacement of individual files alone
does not make those publications a transaction.

On resume, the trainer now:

- verifies best/last manifest, normalized config, lineage and executable-source identity;
- retains a stronger best model published up to one epoch ahead of the committed boundary;
- restores `validation.json` from validation metrics embedded in that exact best checkpoint;
- atomically truncates CSV rows beyond the committed epoch, including an incomplete trailing row;
- restores coverage sidecar counts from the committed training state;
- records the repair and retained best epoch/score in `resume_recovery.json`.

The replayed epoch may obtain a lower validation score; it cannot overwrite the
stronger retained model. CSV records the replayed training trajectory, while
`best.pt` and `validation.json` identify the actually selected model. Missing or
contradictory committed metrics, unrelated best checkpoints, or changed recorded
executable sources fail explicitly. Checkpoints created before source hashes or
embedded validation metrics need a matching existing validation sidecar; their
historical source identity cannot be reconstructed by this repair.

This is epoch-boundary recovery, not SIGTERM handling or mid-epoch continuation.
An interrupted partial epoch is replayed. Use a persistent run directory and
allow sufficient scheduler walltime for validation and checkpoint writes.

Verification: `tests/test_unified.py` simulates termination after improved best
and CSV publication, before `last.pt`; corrupts the validation sidecar and adds
a partial CSV tail; then verifies exact final weights/coverage, unique epoch rows,
and preservation of a0.80 best against a0.75 replay. Unrelated manifest, lineage
and source identities are rejected. All29 unified tests passed on CPU with two
threads; no GPU training was used for this change.

Full-data runs also bind checkpoints to schema2 prepared-content identity.
Startup re-reads the referenced input bytes; both the frozen search-plan identity
and the integrity audit must match. Manifest metadata alone cannot conceal a
modified image, mask, FOV or photograph geometry. Resume refuses changed data
before any optimizer update. Executable source identity includes the integrity
checker itself.
