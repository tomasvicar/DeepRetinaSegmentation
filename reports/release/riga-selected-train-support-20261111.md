# Three RIGA TRAIN records: support and actual contribution audit

Observed `2026-11-11T00:51:21.042000+00:00`. Frozen training source `6b3dfad605a256e904a877efcdcc1f3cc6135dd0`;
`unified.py` SHA256 `3260a00bc03b44544556adf5090a412318963fe1b121f7627f265c2fe3aae72d`.
Manifest SHA256 `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`.

Only the three named TRAIN masks and live coverage/CSV metadata were read. No
Torch/model/Dataset execution, TEST mask access, data mutation, or training change.

| Record suffix | Options on same photo | Actual photo visits | Actual record visits | Disc/cup positive pixels | Valid pixels per head |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BinRushed2_image44_observer4_disc_cup` | 12 | 56 | 0 | 9760/3740 | 360052 |
| `MagrabiFemale_image15_observer3_disc_cup` | 6 | 51 | 2 | 14956/4054 | 788446 |
| `MagrabiaMale_image33_observer3_disc_cup` | 6 | 46 | 0 | 11303/1220 | 786526 |

The captured CSV's last epoch is 32; the contemporaneous coverage records
5,698 of 5,700 contributing annotation records and all 2,529 training photos.
The previously missing female observer has now contributed twice. Files are read
independently while the job runs; this is a timestamped observation, not a frozen
terminal checkpoint QA or a guarantee that another epoch cannot finish meanwhile.

All three masks are usable, finite, binary, and have positive disc and cup support
inside valid FOV. Cup has zero supported pixels outside disc. The annotations are
not quarantined or empty. Geometry and exact mask hashes are in the adjacent JSON.

The first photo is an exact decoded alias of `BinRushed3/image24`, so its options
contain twelve annotation variants, not six. The other two photos have six.
Frozen `UnifiedDataset.sample` chooses `torch.randint(len(options))` on each training
sample. It does **not** cycle observers deterministically. Disc/cup remains a paired
annotation bundle. The returned trace is retained if either disc/cup validity
survives the crop. Contribution is counted only after a successful optimizer step.

For every inspected photo, the sum of actual sibling annotation visits equals its
actual photo visits. Thus no disc/cup trace was lost from these counted samples
because of crop validity. The still-zero records are consistent with stochastic
observer nonselection, rather than unusable masks; unseen failed/prefetched samples
are not claimed as contribution. No sampled tensors or training RNG were altered.

The JSON additionally gives `(1-1/N)^K` for uniform IID draws at the **observed**
photo visit counts. This is only a simplified selection diagnostic, not a p-value
or future-coverage forecast: records were chosen because they were missing, many
records exist, and actual worker RNG streams are deterministic/correlated with
other random operations. No guarantee of contribution by epoch 60 follows.

The strict completion gate still requires all 5,700 records to have actual positive
visit counters. Neither sampler nor data nor gate was changed.

## Future experiment, not a change to this run

A separate recipe could retain photo coverage-first sampling and maintain an
explicit observer-cycle counter per photo and task. The twelve-variant aliased
photo would then have predictable observer scheduling rather than relying on
independent random selection. Actual crop support and successful-update counters
would still be required; merely scheduling a record is not contribution.

Such an experiment requires a new source/recipe identity and declared seed/RNG
lineage, with measured validation quality and a matched compute budget. It must
not relabel current random-sampling receipts as cyclic or migrate the running
checkpoint silently. The current frozen source, sampler, counters and completion
gate remain unchanged. Future actual visits must be observed, not predicted.
