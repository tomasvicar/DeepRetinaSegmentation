# Fixed terminal E60 baseline: full TRAIN annotation exposure

This is a usable baseline, not the final optimized model or an independent TEST result.
The original completed control trained 60 epochs (491,520 samples / 122,880 optimizer
steps) on 2,529 TRAIN photos from 21 sources. Its terminal checkpoint contains actual
contributions from all 5,700 eligible annotation records (minimum 66 visits per photo).
Checkpoint SHA256: `10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2`.

The validation-selected E52 checkpoint is preserved separately in
[the best-snapshot report](../v2-control60-best-val/README.md): it includes 5,699 of
5,700 annotation records. We use the explicitly fixed terminal E60 checkpoint for
this complete-annotation baseline; it is not renamed to the best checkpoint.

Fresh CPU evaluation on all 770 VAL photos took 459.79 seconds. Against the
nearest-corrected original pipeline, the actual pooled Dice values are:

| Task | Original | Terminal E60 |
| --- | ---: | ---: |
| Vessel | 0.770123 | 0.750472 |
| Artery | 0.689522 | 0.677109 |
| Vein | 0.697286 | 0.754314 |
| Disc | 0.894808 | 0.934122 |
| Cup | 0.435399 | 0.872808 |

Macro dataset/head Dice rises from 0.667030 to 0.788763; conditional AV accuracy
rises from 0.748566 to 0.788037. Vessel segmentation regresses: paired 95% bootstrap
interval for Dice difference is [-0.023920, -0.015335]. Artery pooled Dice also falls.
The aggregate improvement therefore does not establish improvement on every task.
Original models remain available. A separately protected full-training experiment
increases vascular sampling; no improvement is claimed before its actual evaluation.

`validation.json` contains the fresh enriched report, `producer-terminal.json` the
actual successful producer exit, and `paired-comparison.json` the strict metadata
comparison with 2,000 paired group-bootstrap replicates (seed 42). The externally
pinned `reviewed-context.json` binds the fixed-terminal role and original completion
QA. The selected snapshot, source, data and inference protocol remain explicit.
These are exploratory VAL comparisons with the historical-exposure and preprocessing
qualifications recorded in the paired report. No TEST scoring or final selection was
authorized by this comparison.
