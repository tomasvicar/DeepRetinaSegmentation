# Full validation check of the alternative original input profile

The original universal disc/cup, dedicated vessel and v9 A/V chain was evaluated
on the same 770 validation photographs using the alternative original Python
single_net local-contrast normalization. This completes the comparison that was
previously only a small, pre-patient-review diagnostic. No test scores were opened.

| Output | Raw profile Dice | Normalized profile Dice |
|---|---:|---:|
| Vessel |0.770123|0.542969|
| Artery |0.650863|0.496960|
| Vein |0.664795|0.516789|
| Disc |0.807155|0.009490|
| Cup |0.401057|0.010309|

The dataset-balanced five-head mean is 0.610351 raw versus 0.308036 normalized.
The selected annotation support and positive totals match exactly for every
head. These results support the current raw default for this original chain;
they do not recover which inherited Config value was used to build historical
training files. Dedicated disc checkpoints require their own profile comparison.

Both runs use the same explicitly uncertified raw Python A/V proxy. Changing the
vessel preprocessing also changes the binary vessel input to A/V; this table is
therefore a complete-chain comparison, not an isolated A/V photometry ablation.
The separately calibrated MATLAB-style A/V profile has not supplied these scores.
Historical exposure and original training-file provenance remain unresolved.

Execution used immutable source `87cf045`, CPU only on Ada, eight threads and
niceness 5. Actual observed wall time was 2639.74 seconds (43m59.74s), with
17498.95 CPU seconds (4.86 core-hours); procfs sampling resolution was 0.1s.
The machine also ran other project work, so this is an operational observation,
not a latency benchmark against the earlier raw run.

Evidence: [scores and per-photo counts](original-strong-normalized-val.json),
[launch configuration](ada-original-strong-normalized-val.launch.json),
[observed timing](original-strong-normalized-val.timing.json).
