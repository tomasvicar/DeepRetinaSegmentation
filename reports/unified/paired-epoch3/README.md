# Early paired validation: an aggregate gain still hides vascular regressions

This is an **exploratory epoch 3 diagnostic**, not a final new-model claim or
independent held-out comparison. The new LR 0.0003 model was trained from scratch;
the reference is the original dedicated vessel model, universal disc/cup model
and original v9 A/V model using the existing uncertified raw Python proxy.
Historical exposure/profile uncertainty remains. Dedicated original disc models
and the improved MATLAB preprocessing still need assessment before final selection.

The comparison aligns all 770 validation photos, all selected annotation-source
assignments, all annotation record IDs and valid/positive counts. It rejects
missing/duplicate photographs, mismatched input manifests, identity groups across
partitions, inconsistent GT support and summaries that disagree with per-photo
counts. Aggregate scores reproduce both original reports exactly.

There are 612 reviewed split groups, stratified by canonical source. All images of
a known patient/alias group share each bootstrap draw and both models use the
same draw. **These are not 612 proven independent patients**: opaque IDs remain
source/photo groups. The 2000 percentile draws with seed 42 condition on these checkpoints
and data; they do not measure training-seed variation, resolve historical
exposure or make validation used for tuning an untouched benchmark.

| Output | Original pooled Dice | New E3 pooled Dice | New minus original, paired 95% interval |
|---|---:|---:|---:|
| Vessel |0.7701|0.6896|−0.0805 [−0.0944, −0.0683]|
| Artery |0.6509|0.4294|−0.2214 [−0.2436, −0.1995]|
| Vein |0.6648|0.5976|−0.0672 [−0.0819, −0.0532]|
| Disc |0.8072|0.8422|+0.0351 [+0.0056, +0.0607]|
| Cup |0.4011|0.6964|+0.2953 [+0.2643, +0.3189]|

The declared dataset-balanced five-head mean rises 0.610351→0.641093, a paired
delta 0.030742 [0.013554, 0.045102]. That aggregate does **not** establish that the
new network is better for vessels/A/V. Continue the full training/search budget
and examine those heads explicitly; retain the original models as usable choices.
Both pooled and dataset-balanced per-head intervals are in
[the checked result](comparison-checked.json).

## Provenance and reproduction

The new report is extracted directly from the safe-loaded `best.pt` embedded
validation metrics, preserving its epoch/steps, model hash, complete config and
training/data hashes in [the snapshot](lr0003-epoch003-validation.json).
Its checkpoint SHA256 is
`858a0eab3ec2c6b3a3b3b2344f0fe1b53eff49d8138e30465d5ff5b18f23c21d`.
An exact verified copy is preserved on old under
`~/code/DeepRetinaSegmentation_unified_trials_87cf045/milestones/lr_0003-seed42/epoch-003/best.pt`.
The ongoing trial, its optimizer and source 87cf045 were not changed.
Original raw data and checkpoints in ownCloud were read only.

```bash
uv run python scripts/compare_predictions.py \
  --reference reports/unified/original-strong-raw-val.json \
  --candidate reports/unified/paired-epoch3/lr0003-epoch003-validation.json \
  --manifest /path/to/unified25/manifest.json \
  --draws 2000 --seed 42 --output /tmp/paired-validation.json
```

This opens saved counts and manifest metadata, not models or masks. Report/input
hashes and the comparison module hash are retained. It uses each source's pooled
confusion counts for Dice, then averages supported sources per head and heads for
the macro metric; it does not silently substitute a mean of photo Dice values.
Undefined bootstrap metrics stay undefined and their defined draw counts are
reported. Fourteen focused tests cover pairing/identity failures, summary
consistency, resampling paired eyes together, deterministic grouped sampling and a synthetic case where the
aggregate gain masks A/V regression. Ruff and formatting checks passed.

`comparison.json` retains the initial exploratory calculation before the added
summary-consistency gates; its numerical results match the checked result.
