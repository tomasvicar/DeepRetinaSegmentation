# Full original VAL after the MATLAB nearest-coordinate correction

Actual source `d14aa14` completed all 770 VAL photographs on 2026-11-11
01:37:51 UTC, return code 0, 3,247.29 seconds. No TEST evaluation. The only production
source change relative to 193662c is `legacy/matlab_resize.py`; four original
checkpoint hashes, evaluation-script bytes, runtime versions, canonical manifest,
annotation inventory and valid/positive GT count support remain identical.

Whole `validation.json` SHA256:
`182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836`.
The historical 193662c report is retained separately with SHA 53b5c0a6….

## Measured effect

- Vessel, disc and cup counts are unchanged on all 770 photos.
- Only two HRF photos have changed A/V counts: four photo/head count pairs.
- Conditional A/V has 18 fewer correct pixels out of 2,626,284 known pixels;
  end-to-end correct pixels decrease by 25. The scoring domain is unchanged.
- Macro dataset/head Dice changes 0.6670315865775472→0.6670295852091139
  (difference−0.00000200137).
- Conditional A/V accuracy changes 0.7485732693037006→0.748566415513326.

This correction improves independently verified MATLAB nearest-operator fidelity.
It is **not a measured predictive-quality improvement**. The tiny negative VAL
change is retained explicitly; no threshold or model was tuned to reverse it.
The remaining raw photometry residuals and historical exposure limitations still
apply. Any final original policy must bind the report/source actually used.

`count-differences.json` retains the exact changed photo/count records.
`paired-comparison.json` uses the unchanged generic comparison, 2000 draws/seed42,
612 source-stratified split groups, all head/source intervals and conditional/end-to-end A/V.
These conditional intervals do not establish independent patients or seed uncertainty.

## Reproduce the metadata comparison

```bash
uv run python scripts/compare_predictions.py \
  --reference reports/frozen-original-val-reproduction/validation.json \
  --candidate reports/frozen-original-val-nearest/validation.json \
  --manifest /path/to/verified/corpus-v2/manifest.json \
  --draws 2000 --seed 42 --include-sources --include-av \
  --output /path/to/new-comparison.json
```

The two optional CLI flags expose existing generic source/A/V reporting; default
behavior is unchanged. The command reads saved reports and manifest JSON only,
never models or image/GT arrays. Existing outputs are preserved. Actual production
of these full VAL reports is separate from this metadata-only comparison.
