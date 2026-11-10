# Calibrated original A/V validation refresh

`scripts/refresh_original_av.py` changes only A/V preprocessing in a completed
raw-profile original reference. The universal disc/cup and vessel metrics remain
from the original full report. On every photo with A/V annotations, it reruns the
same dedicated vessel weights and rejects any mismatch in TP/FP/FN/valid counts
before refreshing the A/V predictions. Source RGB identity and the complete
prepared-content fingerprint are checked again.

The new profile uses the canonical FOV restored to the original frame, the
MATLAB-compatible generic provided-mask center/radius estimator, scalar bicubic
resize and the separately validated Python photometric stages. It does not
recover an unavailable historical native FOV or establish old training-file
provenance. Numerical residuals are in [MATLAB_AV_PROFILE.md](MATLAB_AV_PROFILE.md).

Start with a small CPU canary:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv run python scripts/refresh_original_av.py \
  --reference reports/unified/original-strong-raw-val.json \
  --manifest /path/to/unified25/manifest.json \
  --data-root /path/to/ownCloud/fundus_databases \
  --vessel src/deepretina/pretrained/legacy-single-net-vessels.pt \
  --av src/deepretina/pretrained/legacy-av-v9.pt \
  --threads 4 --limit 2 --output /tmp/av-calibrated-canary.json
```

The limited report is explicitly a preview and contains only processed A/V
photos; it cannot stand in for a full five-head comparison. After the canary
passes, omit `--limit` to refresh every validation photo with A/V support and
reaggregate all heads and annotation sources. Output creation is exclusive;
earlier reports are preserved. No test split is accepted and no GPU is used.
Empty/full vessel inputs remain undefined for the original standardization and
are disclosed as unscored conditional cases while counting end-to-end misses.

Six focused tests verify exact preservation of other heads, source aggregation,
conditional versus end-to-end accounting, and rejection of missing/duplicate or
mismatched A/V ground truth. The actual two-photo CPU canary passed in 10.67s,
including exact reproduction of original vessel counts. Its
[report](../reports/unified/original-calibrated-av-canary.json) retains geometry
and A/V confusion counts. The full refresh started on Ada CPU at 13:50:58 UTC;
[launch provenance](../reports/unified/original-calibrated-av-val.launch.json)
records the immutable source, command and pins. It must complete before this
profile supplies a full reference score.
