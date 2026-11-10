# Residual A/V paired trial: root-review draft

The adjacent plan-av-residual-cached-draft.json is deliberately **not runnable**:
its purpose differs from full_data_validation_search and the existing runner
rejects it before training. After root review/QA, publish a new separately hashed
approved plan with the runner's standard purpose and approved provenance. Keep
this draft unchanged as evidence; no GPU trial has been launched here.

Source candidate:3157bb957fcf6e6f632a5c49d6df2c8a78ea3747.
Execution code SHA256:8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b.
Training-source SHA256:dd69012a1741000ae5841634094094397c690a2371c9a91aa411344a436f1136.
The code digest was computed independently from those exact Git blobs and matched
the populated /tmp source tree. Reports/tests/docs do not alter that code digest.

Control config is copied from the actual old lr_0003-seed 42/run.json (path and
snapshot SHA retained in the draft), not reconstructed defaults. It matches the
historical plan except the actual device cuda:0 replacing auto. The residual
config differs from that control in exactly one field: architecture. Both retain
lr 0.0003, seed 42, base 12/depth 4/BatchNorm, all loss/sampling/augmentation settings,
8192 epoch samples, batch 4/patch 384 and the original cosine horizon of 60 epochs.
Canonical identity 61032be...1693e and full cache 4e39e40d...aa8591 match the reviewed
cached-width16 plan; the exact runtime cache path and full pins are in JSON.

Both trials start fresh in separate new outputs under the same new immutable
source and approved plan. First stop each at epoch 8 without shortening epochs 60.
After E8 control QA, resume that fresh control directly to 60 (--resume --final),
keeping config/scheduler/cache/source identical. Residual continuation remains a
separate validation decision. Historical E8 metrics are contextual, not a
replacement for this fresh matched-source control. Record actual device/thread/
environment policy for both trials; do not invent old launcher thread counts.

Illustrative commands only AFTER approval, source checkout and cache-gate QA:

```bash
python /ABS/NEW_IMMUTABLE/scripts/unified_hpo.py run \
  --plan /ABS/APPROVED_PLAN.json --trial lr_0003_control_cached \
  --output /ABS/FRESH_CONTROL --derived-cache /ABS/FULL_CACHE --stop-after-epoch 8
python /ABS/NEW_IMMUTABLE/scripts/unified_hpo.py run \
  --plan /ABS/APPROVED_PLAN.json --trial lr_0003_av_residual_cached \
  --output /ABS/FRESH_RESIDUAL --derived-cache /ABS/FULL_CACHE --stop-after-epoch 8
# After control E8 QA, continue the SAME fresh control to the planned 60 epochs:
python /ABS/NEW_IMMUTABLE/scripts/unified_hpo.py run \
  --plan /ABS/APPROVED_PLAN.json --trial lr_0003_control_cached \
  --output /ABS/FRESH_CONTROL --derived-cache /ABS/FULL_CACHE --resume --final
```

No checkpoint migration, source87 modification or test evaluation. Architecture
quality and speed improvement are hypotheses, not established results.

Final CPU verification of the reviewed source plus interface tests: **563 full-suite tests passed**, 58 existing warnings. No pytest cacheprovider, CUDA use or external training-source writes.
