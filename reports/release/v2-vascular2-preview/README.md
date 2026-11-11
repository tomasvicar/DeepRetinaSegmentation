# Experimental vascular2 E59 preview release evidence

Model ID: `unified-v2-vascular2`. These are unchanged audit-approved BEST E59 bytes,
SHA256 `81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b`,
5,453,234 bytes. The selected snapshot already has all 2,529 TRAIN photos and all
5,700 annotation contributors, with minimum visits 64/2. The run completed 60 epochs;
E59 is validation-selected and must not be called a terminal60 model.

The model is an experimental full-data preview, not final optimized, all-task
superior or TEST-qualified. Fresh VAL macro 0.8038980942 and conditional A/V 0.810257
coexist with a vessel regression against the strongest original reference
(0.756713 vs 0.770123). The model card discloses these tradeoffs and the shared
validation/selection qualifications.

- `model-qa.json`: safe CPU weights-only load, actual checkpoint/config/coverage.
- `standalone-completion-summary.json`: unchanged public typed completion summary
  from audit commit 1a2ea5f, retaining original strict QA SHA and separate best/last roles.
- `fresh-validation-summary.json` and `paired-comparison-summary.json`: compact
  metadata from that commit's actual reports; full source-report SHA values retained.
- `real-cli-smoke.json`: actual CPU 1 public CLI on verified DRIONS TRAIN image 001,
  without GT decoding/scoring. Original geometry, finite probabilities, A/V
  exclusivity/subset and cup containment passed, all output paths existed.
- `preserved-models.json`: all ten previous checkpoint bytes, registry entries and
  original aliases unchanged.
- `package-check.json`: final uv-built wheel with all eleven exact model SHA values;
  45 focused bundled/scalar CPU tests passed. No new training or fresh validation
  forward was performed for this release preparation.

The source implementation only adds registry/test/documentation entries and the
approved weight file. The earlier scalar compatibility was independently merged
into stable. No training/inference implementation, thresholds or default aliases
are changed here. HPO and final candidate selection remain separate.

Full source validation evidence:
[public paired audit](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/1a2ea5f/reports/vascular2-paired-val).
Raw photographs, masks, overlays, private operational receipts and Meta endpoints
are not included in this release evidence.
