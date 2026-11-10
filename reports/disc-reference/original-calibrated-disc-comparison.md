# Strong original reference on the common validation support

Count-only composition: all770 photos, V/AV/cup unchanged exactly, disc replaced only on691 supported photos. No predictions, new containment, or test evaluation.

| Reference | Disc pooled Dice | Disc macro-dataset Dice | Five-head macro-dataset Dice | Role |
| --- | ---: | ---: | ---: | --- |
| Calibrated AV + universal disc | 0.807155014 | 0.579029284 | 0.624398124 | Superseded weak disc baseline |
| Calibrated AV + specialized disc0 | 0.894807908 | 0.729530091 | 0.654498285 | Primary, predeclared macro metric |
| Calibrated AV + specialized disc1 | 0.918482017 | 0.727019255 | 0.653996118 | Pooled-disc winner alternative |

The primary reference uses `legacy-disc-0` with raw legacy25 input, chosen by the metric fixed before the complete native run. `legacy-disc-1` has higher pooled disc Dice and is retained explicitly as an alternative. Do not compare new models only with the weaker universal-disc baseline.

The 691 photo IDs, annotation record lists/sources, valid-pixel counts and positive-GT counts match exactly. Non-disc per-photo, pooled, per-dataset and macro values and AV accuracy/profile are bitwise-equal Python/JSON values. All headers are reaggregated from the same per-photo confusion counts.

These are exposure-qualified original references: historical training-photo crosswalk and normalization provenance remain unresolved; validation selection does not establish independent test improvement.

Authoritative artifacts: `original-calibrated-av-disc0-raw-val.json`, `original-calibrated-av-disc1-raw-val.json`, `composition-verification.json` (baseline/checkpoint/helper/runtime hashes).
