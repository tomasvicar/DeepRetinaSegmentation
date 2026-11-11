# Six predeclared TRAIN-only sampling scenarios

These are analytic exposure scenarios, not trained candidates or measured quality.
No winner is selected. Production sampling, data and running source remain unchanged.

## Definitions

- All scenarios retain V2 TRAIN photos/labels, 8,192 samples and one shuffled
  coverage-first visit to every2,529 photos, leaving5,663 replacement extras.
- For existing populated source/task memberships, multiply each `vessel` or `av`
  key by λ∈{1,1.5,2}; every disc/cup key retains multiplier1. A photo weight is
  the sum of these weighted inverse-population memberships. Apply this change
  only to the replacement mixture; it does not alter the mandatory photo pass.
- When a photo has both binary-vessel and AV task bundles, choose the AV bundle
  with probability p∈{0.5,0.75}, otherwise choose binary-vessel. With only one
  vascular task, retain its probability1. Within each chosen task, observers remain
  uniformly selected. Disc/cup is a separately paired bundle as before.
- All probabilities/expected counts precede crop, augmentation and successful
  optimizer-update filtering. They are not target-pixel counts or actual visits.

The total key mass is `Z(λ)=15λ+20`. Given vascular contribution `A_i` and
DC contribution `D_i`, expected draws of photo i are
`1 + 5663*(λ*A_i+D_i)/(15λ+20)`. Expected head supervision multiplies this by
the probability that the selected observer/task has valid support for that head.
Task events overlap; their counts need not sum to8192.

## Outcomes

| λ | p(AV when both) | Vessel | Conditional AV | Disc | Cup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 3426.40 | 905.16 | 6156.16 | 3871.28 |
| 1 | 0.75 | 3426.40 | 1357.73 | 6156.16 | 3871.28 |
| 1.5 | 0.5 | 3911.80 | 1069.65 | 5731.26 | 3479.12 |
| 1.5 | 0.75 | 3911.80 | 1604.47 | 5731.26 | 3479.12 |
| 2 | 0.5 | 4251.58 | 1184.79 | 5433.83 | 3204.60 |
| 2 | 0.75 | 4251.58 | 1777.18 | 5433.83 | 3204.60 |

**V1 reference at the same8192 samples:** vessel4,220.14; AV1,134.27;
disc5,664.01; cup3,514.93.

| λ | p | Vessel change vs V1 | AV change vs V1 | Disc change vs V1 | Cup change vs V1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | -18.8% | -20.2% | +8.7% | +10.1% |
| 1 | 0.75 | -18.8% | +19.7% | +8.7% | +10.1% |
| 1.5 | 0.5 | -7.3% | -5.7% | +1.2% | -1.0% |
| 1.5 | 0.75 | -7.3% | +41.5% | +1.2% | -1.0% |
| 2 | 0.5 | +0.7% | +4.5% | -4.1% | -8.8% |
| 2 | 0.75 | +0.7% | +56.7% | -4.1% | -8.8% |

Increasing λ moves photo draw exposure towards vascular sources, with a DC tradeoff.
Atλ2,p0.5, the expected vessel/AV counts are close to/slightly above the V1
reference, while disc and cup counts fall below it. This is an exposure observation,
not a preferred model or a quality prediction. Increasing p alone recovers more AV
opportunities while leaving vessel draw counts unchanged in this corpus: both
vascular bundle types supply valid vessel support. It can still change which
vessel annotation, target pixels and background support enter training.

All six retain the four added DC sources and one photo visit per epoch; lower
extra exposure is not removal of their annotations. Random observer selection
does not guarantee visits to every annotation by a fixed terminal epoch. Strict
completion must continue to use actual contribution receipts, not these expectations.

## Sampling is not loss weighting

Increasing an AV loss multiplier acts only when that objective is already sampled.
It does not create additional observer/domain visits, recover a crop that missed
valid AV pixels, or change which image reaches BatchNorm. Vascular sampling changes
the image mixture and all jointly available objectives on those images, including
disc supervision coannotated with vessels. Changing p alters annotation choice and
potentially vessel target/support density. Neither operation is generally equivalent
to multiplying one loss term; neither is guaranteed to improve Dice or accuracy.

A separately authorized experiment would freeze a new recipe/source/seed lineage,
publish one sampling policy before training, retain the same compute budget, and
measure actual head/source visits plus validation per-head/source and conditional
AV tradeoffs. The grid was predeclared in response to earlier validation concerns;
this diagnostic computes no VAL/TEST scores and selects no parameter or model.
Nothing is promoted from this table.

## Evidence and reproduction

The fixed reference report SHA is `271f79a983e79628bd3d6073443674d63576fed5f5370cc47c81b8aa7055a6db`.
Before emitting results, the helper required identical input pins and exact equality
of all original V1/V2 task expectations, owner/source tables and selected annotation
source counts with that retained report forλ1,p0.5. This check passed. Every scenario
also checked the underlying unmodified base weights against the byte-pinned frozen
method before applying its research-only multiplier.

Actual scan: 26.61seconds on Ada CPU1/nice15/idle-I/O/CUDA-hidden;
5,694 unique cached TRAIN label-file/task decodes,
95,574,013 compressed bytes rehashed against the pinned index.
No per-photo support flags were retained by the earlier report, so the verified
TRAIN cache was scanned once more; all six scenarios reused that one in-memory
flag table. No RGB, VAL or TEST targets were decoded. No Torch/model or training
execution occurred. Three tiny analytic CPU tests passed, including exact default
scenario reproduction, separate effects of λ and p, and unknown-label exclusion.
The existing global canonical/cache identity audit was not repeated in bulk.

Use the same pinned-input command in
[the original audit](../sampling-exposure-v1-v2/README.md#reproduce-the-diagnostic),
with a fresh output path and these additional arguments:

```bash
--scenario-grid \
--reference-report /path/to/original-exposure.json \
--expected-reference-sha256 271f79a983e79628bd3d6073443674d63576fed5f5370cc47c81b8aa7055a6db
```

The flags extend only `scripts/audit_sampling_exposure.py`, an analysis helper.
They are not training CLI settings and do not implement a production sampler.

## Shared verified TRAIN task flags

`taskflags.json` retains the last TRAIN-only scan's reusable per-record task
availability, with manifest/index/frozen-source pins. It contains5,700 records
and no images, masks, coordinates, logits or GT arrays. Exact file-byte SHA256:
`62a4dd1497b8455ac499952b1191fd7fee1e72f4913f8c2ae492ced088049c17`.
The adjacent sidecar and provenance record pin the file, support-only semantics,
CPU/IO limits and final26.96-second scan. This was the last repeated label scan;
further sampler metadata checks should reuse these flags and the pinned manifest.

The research helper can produce this artifact with `--task-flags-output PATH`.
The output is exclusively created, tied to the same input byte pins, and restricted
to eligible TRAIN IDs. A fixture includes a nonexistent heldout mask to ensure
that export does not decode it. Metadata export does not update any production
sampler or model. Grouping/photo/source identity still comes from the byte-pinned
manifest, not inferred from record-name prefixes.
