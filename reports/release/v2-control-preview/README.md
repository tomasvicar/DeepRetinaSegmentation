# Draft release ownership: unified-v2-control

**Not a completed-model announcement.** This branch is prepared from stable
`42aa3e3`. It currently adds no checkpoint, registry entry, alias or changed default.
The exact nine current PT files and original aliases are pinned in the adjacent
`draft-plan.json` and have been read-only SHA-verified.

The intended addition is a separately named `unified-v2-control`: one five-output
RGB **21-source corpus-v2 control baseline**. Expected training support is2,529
photos /5,700 annotation records, on the fixed6b source and17a/00d9/2591/18c data,
geometry and cache identities recorded in the draft. These expected counts do not
stand in for actual terminal coverage or completion evidence.

Wait for the parent's immutable actual E60 best artifact, SHA256, strict terminal
QA, selected epoch, training selection score and fresh validation report before
copying weights or completing model metadata. In particular, mutable `best.pt`,
unknown hashes, an E58 snapshot and pseudo-complete placeholders are not eligible.
The fresh report must keep source/head/conditional-AV tradeoffs visible; a source
macro increase alone cannot establish an all-task improvement.

The existing checksum resolver and unified-family dispatcher already support the
new name once a complete registry entry is added. Wheel/sdist rules include all
pretrained files; no runtime dispatcher change is planned. The exact-set bundled
model test will need a tenth name and a model-specific E60/selected-epoch check.
No full suite or real forward has been run during this draft stage.

Public release evidence will contain model/data/source/geometry checksums, actual
training CSV/summary/QA and validation qualifiers. It will not package private
worker/PBS/Meta endpoints, upload receipts, raw images or ground-truth masks.
The V1 registry's stale wording about additional sources being "under audit" can
be clarified at completion: they are prepared in separate V2 and were never used
by that V1 checkpoint. The original model aliases and threshold0.5 stay intact.

Planned post-QA colleague example (not executable before registration):

```bash
uv run deepretina predict --model unified-v2-control \
  --input /path/to/TRAIN/drions-image001.jpg --dataset drions \
  --output predictions/unified-v2-control --device cpu
```

The CPU release smoke will first prove RAW photograph membership in the fixed
TRAIN corpus, then check original-frame output geometry, finite probabilities,
A/V exclusivity/subset of vessel and cup containment. It will not score GT.
DRIONS30° is an explicitly documented nominal research assumption carried by the
portable policy; the example does not call it a measured camera calibration.

No optimized/final, clinical, heldout-TEST or superiority claim follows from this
preview. The five originals, V1 baseline and three pilots remain available under
their existing names.
