# Epoch1 validation-only normalization probe

Checkpoint: `runs/unified25/baseline-lr001-seed42/best.pt`, immutable executable
source87cf045. The complete machine-readable paired results and checkpoint hash
are in `probe.json`; local ignored `overlays.png` shows predictions/reference
outlines. No heldout test arrays or metrics were opened. No weights, source files,
prepared data or BN buffers in the original model were modified. GTX1070 timed
pipeline3.29seconds; no optimizer updates. This is a causal diagnostic, not a
quality-improvement result or a replacement inference policy.

Twelve validation photographs were selected before outcomes: two spaced sorted
photos each from REFUGE and G1020; one each from Drishti, RIM-ONE, ONHSD, RITE,
HRF, IOSTAR, ARIA and AVRDB. Both variants use the same512tile/64overlap pipeline,
RGB/255−0.5 normalization, threshold0.5, hierarchy and reference support. The
comparison model is a deep copy whose BN running means/variances are replaced
withNone and tracking disabled while the rest stays in eval mode. It therefore
uses per-tile batch/spatial statistics (batch1), not fitted GroupNorm or calibrated
population estimates. The checkpoint SHA and original BN buffers were checked
unchanged afterwards.

| Mean available per-photo Dice | Stored running stats | Temporary tile stats |
|---|---:|---:|
| Vessel |0.447|0.601|
| Artery |0.055|0.049|
| Vein |0.589|0.582|
| Disc |0.450|0.485|
| Cup |0.214|0.421|

These averages are small-subset per-photo diagnostics, not the full validation
macro-dataset objective. Responses differ substantially by photo/domain:

- First REFUGE: disc0.187→0.425, cup0.067→0.283; enormous disc prediction reduces182013→69624pixels against18799reference. False positives remain.
- Second REFUGE: disc0.832→0.426, cup0.546→0.191; temporary normalization worsens a good prediction.
- Drishti: disc0.084→0.692, cup0→0.800; RIM: disc0.204→0.620, cup0→0.665. Stored stats missed these structures almost entirely.
- HRF vessel0.306→0.668 and ARIA vessel0.295→0.582; RITE vessel0.695→0.671. Other disc examples also worsen under temporary stats.

Four annotation-centered384train patches were checked without augmentation or
updates. Total masked objective running→temporary: RITE0.742→0.720,
REFUGE0.580→0.408, RIGA0.417→0.377, UOA0.636→0.840. GT/prediction overlays
and channel support are spatially coherent; RIGA disc/cup and UOA disc-only
supervision follow their intended semantics. This spot-check did not identify a
channel swap or image/target geometry bug. It does not rule out every data issue.

Full epoch1 AV conditional accuracy0.5618 is only modestly above the always-vein
reference proportion0.5445 (1430060vein vs1196224artery labeled validation pixels).
Training reference counts are3703227artery/4342546vein, not an extreme class
imbalance. The temporary-stat probe does not solve the artery collapse; the AV
head remains weak at this early checkpoint.

Interpretation: mixed-domain/crop-to-tile BN statistics contribute to inference
instability, while substantial undertraining remains plausible. A separately
frozen, from-scratch `norm=group` candidate with identical data/budget/selection
objective is justified. Keep the original BN baseline and evaluate learning curves;
do not transfer this probe's temporary stats as a claimed validated improvement.
No trained GroupNorm quality conclusion follows from changing BN at inference.
