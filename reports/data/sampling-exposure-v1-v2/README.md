# Expected TRAIN sampling exposure: frozen V1 versus V2

**At the same 8,192 samples/epoch, V2 has 18.8% fewer expected selected vessel
opportunities and 20.2% fewer selected conditional-A/V opportunities before crop.**
This is a sampler/exposure result, not proof of an accuracy effect or an actual
delivered-batch count. The running model and protocol were not changed.

## Scope and verification

Read-only Ada CPU1, nice15, idle I/O, CUDA hidden. The complete scan took
26.45seconds, reading 95,574,013 compressed bytes
and decoding 5,694 unique cached TRAIN label-file/task pairs
for all5,700 eligible V2 TRAIN records. Every used compressed file was checked
against its pinned cache-index byte length/SHA. Exact cached array equivalence
comes from the existing cache construction proof; this diagnostic did not rehash
all RGB/cache/canonical bytes. No RGB or VAL/TEST targets were decoded.

Frozen `unified.py` SHA `3260a00bc03b44544556adf5090a412318963fe1b121f7627f265c2fe3aae72d`.
V2 manifest SHA `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`.
Copied immutable V1 manifest SHA `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`.
Cache index SHA `db0ce309ee5ecbf18fa76b7dc8e34dee400b1ea656fff5e99accd204d6dce8ac`.
The V1 manifest was taken from the V2 parent-metadata copy, and every complete
parent row was verified unchanged in V2 before selecting TRAIN rows. This avoids
a second GT scan or dependence on a mutable external parent directory.

The reproducible helper independently derives task flags from target/valid/FOV
using the frozen rules. It then runs only the byte-pinned `sampling_weights` AST
method in a NumPy/stdlib diagnostic context and compares every calculated weight
to it (1e-14 numeric tolerance). No Torch/model/training module is imported.

## Fixed budget, different mixture

| Quantity | V1 | V2 |
| --- | ---: | ---: |
| Eligible TRAIN photos | 2221 | 2529 |
| Eligible annotation records | 5281 | 5700 |
| Mandatory shuffled coverage pass | 2221 | 2529 |
| Weighted replacement extras | 5971 | 5663 |
| Populated source/task keys | 29 | 35 |
| Expected extras per source/task key | 205.90 | 161.80 |

DRIONS adds one disc key, GAMMA two disc/cup keys, IDRiD one disc key, and
BIOMISA two disc/cup keys. Their308 new TRAIN photos increase mandatory coverage,
while the normalizing source/task mass grows29→35. An existing key therefore
receives21.4% fewer weighted extras. Alternate observers do not create extra
photo keys. Exact DRIVE/RITE aliases share one photo and the first sorted record
owns its sampling source (`drive`); RITE still contributes AV annotation bundles.

For photo i, `w_i = sum(1 / N[source,task])` over its available memberships.
`Z=sum(w_i)` equals the number of populated keys. Expected draws are
`1 + (8192 - photo_count) * w_i/Z`. The first term is coverage-first; the second
is the expectation of replacement draws, not an observed draw sequence.

| Selected supervision before crop | V1 draws / epoch | V2 draws / epoch | Relative change | Fraction V1→V2 |
| --- | ---: | ---: | ---: | ---: |
| vessel | 4220.14 | 3426.40 | -18.8% | 51.52%→41.83% |
| av | 1134.27 | 905.16 | -20.2% | 13.85%→11.05% |
| disc | 5664.01 | 6156.16 | +8.7% | 69.14%→75.15% |
| cup | 3514.93 | 3871.28 | +10.1% | 42.91%→47.26% |

Tasks overlap: one photo may train vessels and disc/cup together. These fractions
are separate probabilities for a randomly chosen draw and do not sum to100%.
AV denotes conditional artery/vein supervision, not two independently thresholded
outputs. Every task flag requires actual valid support; absent cup and ambiguous
AV do not generate memberships. Positive-class-presence counts are also retained
in JSON, separately from validity-based eligibility.

Available vascular photos stay514 and AV photos129. Disc availability rises
1991→2299, cup1506→1649. On photos with both binary-vessel and AV records, the
frozen sampler randomly chooses one vascular task, then one observer for it.
Thus available AV photo opportunities2268.53→1810.31 become selected conditional
AV opportunities1134.27→905.16. This factor-of-two choice was already present in
V1. DC observers likewise remain paired; valid-cup availability is averaged over
the actual eligible observer options rather than assumed for every row.

The selected-supervision expectation precedes crop/augmentation. Crops can remove
a particular head even when another head keeps the photo supervised. It does not
count target pixels, gradient magnitudes, foreground difficulty, optimizer steps
or actual successful per-record visits. Those require run coverage/traces; this
diagnostic never substitutes expected values for them.

## Per-source photo draws

Sampling owner is distinct from the source of a randomly selected annotation.
The JSON supplies both owner-based expected photo draws and expected selected
annotation-bundle counts; coannotated photos can contribute multiple bundles.

| Sampling owner | V1 expected photo draws | V2 expected photo draws |
| --- | ---: | ---: |
| aria | 498.79 | 410.60 |
| avrdb | 470.79 | 382.60 |
| biomisa | 0.00 | 368.60 |
| chasedb1 | 221.90 | 177.80 |
| drhagis | 434.79 | 346.60 |
| drions | 0.00 | 271.80 |
| drishti | 451.79 | 363.60 |
| drive | 427.79 | 339.60 |
| erlangen | 268.90 | 224.80 |
| g1020 | 1020.79 | 932.60 |
| gamma | 0.00 | 422.60 |
| hrf | 438.79 | 350.60 |
| idrid_disc | 0.00 | 215.80 |
| iostar | 635.69 | 503.40 |
| onhsd | 258.90 | 214.80 |
| refuge | 811.79 | 723.60 |
| riga | 921.79 | 833.60 |
| rimone | 506.79 | 418.60 |
| stare | 217.90 | 173.80 |
| uoadr | 604.79 | 516.60 |

## Separate future experiment; no live change

Keep V2 files/splits/seed, the8192-sample budget, architecture, losses and schedule
fixed, but predeclare explicit vessel/AV priorities for the replacement mixture.
Calibrate task/source emphasis using TRAIN-only expected exposure, with the V1
vessel/AV opportunity levels as reference targets; publish the resulting disc/cup
tradeoff before training. Preserve all-photo coverage-first and pair each AV
observer vessel input/target consistently. This is a new sampling recipe/source
and seed/RNG lineage, not a silent checkpoint continuation. An independently
declared per-photo/task observer cycle can be tested separately rather than
bundled into the same causal ablation.

An alternative increases the exposure budget, but that is a compute-budget
confound: report actual delivered head/source exposure, steps and wall time and
compare matched budgets as well. No quota automatically guarantees better Dice.
Use validation per-head/source Dice and conditional AV, not only overall macro,
to assess a frozen candidate. Do not choose these priorities from TEST.

Context supplied by the coordinator, not re-evaluated here: V1 selectedE56 had
conditional AV0.79866 and vessel0.76082; an interim V2E25 snapshot had0.74044 and
0.74383 respectively (original conditional AV0.74857). Epoch budgets and other
recipe differences prevent attributing this difference solely to sampling.
The sampler audit establishes a plausible dilution mechanism, not causality.
Final E60 actual results and complete strict contribution coverage remain needed.
Early finite HPO pruning by three-epoch overall macro can favour faster-learning
DC objectives while vascular objectives remain undertrained. That is a protocol
risk for a separately reviewed future experiment; no running selection rule was
modified here.

## Reproduce the diagnostic

This specific comparison uses the frozen prepared artifacts and their existing
verified label cache; it is not a prerequisite for a colleague's RAW-only training
workflow. Supply local paths to those exact artifacts. Choose a new output file.
The V1 parent manifest is the retained metadata copy inside V2.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/audit_sampling_exposure.py \
  --manifest /path/to/corpus-v2/manifest.json \
  --parent-manifest /path/to/corpus-v2/parent-v1-metadata/manifest.json \
  --cache-root /path/to/verified-v2-cache \
  --frozen-unified /path/to/frozen-6b3dfad/src/deepretina/unified.py \
  --expected-manifest-sha256 17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767 \
  --expected-cache-index-sha256 db0ce309ee5ecbf18fa76b7dc8e34dee400b1ea656fff5e99accd204d6dce8ac \
  --expected-frozen-unified-sha256 3260a00bc03b44544556adf5090a412318963fe1b121f7627f265c2fe3aae72d \
  --epoch-samples 8192 --output /path/to/new-exposure.json
```

The exact result contains no Monte Carlo draws; the expectation is analytic once
TRAIN support flags and the frozen weighting function are fixed. Floating-point
last digits can reflect set summation order; they are not additional precision
about actual delivered samples. Full cache fingerprint verification was performed
before these artifacts were used for training and was not repeated by this
lightweight diagnostic; each TRAIN label file used here is nevertheless rehashed
against the explicitly pinned index before decoding.
