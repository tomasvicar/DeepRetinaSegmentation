# Fresh retraining with native original architectures

`deepretina.legacy_training` trains the exact native single_net or residual A/V
architecture with newly initialized parameters. This is the explicitly named
`native_legacy_canonical_fresh_v1` recipe. It does not reproduce the historical
optimizer, unresolved historical raw/normalized training photometry, old data
exposure or model quality. No original checkpoint tensors are loaded or reset.
Existing original weights, registry aliases and prediction defaults are unchanged.

## Run an explicit fresh training job

Prepare and audit canonical `legacy25_raw`,25px/degree data with fixed TRAIN/VAL/TEST
memberships first using the [RAW-only 21-source workflow](UNIFIED_QUICKSTART.md#prepare-all-21-annotated-training-sources-from-raw).
The basic capped `deepretina prepare` output is not a substitute.
Geometry-policy-bearing V2 preparations retain their policy in
new checkpoints. Use a new run directory outside the immutable prepared directory.
The module is directly runnable; the public `train-legacy` CLI registration uses
the same stdlib-only `legacy_training_cli.add_cli_args`/`train_from_args` API.
Parser construction/help does not import Torch; runtime imports are lazy.

```bash
uv run deepretina train-legacy \
  --manifest data/rebuilt-training21/manifest.json \
  --config configs/native-legacy-vessel.toml --device cuda:0 \
  --output runs/native-vessel
```

For the remaining three native families, choose separate new output directories:

```bash
uv run deepretina train-legacy --manifest data/rebuilt-training21/manifest.json \
  --config configs/native-legacy-disc.toml --device cuda:0 --output runs/native-disc
uv run deepretina train-legacy --manifest data/rebuilt-training21/manifest.json \
  --config configs/native-legacy-universal.toml --device cuda:0 --output runs/native-universal
uv run deepretina train-legacy --manifest data/rebuilt-training21/manifest.json \
  --config configs/native-legacy-av.toml --device cuda:0 --output runs/native-av
```

These are independent jobs; the examples are alternatives or sequential runs,
not a recommendation to launch all four on one GPU simultaneously. Substitute
`--device cpu` for a CPU-only machine. The native A/V model trains only eligible
A/V-labelled sources; it does not invent A/V supervision for the other sources.

The four production presets are `native-legacy-vessel.toml`,
`native-legacy-disc.toml`, `native-legacy-universal.toml` and
`native-legacy-av.toml`. Defaults use CPU,60epochs, Adam(lr0.0001,weight_decay0),
seed42 and batch1. Device/epochs/batch size/lr/seed can be explicit CLI overrides;
AV always requires batch1 and rejects a patch size. `--expected-content-sha256`
can additionally pin the complete prepared fingerprint. There is no resume,
pretrained initialization, historical teacher, cache or TEST scoring option.

With `epoch_samples` omitted, each epoch includes enough photo visits to cycle
every eligible observer bundle once. An explicit sample budget must include this
whole coverage-first pass; extras sample eligible photos. All presets omit an
architecture override. The programmatic `architecture` override is explicit and
recorded as a custom architecture, intended for small CI checks; such checkpoints
must not be described as the verified original production specification.

## Exact architectures and outputs

| Task | Fresh native constructor | Outputs | Training patch |
| --- | --- | --- | --- |
| vessel | `single_net.Unet(filters=32,in_size=3,out_size=1,do=0,depth=4)` | vessel |320×320 |
| disc | same, `out_size=1` | disc |192×192 (disc0 metadata) |
| universal | same, `out_size=3` | vessel,disc,cup |288×288 |
| av | `LegacyAVNet(encoder_channels=[4,32,64,128,256],decoder_channels=[256,128,64,32],head_channels=128,num_classes=3)` | unused logit0,artery,vein | full canonical frame |

These specifications and state-dictionary shapes are tested against the safe
converted bundled metadata on the meta device. Native layer names, BatchNorm,
forward padding, residual blocks and logit counts are preserved. The universal
model has three native outputs. It does not receive a replacement five-head decoder.

Single_net input is canonical RGB/255−0.5. Patches must be divisible by16; small
images use the native tiler's reflect/replicate image extension with invalid target
support outside the image. The original constructor/forward handles its own native
padding. The new loss averages masked BCE+soft Dice over selected heads with known
support; unknown heads have zero gradient. Validation uses native `tiled_logits`,
then the same masked loss over the eligible full prepared photograph.

## Native A/V recipe

The saved profile is **`legacy25_raw_av4`**. Inputs are full canonical RGB/255−0.5
plus `(same-observer GT vessel>=0.5) & vessel-validity`. Invalid/outside-FOV vessel
pixels become0. Ambiguous vessels and crossings remain in the input vessel mask;
they do not enter the class loss. The model retains its original internal
per-image/channel sample-standardization without epsilon. Exactly constant RGB or vessel
channels (detected by equal channel minimum/maximum, including float32 constant0.1), nonfinite inputs and frames smaller than16 are rejected with the record
ID before outputs/device setup. Arithmetic is not changed to accommodate them.

A/V training uses batch1 and the full prepared frame. It performs no patch crop,
letterbox, external pad or resize. Native output interpolation restores the input
spatial dimensions. Only logits1:3 enter CE on valid known artery XOR vein pixels;
logit0 and unknown/ambiguous pixels receive zero class-loss gradient. Each selected
A/V observer supplies its own vessel and artery/vein targets. Vessel-only rows
cannot become A/V samples, even when they share a photograph.

Deployment requires a separately supplied vessel mask and the new saved-profile
prediction route. GT-vessel training versus an imperfect deployment vessel mask is
a distribution difference to measure. No historically exposed vessel model is
silently used for retraining or selected as a new inference default.

## Predict a freshly trained checkpoint

Native vessel/disc/universal checkpoints use the regular prediction command:

```bash
uv run deepretina predict --checkpoint runs/native-vessel/best.pt \
  --input /path/to/fundus.jpg --dataset drive --fov /path/to/fov.png \
  --output predictions/native-vessel --device cpu
```

The checkpoint carries its declared raw profile and any portable camera policy.
`--normalized` / `--no-normalized` explicitly override single-net normalization;
the override is recorded. Older converted checkpoints keep their existing raw
default and historical-provenance qualifications.

A retrained AV checkpoint uses its saved `legacy25_raw_av4` profile automatically.
Supply the original-frame vessel mask from your separately chosen segmentation:

```bash
uv run deepretina predict --checkpoint runs/native-av/best.pt \
  --input /path/to/fundus.jpg --dataset drive --fov /path/to/fov.png \
  --vessel-mask predictions/native-vessel/vessel.png \
  --output predictions/native-av --device cpu
```

No vessel model is selected implicitly for this new recipe. RGB and the vessel
mask receive the same canonical crop/rescale; input vessel support is limited to
the FOV. The full crop enters the unchanged network, then nearest classes and
vessel support return through the recorded crop geometry. Conditional
probabilities return linearly, are gated by restored support and sum to one on
that support. Outputs include `classes.png`, artery/vein masks, overlay,
probabilities, vessel support and explicit geometry/profile metadata.

Historical `--av-raw-*`, DICOM and historical prepared-input modes are rejected
for this newly trained profile. Those modes remain available for the original
v9 checkpoint. This separation prevents applying incompatible photometry or
stretching a canonical crop across the original photograph.

## Eligibility, coverage and checkpoint selection

Eligible TRAIN/VAL rows and observer bundles are filtered for the selected heads
before photo sampling/crop construction. Original source/photo split-leak checks,
usable/exclusion flags and full preparation metadata are retained. Different-task
bundles on one photo merge only their actually valid selected supervision. Within
a task, sorted observer bundles cycle rather than mixing their vessel/class targets.
A single_net crop is centred on known selected-head validity; delivered annotation
IDs count only records with selected support remaining in that crop.

Coverage records actual eligible/seen photos, eligible/contributing annotation
IDs, positive per-photo/record visits and explicit all-eligible flags after successful
optimizer updates. A selected record that contributed no selected-head support is
not counted. The full observer pass is scheduled, while actual crop support and
contribution counts remain separate evidence. Heldout target arrays are never
loaded by the training/validation sample reader. The initial canonical byte audit
hashes the complete prepared artifact for identity; this is not TEST decoding or
model evaluation.

VAL visits the same per-photo observer cycle and averages masked loss across its
slots. If one task has fewer observers than another on a coannotated photo, its
observers repeat cyclically within that slot cycle. Best checkpoint selection
minimizes this explicitly defined new VAL loss; no threshold or TEST result is used.
The recipe applies no augmentation beyond seeded valid-support crop selection.

`best.pt` and `last.pt` use existing safe `legacy_single_net`/`legacy_av` families,
architecture/state dictionaries, native channels and tiling. Both carry the new
recipe/config, fresh lineage, source-byte hash, canonical manifest/content pins,
optional validated embedded geometry policy, seed, actual coverage and epoch/steps.
Optimizer and RNG states are retained for inspection; no cross-source resume claim
is made. Checkpoint publication is atomic within the new owned run directory.
`run.json`, `coverage.json`, `metrics.csv` and `training_summary.json` retain the
new recipe and actual completion evidence. Loaders use `weights_only=True`.

Functional CPU tests run actual synthetic Adam updates and full-frame VAL for all
four families, verify old-tensor loading is forbidden during training, reload safe
checkpoints and produce finite native logits. Production default metadata/shapes,
masked gradients, observer pairing, coverage, heldout isolation, constant-channel
rejection and geometry-policy capture are checked. No real corpus, GPU training,
TEST scoring, active frozen source or bundled model is modified by this change.
