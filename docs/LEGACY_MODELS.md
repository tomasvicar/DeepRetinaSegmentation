# Original trained single_net models

These are the user's original trained models from
`fundus_databases/models/single_net_retina_jednotna_sit`, converted without
retraining or changing weights. They are distinct from the newer one-database
pilot networks. The architecture and original prediction routine come from
commit `7a152c31c2947ddf760716dfa8a6dfc4d2b881c9` of
[DeepRetinaSegmentation](https://github.com/tomasvicar/DeepRetinaSegmentation).

| Converted file in local `models/` | Original run | Output order | Trained tile / overlap border | Parameters | Original indexed images |
| --- | --- | --- | --- | ---: | ---: |
| `legacy-single-net-universal.pt` | `test_best` = `test_13` | vessel, disc, cup | 288 / 36 | 1,304,355 | 3,803 |
| `legacy-single-net-vessels.pt` | `test_vessels_seg_best` = iteration 6 | vessel | 320 / 40 | 1,304,289 | 646 |
| `legacy-single-net-disc-0.pt` | `test_disk_0` | disc | 192 / 24 | 1,304,289 | 3,506 |
| `legacy-single-net-disc-1.pt` | `test_disk_1` | disc | 256 / 32 | 1,304,289 | 3,506 |

The universal checkpoint's saved index contains fourteen databases: ARIA, AVRDB,
CHASEDB1, DRHAGIS, Drishti-GS, DRIVE, G1020, HRF, IOSTAR, ONHSD, REFUGE, RIGA,
STARE and UoA-DR. Its recorded partitions contain 1,921 training, 917 validation
and 965 test images. These are recovered assignments, not a newly generated
split. The index includes available annotation names, including multiple
observers. Having A/V annotations in that index does not mean this three-output
network was trained to classify arteries and veins; that uses the separate
original A/V network.

## Compatibility and portable format

`legacy/single_net.py` retains the actual architecture: linear widths
`[32,64,96,128]`, three convolutions per encoder block, BatchNorm, ReLU and
3×3 transpose convolutions. Registered module names and even unused BatchNorm
parameters remain compatible with the original state dictionary. The historical
forward padding order is preserved; arbitrary photograph geometry is handled by
square trained-size tiles in `tiled_logits`.

The original routine averages **logits** using its convolved border window.
Sigmoid follows blending. The port uses the exact float64 SciPy window and tile
traversal order, caching the reusable window. Averaging sigmoid probabilities
would change the original model's inference and is not substituted here.

Original files contain Python model, Config and Log objects. The one-time
`convert_single_net(original, destination)` operation is specifically for these
trusted user-owned files. Production `load_single_net(converted, device)` uses
`torch.load(..., weights_only=True)` and reconstructs the architecture from plain
metadata. Conversion does not modify the original file or overwrite an existing
converted destination. It does not need the original Python modules installed
at runtime.

The portable payload contains `format_version=1`, `family="legacy_single_net"`,
`architecture`, `state_dict`, canonical `channels`, `preprocessing`, `tiling`,
`provenance`, the actually serialized Config instance fields, and the full
`training_index`. The family distinguishes these originals from the newer pilot
checkpoint format and the separately ported original A/V family.

## Input preprocessing: verified code, uncertain historical profile

The network input is RGB `float32 / 255 - 0.5`; it must not be divided by another
0.5. Original images were processed at 25 pixels per degree using the
source-specific field of view, crop and resize behavior. The compatible
`legacy/preprocessing.py` implements that original loader policy and optional
original local-contrast/CLAHE transformation.

The historical choice between raw and local-contrast-normalized images **cannot
be recovered confidently from these checkpoints alone**. Config stored the
`dataset_fname` as a class attribute, which the pickle did not serialize. The
preserved Config source names `data_25.hdf5`; the later loader script generates
`data_25_normalized.hdf5`. Both versions existed on the old share. The provided
uncommitted diff changes neither of those files after line-ending normalization,
and the actual universal pickle contains neither dataset filename nor a
normalization marker. Saved per-image records contain masks/database/split, with
no image statistics or preprocessing flag.

The portable metadata therefore marks the raw default as **source-inferred** and
sets `normalization_verified=false`. The raw and original-normalized profile must
remain explicit options. A validation-only comparison can assess plausibility;
it cannot retroactively prove the historical training filename. Do not tune this
choice against the saved test partition.

## Evidence and original split exports

`reports/legacy/single_net_conversion.json` records original and converted file
hashes, architecture, actual tile sizes, database counts and numeric checks.
For all four actual original checkpoints:

- Every state-dictionary tensor is unchanged.
- Original-source and ported forward logits match exactly on fixed ramp and
  fixed random inputs at the checkpoint's trained tile size: maximum error **0**.
- The original four-tile prediction routine and ported logit blending match
  exactly on a rectangular input: maximum error **0**.

These checks establish faithful model conversion on the same prepared inputs;
they do not establish a new segmentation accuracy improvement or resolve the
historical raw/normalized profile uncertainty.

Exact image IDs, original annotation names and original assignments are exported
in `reports/legacy/{universal,vessels,disc-0,disc-1}_training_index.json`. They can
be joined to regenerated annotations for future fair comparisons and training.
The original tensors and the raw ownCloud data remain read-only.

Golden unit tests vendor only the exact original `Unet.py` and
`predict_by_parts.py` under `tests/reference/single_net/`, with source commit and
SHA-256 in `provenance.json`. These fixtures are checked before execution, so a
shallow clone needs no historical Git objects to rerun the comparison. Production
imports the cleaned implementation, not the fixtures.
