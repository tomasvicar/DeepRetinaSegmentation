# Integration review — 2026-11-10

Reviewed the shared preparation writer, archive sources, CLI, manifest loader,
training support flags and prediction/evaluation behavior after the adapter
migration. This review did not change those implementation files.

## Verified behavior

- Invalid annotations remain inspectable in prepared artifacts but records with
  `usable_for_training=false` are excluded from training and evaluation loading.
- A disc-only training dataset records `[true,false]` channel support in its
  checkpoint. Reload restores that support. Evaluation excludes the unsupported
  cup channel; prediction writes a zero placeholder and explicitly identifies
  unsupported/untrained probabilities in NPZ and JSON metadata.
- Original format-version-1 pilot checkpoints lacking `supervised_channels` still
  load with all originally trained channels enabled. Independent CPU evaluation
  of two held-out images each from `drive_bce_dice`, `drishti_bce_dice` and
  `rite_bce` retained identical valid-pixel support. Confusion counts differed by
  at most five pixels from saved GPU results; the observed Drishti difference was
  one threshold-boundary pixel. These checks establish compatibility, not a new
  accuracy improvement.
- The actual auxiliary preview contains 108 records from fourteen datasets,
  exactly three source groups per dataset. All 75 referenced copied auxiliary
  resources exist. Five supervised Erlangen raster resources are represented by
  decoded target/valid arrays rather than duplicate original rasters.
- Core DRIVE/CHASE observer records and IOSTAR task records now use consistent
  source-image keys; regression checks verify shared photographs remain shared.
- Source rejects conflicting ZIP canonical copies using size/CRC identity. A
  direct same-size/different-content TAR duplicate reproduction also raises an
  error after the integration fix added duplicate-member content comparison.

## Finding during independent execution

The newly added flat-directory UBMI override regression initially failed while
nine other targeted source/partial-supervision/completion checks passed. A case
directory supplied directly as `--source ubmi=...` indexes its photograph at the
Source root. The adapter's parent became `.`; matching `./...` and requiring an
internal leading `/ImageAnalysis/...` missed its root-relative auxiliary mask.
The reported fix uses an empty prefix for that parent and matches the annotation
folder against a slash-prefixed virtual path. This affects resource preservation;
UBMI remains image-only and does not fabricate training supervision.
