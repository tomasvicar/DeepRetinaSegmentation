# Colleague workflow audit of for_Leipzig f1c4c99

## Verified usable paths

A colleague with the repository and the shared RAW `fundus_databases` folder now
has a documented path through uv installation, RAW-only 21-source preparation,
patient/alias audit, five-head unified training/resume, and raw prediction.
No opaque author-prepared directory is required. The historical fingerprints
are retained as historical evidence, not promised for an arbitrary fresh build.

All nine bundled model files exist and match their registry SHA256 values:
five original checkpoints, three named pilots, and the completed corpus-v1
unified baseline. Original aliases remain separate from new models. All four
native retraining presets exist (vessel, disc, universal, A/V); original topology
and the explicit new canonical training recipe are distinguished from historical
training replay. Retrained A/V prediction requires supplied vessel support and
its own saved input profile, as documented.

## Concrete documentation corrections

- Installation now supplies the actual repository clone/branch and working-directory steps.
- The raw A/V options now consistently describe three mutually exclusive profiles.
- `DATASETS.md` and the README explicitly scope `max_size`/complete-frame/no-angle
  geometry to the basic pilot converter. Canonical native/unified trainers must
  use the separate `legacy25_raw` physical-grid preparation.
- Native retraining links directly to the RAW-only preparation step and gives
  concrete commands for all four presets with distinct output paths. It does not
  imply four simultaneous jobs on one GPU.

## Bounded evidence and remaining limits

The adjacent JSON records 13 actual parser checks and all nine file hashes.
Parsing included native/unified training, unified resume, capped preparation,
inspection, original prepared A/V, source-auto A/V and new native A/V prediction.
Preset task values were checked. These were stdlib-only checks: no Torch import,
command dispatch, image decode, model forward, preparation, training or test
execution. This audit does not add a clean-machine end-to-end quality claim.

Retained practical limitations are explicit: exact supplied archive layouts are
required; full preparation may take hours and keeps an intermediate independent
parent; a failed build needs fresh output paths; native training currently has
no resume/cache support; native A/V uses GT vessel input during training and a
supplied segmentation at deployment; nominal-camera and historical photometry
uncertainty remain. Canonical preparation is composed from previously tested
steps, not demonstrated to regenerate a historical fingerprint from arbitrary
RAW versions. Cross-platform copy fallback exists, but full Windows/macOS package
and GPU installation were not certified here. None of these qualifications is
silently converted into a model-quality or historical-reproduction claim.
