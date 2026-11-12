# Testing and verification

This is the test inventory for product `for_Leipzig` at
`cb507d1fbc7facec4ab86ced137a3e4885c39e24` (12 November 2026).
**No entire pytest suite, model forward, training, GPU operation or dataset
decoding was rerun for this inventory.** The separate34-case mocked defaults
module check is recorded below. Historical PASS records certify their own
source snapshots and scopes, not an unmeasured entire latest checkout.

## Reproduce checks

From the repository root, install the locked development environment once:

```bash
uv sync --frozen --dev
# CPU-focused execution; some cases really optimize tiny synthetic networks.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
```

These are **commands to run**, not newly observed results. Use a writable temporary
directory with sufficient space; an owned HOME-backed `TMPDIR` can avoid a small
shared `/tmp`. Do not run the whole suite on a busy workstation assuming it only
reads metadata: bundled-weight cases deserialize real checkpoints; several cases
perform small CPU forwards/training, write temporary prepared data/checkpoints,
or launch owned child processes to check termination cleanup. Environment
variables above hide CUDA for ordinary CPU checks; the documented historical GPU
functional checks below were separate explicitly bounded runs.

The per-module commands below assume `uv sync --frozen --dev` already succeeded.
`--no-sync` avoids silently changing that environment during a selected check.
To inspect pytest's actual parametrized collection, use
`uv run --no-sync pytest --collect-only -q` (imports test modules; not performed
for this inventory). A specific test can be selected with
`uv run --no-sync pytest -q tests/test_legacy_av.py::test_state_dictionary_roundtrip_preserves_logits`.

### External inputs and skip boundaries

Most fixtures are synthetic or vendored reference data. The original-v9 conversion
fixture in `test_legacy_av.py` optionally reads the owned historical pickle from
`DEEPRETINA_AV_CHECKPOINT` (default author's ownCloud model path); **two dependent
tests skip** if absent. Do not substitute an untrusted pickle. Other bundled
checkpoint tests use the safe weights distributed in the repository. Check the
pytest skip summary (`-rs`) rather than interpreting an omitted external-input
case as numeric parity. `cv2` is also checked with `importorskip` in one case;
it is included in the locked project dependencies. Historical torchvision crop
verification is documented in [LEGACY_AV.md](LEGACY_AV.md); ordinary tests use
the pinned vendored crop adapter, not a mandatory torchvision runtime.

No MATLAB executable is needed to run MATLAB-oracle tests: committed synthetic
fixtures and reference provenance are read by Python. Tests/reference modules are
**oracles/helpers**, not additional independently collected `test_*.py` modules.
Generating new MATLAB goldens is a separate development activity.

## Every test module

Counts below are **AST test definitions**, not PASS totals and not expanded
parametrized cases. The machine-readable [inventory](../reports/testing-inventory-20261112/inventory.json)
lists every definition, line, decorator and exact test-file SHA. Discovery scans
**all172 Git-tracked Python files**, including scripts/reports/reference modules,
for module/class `test*` definitions regardless of filename or class prefix.
No additional nonstandard test-definition modules were found in this product
snapshot; all62 discovered modules are under `tests/test_*.py`. This exposes
individual test scope and parametrization without running imports. Regenerate
it from the repository root with stdlib Python:

```bash
python reports/testing-inventory-20261112/discover.py > /path/to/new-inventory.json
```

The recorded inventory binds the snapshot named above. After test-source changes,
review a new inventory and its source identity rather than silently replacing the
old evidence. The discovery script does not execute tests or claim collection
success.

Total: **62 modules, 451 definitions**.

| Module | Definitions | Purpose | Reproducible command |
| --- | ---: | --- | --- |
| [test_adapter_coverage.py](../tests/test_adapter_coverage.py) | 3 | Prevent legacy dataset migrations from silently disappearing from the CLI. | `uv run --no-sync pytest -q tests/test_adapter_coverage.py` |
| [test_adapters_additional.py](../tests/test_adapters_additional.py) | 6 | Native annotation semantics, unknown supervision and official split guards. | `uv run --no-sync pytest -q tests/test_adapters_additional.py` |
| [test_adapters_auxiliary.py](../tests/test_adapters_auxiliary.py) | 13 | Adapter contracts: auxiliary labels must never become negative segmentations. | `uv run --no-sync pytest -q tests/test_adapters_auxiliary.py` |
| [test_adapters_segmentation.py](../tests/test_adapters_segmentation.py) | 10 | Annotation semantics and geometry regressions for legacy adapters. | `uv run --no-sync pytest -q tests/test_adapters_segmentation.py` |
| [test_augment.py](../tests/test_augment.py) | 6 | Scientific geometry, ignored-support and reproducibility checks for augmentation. | `uv run --no-sync pytest -q tests/test_augment.py` |
| [test_benchmark_manifest.py](../tests/test_benchmark_manifest.py) | 7 | Scientific eligibility must not be inferred from missing provenance. | `uv run --no-sync pytest -q tests/test_benchmark_manifest.py` |
| [test_biomisa_policy.py](../tests/test_biomisa_policy.py) | 3 | Fixed source filename roles must never become per-image area sorting. | `uv run --no-sync pytest -q tests/test_biomisa_policy.py` |
| [test_bundled_inference_defaults.py](../tests/test_bundled_inference_defaults.py) | 12 | Prediction-only defaults routing with fake metadata/predictors; no NN or weights. | `uv run --no-sync pytest -q tests/test_bundled_inference_defaults.py` |
| [test_bundled_models.py](../tests/test_bundled_models.py) | 3 | The shared package must include usable, intact inference checkpoints. | `uv run --no-sync pytest -q tests/test_bundled_models.py` |
| [test_cli.py](../tests/test_cli.py) | 2 | TOML/CLI precedence and rejection of unknown configuration before training. | `uv run --no-sync pytest -q tests/test_cli.py` |
| [test_comparison.py](../tests/test_comparison.py) | 6 | Paired grouped metric comparisons, support consistency, leakage and regression visibility. | `uv run --no-sync pytest -q tests/test_comparison.py` |
| [test_corpus_v2_proposal.py](../tests/test_corpus_v2_proposal.py) | 1 | A proposal cannot turn frozen holdout photo/patient aliases into training. | `uv run --no-sync pytest -q tests/test_corpus_v2_proposal.py` |
| [test_derived_cache.py](../tests/test_derived_cache.py) | 8 | Lossless derivatives stay explicit, portable, pinned and fail closed. | `uv run --no-sync pytest -q tests/test_derived_cache.py` |
| [test_engine.py](../tests/test_engine.py) | 9 | Pilot model geometry, masked loss, tiling, training/checkpoint/prediction cycle and AV detection support. | `uv run --no-sync pytest -q tests/test_engine.py` |
| [test_geometry_policy.py](../tests/test_geometry_policy.py) | 14 | Portable preparation policy, original native dimensions, and old model paths. | `uv run --no-sync pytest -q tests/test_geometry_policy.py` |
| [test_hpo.py](../tests/test_hpo.py) | 6 | Validate search decisions without reading held-out test data or training models. | `uv run --no-sync pytest -q tests/test_hpo.py` |
| [test_hpo_context.py](../tests/test_hpo_context.py) | 10 | Context, normalization and crop changes stay opt-in and keep label geometry. | `uv run --no-sync pytest -q tests/test_hpo_context.py` |
| [test_integrity.py](../tests/test_integrity.py) | 3 | Data identity must survive relocation and detect annotation changes on resume. | `uv run --no-sync pytest -q tests/test_integrity.py` |
| [test_legacy_annotation_completion.py](../tests/test_legacy_annotation_completion.py) | 4 | Secondary observers, disc-only annotations and external source overrides. | `uv run --no-sync pytest -q tests/test_legacy_annotation_completion.py` |
| [test_legacy_av.py](../tests/test_legacy_av.py) | 10 | Original A/V topology, safe conversion, decoding and reference-numeric checks. | `uv run --no-sync pytest -q tests/test_legacy_av.py` |
| [test_legacy_canonical_inference.py](../tests/test_legacy_canonical_inference.py) | 11 | Saved native-training recipes must survive raw-image CLI prediction. | `uv run --no-sync pytest -q tests/test_legacy_canonical_inference.py` |
| [test_legacy_cli.py](../tests/test_legacy_cli.py) | 12 | Public command-line routes distinguish verified originals from pilot models. | `uv run --no-sync pytest -q tests/test_legacy_cli.py` |
| [test_legacy_fov_optimization.py](../tests/test_legacy_fov_optimization.py) | 2 | The CPU optimization must preserve the historical disk morphology exactly. | `uv run --no-sync pytest -q tests/test_legacy_fov_optimization.py` |
| [test_legacy_inference.py](../tests/test_legacy_inference.py) | 6 | Integration checks using original checkpoints distributed with the package. | `uv run --no-sync pytest -q tests/test_legacy_inference.py` |
| [test_legacy_preprocessing.py](../tests/test_legacy_preprocessing.py) | 10 | Golden compatibility with original Python preprocessing and stored DICOM units. | `uv run --no-sync pytest -q tests/test_legacy_preprocessing.py` |
| [test_legacy_single_net.py](../tests/test_legacy_single_net.py) | 6 | Numerical compatibility with the original architecture and checkpoint format. | `uv run --no-sync pytest -q tests/test_legacy_single_net.py` |
| [test_legacy_training.py](../tests/test_legacy_training.py) | 18 | CPU-only functional fresh-native training on synthetic prepared photographs. | `uv run --no-sync pytest -q tests/test_legacy_training.py` |
| [test_matlab_av.py](../tests/test_matlab_av.py) | 7 | Public-API MATLAB numerical oracles and strict calibrated profile contracts. | `uv run --no-sync pytest -q tests/test_matlab_av.py` |
| [test_matlab_clahe.py](../tests/test_matlab_clahe.py) | 7 | Synthetic MATLAB-oracle comparisons; real-photo arrays remain outside Git. | `uv run --no-sync pytest -q tests/test_matlab_clahe.py` |
| [test_matlab_firstcut.py](../tests/test_matlab_firstcut.py) | 7 | First-cutoff required contract: source oracles, not full-vector certification. | `uv run --no-sync pytest -q tests/test_matlab_firstcut.py` |
| [test_matlab_fov.py](../tests/test_matlab_fov.py) | 5 | Synthetic public-MATLAB oracle tests; no MATLAB or real photos at test runtime. | `uv run --no-sync pytest -q tests/test_matlab_fov.py` |
| [test_matlab_fov_automatic.py](../tests/test_matlab_fov_automatic.py) | 8 | Synthetic MATLAB source-stage goldens; no production profile activation. | `uv run --no-sync pytest -q tests/test_matlab_fov_automatic.py` |
| [test_matlab_fov_source.py](../tests/test_matlab_fov_source.py) | 6 | Optional source-special stages; public defaults stay unchanged. | `uv run --no-sync pytest -q tests/test_matlab_fov_source.py` |
| [test_matlab_nearest_coordinates.py](../tests/test_matlab_nearest_coordinates.py) | 2 | Reference nearest-center coordinates, including the retained DiaRet tie. | `uv run --no-sync pytest -q tests/test_matlab_nearest_coordinates.py` |
| [test_matlab_source_auto.py](../tests/test_matlab_source_auto.py) | 5 | Optional automatic-source composition: no network or actual GT reader. | `uv run --no-sync pytest -q tests/test_matlab_source_auto.py` |
| [test_matlab_source_cli.py](../tests/test_matlab_source_cli.py) | 7 | Source profile opt-in routing and paired preprocessing, without model forwards. | `uv run --no-sync pytest -q tests/test_matlab_source_cli.py` |
| [test_matlab_threshold_study.py](../tests/test_matlab_threshold_study.py) | 4 | Fixed250-bin source goldens and explicitly uncertified threshold candidate. | `uv run --no-sync pytest -q tests/test_matlab_threshold_study.py` |
| [test_native_av_real_smoke_wrapper.py](../tests/test_native_av_real_smoke_wrapper.py) | 5 | Exercise only tiny stdlib child processes; never import Torch or real data. | `uv run --no-sync pytest -q tests/test_native_av_real_smoke_wrapper.py` |
| [test_native_bf16_smoke.py](../tests/test_native_bf16_smoke.py) | 3 | CPU-only checks of the explicit GPU smoke harness; no CUDA calls. | `uv run --no-sync pytest -q tests/test_native_bf16_smoke.py` |
| [test_native_real_smoke_cleanup.py](../tests/test_native_real_smoke_cleanup.py) | 1 | Actual SIGTERM cleanup with owned sleep processes; no trainer/NN/data access. | `uv run --no-sync pytest -q tests/test_native_real_smoke_cleanup.py` |
| [test_original_composed.py](../tests/test_original_composed.py) | 6 | Toy forwards exercise the literal benchmark operation without bundled weights. | `uv run --no-sync pytest -q tests/test_original_composed.py` |
| [test_partial_supervision.py](../tests/test_partial_supervision.py) | 2 | Partial optic-disc labels must never produce a supposedly trained cup output. | `uv run --no-sync pytest -q tests/test_partial_supervision.py` |
| [test_photometry_cache.py](../tests/test_photometry_cache.py) | 13 | Small immutable RGB derivatives; no full corpus, real-model training or GPU. | `uv run --no-sync pytest -q tests/test_photometry_cache.py` |
| [test_prediction_output.py](../tests/test_prediction_output.py) | 5 | A second model must not leave a mixture of old and new prediction files. | `uv run --no-sync pytest -q tests/test_prediction_output.py` |
| [test_prepare.py](../tests/test_prepare.py) | 12 | Regression tests for scientific label semantics and source separation. | `uv run --no-sync pytest -q tests/test_prepare.py` |
| [test_prepare_corpus_v2.py](../tests/test_prepare_corpus_v2.py) | 15 | Parent immutability, alias holdouts and nominal camera provenance for v2. | `uv run --no-sync pytest -q tests/test_prepare_corpus_v2.py` |
| [test_prepare_training_data.py](../tests/test_prepare_training_data.py) | 5 | RAW-only workflow ordering and identity guards, without an author-owned parent. | `uv run --no-sync pytest -q tests/test_prepare_training_data.py` |
| [test_prepare_unified_data.py](../tests/test_prepare_unified_data.py) | 6 | Canonical physical frames, multi-annotator identity and split-safety regressions. | `uv run --no-sync pytest -q tests/test_prepare_unified_data.py` |
| [test_product_photometry.py](../tests/test_product_photometry.py) | 6 | Product photometry integration: synthetic RGB and mocked pointwise inference only. | `uv run --no-sync pytest -q tests/test_product_photometry.py` |
| [test_reference_evaluation.py](../tests/test_reference_evaluation.py) | 3 | Reference metrics must preserve unknown labels and missed-vessel errors. | `uv run --no-sync pytest -q tests/test_reference_evaluation.py` |
| [test_refresh_original_av.py](../tests/test_refresh_original_av.py) | 2 | A new A/V reference must preserve every other original prediction count. | `uv run --no-sync pytest -q tests/test_refresh_original_av.py` |
| [test_sources.py](../tests/test_sources.py) | 5 | Resource identity and read-only archive behavior. | `uv run --no-sync pytest -q tests/test_sources.py` |
| [test_tpe_preview_release.py](../tests/test_tpe_preview_release.py) | 2 | Actual preview bytes/recipe and precision override; no quality evaluation. | `uv run --no-sync pytest -q tests/test_tpe_preview_release.py` |
| [test_unified.py](../tests/test_unified.py) | 25 | Scientific contracts for one partially supervised five-output RGB network. | `uv run --no-sync pytest -q tests/test_unified.py` |
| [test_unified_cli.py](../tests/test_unified_cli.py) | 14 | Public unified CLI round trips use generated functionality data, never quality claims. | `uv run --no-sync pytest -q tests/test_unified_cli.py` |
| [test_unified_hpo.py](../tests/test_unified_hpo.py) | 5 | Full-data search guards and unchanged scheduling across screen/resume stages. | `uv run --no-sync pytest -q tests/test_unified_hpo.py` |
| [test_unified_loss_balance.py](../tests/test_unified_loss_balance.py) | 7 | Optional v2 vessel emphasis: masked gradients and v1 numerical compatibility. | `uv run --no-sync pytest -q tests/test_unified_loss_balance.py` |
| [test_unified_mixed_precision.py](../tests/test_unified_mixed_precision.py) | 10 | CPU contracts; no real CUDA, real data, or speed/quality claims. | `uv run --no-sync pytest -q tests/test_unified_mixed_precision.py` |
| [test_unified_photometry.py](../tests/test_unified_photometry.py) | 13 | Opt-in full-image intensity contracts; synthetic fixtures, no scientific scores. | `uv run --no-sync pytest -q tests/test_unified_photometry.py` |
| [test_unified_tta.py](../tests/test_unified_tta.py) | 11 | Synthetic CPU contracts for opt-in fixed-weight inference ensembling. | `uv run --no-sync pytest -q tests/test_unified_tta.py` |
| [test_unified_warmstart.py](../tests/test_unified_warmstart.py) | 6 | Original-head fidelity, rectangular geometry, and explicit exposure safeguards. | `uv run --no-sync pytest -q tests/test_unified_warmstart.py` |
| [test_vascular_sampling_weight.py](../tests/test_vascular_sampling_weight.py) | 10 | Frozen6b bit/RNG compatibility and isolated vascular-key weighting; no GT files. | `uv run --no-sync pytest -q tests/test_vascular_sampling_weight.py` |

## Prediction defaults routing check added2026-11-12

The source snapshot above adds one module protecting optional per-bundled-unified
prediction defaults, explicit overrides, legacy/custom/evaluation isolation,
lazy public API and CLI-to-mocked-predict routing. Its **34 parametrized cases
passed in0.14s**, CPU1/CUDAhidden, with fake registries/predictors and no NN,
checkpoint/data/GT/GPU operations or installations. The earlier26-case subset is
not added to this count. One initial mocked evaluation fixture omitted required
`--output`; it was corrected without actual evaluation. Full source scope/results:
[verification](../reports/release/bundled-inference-defaults-20261112/verification.json).
All12 existing model entries/aliases still resolve to previous defaults, and no
real inference-default declaration was added. This is not a fresh entire suite.

## What has actually been observed

The entries below are **existing recorded observations**, not results of the
inventory command. Follow each record for full source/model/input hashes,
commands, backend, return-code receipts and retained qualifications. Where a
record lacks a wall-clock timestamp or complete tested-commit identity, that
limitation is stated rather than inferred from a filename. Separate selected
runs can overlap: do not add their counts to advertise a new full-suite total.

| Area | Recorded source/date and observed result | Scope and boundary | Evidence |
| --- | --- | --- | --- |
| Four original single-net checkpoints | Original reference `7a152c3`; Torch2.6 CPU. Each model has identical converted state tensors and zero maximum logit error on ramp/random native-size inputs and odd rectangular tiled input. Calendar execution timestamp not recorded in this conversion JSON. | Actual saved weights versus original Python model/tiler; synthetic prepared inputs. Historical raw/normalized provenance remains unverified, not reconstructed by this check. | [conversion JSON](../reports/legacy/single_net_conversion.json), [architecture/reference notes](LEGACY_MODELS.md), [`tests/reference/single_net`](../tests/reference/single_net) |
| Original v9 A/V checkpoint | Original reference `7bb0539`; Python3.12.13/Torch2.6 CPU. All75 saved state tensors equal and recorded prepared-input forward logit errors zero; exact model/input hashes retained. Calendar execution timestamp not recorded in this conversion JSON. | Actual owned pickle converted to safe state dictionary, with rectangular/odd synthetic stored-unit inputs. Does not certify arbitrary-camera MATLAB preprocessing. | [conversion JSON](../reports/legacy/av_conversion.json), [AV contract and optional external original](LEGACY_AV.md), [pinned original reference](../tests/reference/av/README.md) |
| Historical whole CPU suite | `0c6eaa7`, 2026-11-10T20:44:29UTC: 2,440 passed, 58 warnings, 45.47s. | An entire earlier source snapshot; not the current62-module inventory's measured status. Parameter expansion explains why case totals exceed definition counts. | [original command/log hash/result](../reports/release/source-cli-integration/verification.json) |
| Native legacy synthetic training | Base `4ec7344`; 2,540 passed/51.94s in the original implementation record, then 2,605 passed/53.52s in an integration record. Calendar timestamps and exact final tested commit are not both present in these JSONs. | Actual small synthetic CPU optimizer/VAL/save/reload, default production architecture metadata, plus wider regression suite at those snapshots. New fresh recipe, not reproduction of unavailable historical optimizer/data pipeline. | [initial verification](../reports/native-legacy-training/verification.json), [integration verification](../reports/release/native-training-integration/verification.json), [recipe](NATIVE_LEGACY_RETRAINING.md) |
| Four original production single-net presets, real data | 2026-11-11 bounded workflow; launch/source metadata in linked receipts. All four actual child exits0: vessel42.56s, disc19212.02s, disc25613.02s, universal44.06s. All74 trainable tensors changed versus same-seed fresh constructors. | Actual one-epoch training on a four-photo TRAIN/VAL subset, native patch sizes, full-frame VAL, finite optimizer/checkpoint and reloaded real64×64 TRAIN crops. Task-specific TRAIN coverage is one or two photos, **not** all2,529/5,700. No TEST. | [workflow/limits/failures](../reports/native-legacy-real-smoke/README.md), [actual verification](../reports/native-legacy-real-smoke/root-four-native-verification.json), [subset proof](../reports/native-legacy-real-smoke/subset-proof.json) |
| Original production full-frame A/V, real data | `d028314` on2026-11-11: actual fresh GPU training child/observer0 (4.753508s) and independent verification child/observer0 (2.652429s). | One unchanged full-frame TRAIN photo/AV record and one full VAL photo; all54 trainable parameter tensors changed, safe reload finite. GT vessel is the training input; this is functional training, not useful new weights or full-corpus quality. Earlier CPU RSS defer is retained. | [GPU scientific proof](../reports/native-legacy-real-smoke/av-GPU-functional-proof.json), [CPU defer and GPU qualification](../reports/native-legacy-real-smoke/README.md) |
| Original composition real-photo parity | `e249b81`, 2026-11-12 corrected attempt03: child0 and external0, 43.1s. | One DRIVE21 **TRAIN** photo/official FOV, same verified four loaded models versus frozen benchmark composition; all five binary masks bitwise equal, finite scores. Not global camera/backend parity, quality or latency evidence. Two failed pre-forward attempts remain documented. | [all three attempts](../reports/original-composed-real-parity-20261112/README.md), [successful result](../reports/original-composed-real-parity-20261112/attempt03/result.json) |
| BF16 compatible product loader/trainer | Product port `903c929` versus reference `72fb321`; actual CUDA synthetic check4.096s, PASS. Calendar timestamp not recorded in this JSON. | Synthetic step has exact model/optimizer/loss/prediction/CUDA-RNG equality. No real photo/GT/TEST or operational checkpoint used. CPU checks and unsupported-device rejection are separate. | [native CUDA proof](../reports/release/bf16-compatibility/native-cuda/verification.json), [CPU/root evidence](../reports/release/bf16-compatibility/README.md), [precision contract](UNIFIED_TRAIN_BF16.md) |
| Product optional TTA | Product base `8b99b64`, joint reference `1f723ad`; 148 unique focused CPU tests confirmed across selected runs. Calendar timestamps are not in the prose summary; hashes/backend in detailed proof. | Actual BEST58 weights on generated3×35×49 input, tiny32/8 tiling: default-none versus previous product and all four modes versus joint reference bitwise equal. No real photo/quality inference. Fixture/setup/quota failures preserved. | [full scope and attempts](../reports/product-unified-tta/README.md), [prepared-forward identities](../reports/product-unified-tta/prepared-forward-parity.json), [verification](../reports/product-unified-tta/verification.json) |
| Product opt-in photometry/cache | Base `d028314`, authority `fd44cdf`; observed selected groups89/8.34s and32/5.61s, plus disclosed tiny generated CLI training cases and extra raw/cache resume checks. Exact per-group scope in JSON; not one summed full suite. | Synthetic once-only geometry/photometry, source identity, no-GT-FOV fallback, descriptor/backend strictness, cache/RNG/optimizer parity. This record explicitly discloses incidental generated training; no real data/model/GPU. | [verification and exceptions](../reports/product-unified-photometry/verification.json), [cross-source identities](../reports/product-unified-photometry/cross-source-verification.json), [feature guide](UNIFIED_PHOTOMETRY.md) |
| Real TRAIN raw photometry-port parity | 2026-11-11 candidate `1fb4a0c` (merged unchanged `a64c15c`), references product `48dd081`/authority `fd44cdf`. Successful comparison children0, 27.314s/24.919s. | One DRIVE TRAIN photo/FOV, genuine TPE58 and raw256 weights, bitwise finite probabilities/masks. First successful TPE child was inside a combined attempt with outer1; raw256 success had outer0. Both original failures and source scopes retained. | [scope and failure qualifications](../reports/product-photometry-real-train-parity/README.md), [numeric summary](../reports/product-photometry-real-train-parity/summary.json) |
| Adapters: real sample checks | Migration audit records22 available new selectors×2 records (44 samples), followed by22 actual integrated `prepare(limit=3,max_size=64)` calls; exact historical source/time not fully bound by prose, so these are bounded historical checks. | Actual decoder/preparation dimensions, finite labels, validity and copied resources. Not every record/archive; unavailable complete UBMI skipped. Sparse AVRDB positives are not dense negatives; image-only resources never become GT. | [migration evidence and limitations](ADAPTER_MIGRATION_AUDIT.md), [segmentation checks](ADAPTERS_SEGMENTATION.md), [auxiliary checks](ADAPTERS_AUXILIARY.md), [machine coverage scopes](../src/deepretina/adapters/coverage.json) |
| Corpus V2 actual preparation audit | Producer `fa1cf03`, completed2026-11-10T21:51:21UTC, exit0/401.70s; frozen reader `3157` independently consumed it. | Mechanical byte/geometry/split/integrity audit: TRAIN2,529 photos/5,700 variants; VAL770/993; TEST856/1,118. Official TEST arrays were mechanically decoded, **not model scored or used to tune**. Parent memberships/bytes preserved; nominal camera policies qualified. | [completion overview](../reports/data/corpus-v2-preparation/README.md), [original terminal](../reports/data/corpus-v2-preparation/complete/terminal.json), [frozen reader](../reports/data/corpus-v2-preparation/complete/frozen3157-reader-check.json), [parent preservation](../reports/data/corpus-v2-preparation/complete/parent-preservation.json) |
| Installed wheel including photometry | Source `5d50fc5`, 2026-11-12 verification record: one build/install/check all actual0, 5.02s/0.67s/2.97s. | All12 weight SHAs/aliases and45 runtime Python modules exact in wheel/isolated target; outside-checkout imports/help,42 existing dependency metadata unchanged. NN/checkpoint/CUDA trapped; no fresh dependencies or inference. | [packaging report](../reports/product-wheel-5d50-photometry/README.md), [exact verification](../reports/product-wheel-5d50-photometry/verification.json) |
| Installed wheel including composed-original API | Source `e249b81`, integrated byte-exact `b282eeb`, 2026-11-12 workflow: observed build0/install0/check0 (4.17s/0.62s/2.82s in receipts). | All46 runtime modules/12 blobs/aliases exact; installed `predict-original`/API/help imports, NN/CUDA trapped. Earlier package-only proof predated a functional tiling defect, so use corrected source and separate actual parity above. | [corrected packaging overview](../reports/product-original-composed-packaging/README.md), [exact times/receipts](../reports/product-original-composed-packaging/verification.json) |
| Leipzig documentation/CLI usability | `2350bbf` source, documentation commit `52d1031`, 2026-11-12: actual original/predict/models help0 without importing Torch; examples parsed with AST. | No prediction/data decode/install validation; checks arguments, path-vs-array API contract and archive member metadata only. Not a fresh whole suite. | [audit record](../reports/release/leipzig-usability-audit-20261112/README.md) |

### Synthetic numerical fixtures and reproducibility

The [`tests/reference`](../tests/reference) directory contains pinned original
Python architecture/tiling implementations, MATLAB-generated synthetic array
oracles, and previous loss/sampling/geometry helpers. Each reference family has
provenance/metadata documenting the original commit or oracle generator, array
hashes and scope. These files permit a Python-only regression run; they do not
supply missing historical training HDF5 or certify every real acquisition.
See [legacy preprocessing](LEGACY_PREPROCESSING.md),
[MATLAB CLAHE](MATLAB_CLAHE.md), [FOV calibration](MATLAB_FOV_CALIBRATION.md),
[source FOV stages](MATLAB_FOV_SOURCE_PORT.md) and
[first-cutoff limitations](MATLAB_FIRST_CUTOFF.md). Numerical checks distinguish
exact source stages from still-qualified complete preprocessing claims.

### Model quality, dataset splits and pending work

Actual validation-quality studies and completed full-training provenance are
linked from the [control](UNIFIED_V2_CONTROL.md),
[vascular preview](UNIFIED_V2_VASCULAR2.md) and [TPE preview](UNIFIED_V3_TPE_PREVIEW.md)
model cards and [experiment notes](EXPERIMENTS.md). They are not pytest functional
checks. Five segmentation-head metrics are Dice; conditional A/V accuracy is an
accuracy metric. Sparse/partial support, historical original-model exposure,
backend differences, adverse source/head results, threshold/TTA exploration and
checkpoint BEST-versus-LAST coverage qualifications remain part of the evidence.
A higher aggregate score does not erase a vessel regression.

`test_*` is pytest naming, not a claim that the held-out dataset TEST partition
was evaluated. Tests protect split leakage and the public unified VAL route's
TEST refusal using generated data. Model TEST evaluation/final selection remains
separate and pending; do not run an ad hoc `--split test`, retag a VAL report or
use a synthetic gate fixture as authorization. Historical original checkpoints
may have known or unresolved training/tuning exposure; unit-test parity cannot
establish independence. Data-preparation audits reading official TEST annotation
formats are explicitly distinct from predicting/scoring those photos.

### Retain failures and exact scope

A stopped or failed attempt is not a PASS. In particular, retain the original
A/V CPU memory defer, composed-original pre-forward identity/tiling failures,
TTA fixture/quota setup failures, photometry comparison outer1 with a separately
successful child, and all source/time/role qualifiers in their linked records.
Corrected later success belongs to its new source/attempt. Do not rewrite a
historical receipt or infer an unobserved external return code.

This inventory update ran **stdlib AST/file-metadata checks and static lint of
the discovery utility only**, as recorded in the [inventory check notes](../reports/testing-inventory-20261112/README.md). Its generated
JSON enumerates source files and definitions; it cannot replace actual pytest
collection, execution logs, real-data parity or installed-wheel proofs.


Additional focused product regression verification: [original run and corrected historical
recipe case](../reports/release/bundled-inference-defaults-20261112/focused-existing-regressions/README.md).
Original run96passed/1failed in16.50s; failure was a historical test missing the
backward-compatible photometric_profile=raw default. Narrow test-only normalization
then passed that one metadata case in2.29s; the96 passing cases were not rerun.
No runtime, weights, registry or model defaults were changed by the correction.
