# Testing and validation evidence

Snapshot: HPO **5cea6d5**, including34e0abf curated stage2 dispatch and the fixed
TRAIN timing-cohort tests. This is an
**AST inventory, not a new test run**. No pytest collection, test/model/Torch import,
NN forward, real image/GT read, GPU query, installation, selection freeze or TEST
evaluation was performed to write it. Source and test files remain unchanged.
Private Meta/host controllers and the separate `for_Leipzig` product catalog have
separate inventories; this page covers every tracked HPO test module, including
the coordinator and the historical and nonstandard report-local modules.

There are **89 test modules / 662 static test definitions**:82/627 under `tests/`,
4/25 in the coordinator, and3/10 report-local. A definition is a `test*` function or
method in parsed source. Parametrization and fixtures expand execution cases;
these numbers are **not collected cases or passed-test counts**. The `Param`
column counts definitions decorated with `parametrize`, not their combinations.
For example curated stage2 dispatch is one definition with six recorded cases
(three modes × two roles). `checks.py` is included despite its nonstandard filename.
Historical module docstrings sometimes describe the state when written; they do
not override the current scientific evidence below. The exact file SHA, test
names/lines and full command for every module are in [testing-inventory.json](testing-inventory.json).

## Reproduce available tests

Run from the repository root in already prepared uv environments. `--frozen`
prevents lock changes; `--no-sync` performs no installs and requires the relevant
dependencies to exist. These recipes were **not executed for this inventory**. `U` directly invokes the
report-local unittest scripts; it is not automatic default pytest discovery.
`T` uses the main Python3.11–3.12/Torch/pytest environment. `C` uses the separate
Python3.12/Optuna5 coordinator environment; the root fixed27 proposal module also
needs that environment. No live study is used: coordinator fixtures create
in-memory/temporary SQLite studies. The fixed27 module additionally needs the
frozen8777 Git object; verify it exists rather than silently replacing the contract.

```sh
# Define once, from the repository root. Hidden CUDA and one math thread are explicit.
T() {
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  uv run --frozen --no-sync --group dev python -m pytest -q "$@"
}
C() {
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools/optuna_coordinator/src \
  uv run --project tools/optuna_coordinator --frozen --no-sync --group dev \
    python -m pytest -q "$@"
}
U() {
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  uv run --frozen --no-sync --group dev python "$@"
}
# Together these explicitly cover all89 modules; pytest's default testpaths covers only tests/.
T tests --ignore=tests/test_fixed27_proposal.py
C tests/test_fixed27_proposal.py tools/optuna_coordinator/tests
T reports/unified/paired-avweight-e008/test_bootstrap_extension.py
U reports/dense-vessel-support-20261112/checks.py
U reports/fixed-timing-input-cohort-20261112/test_cohort.py
```

Per-module commands are in the tables and JSON. Some main tests load **bundled
original/pilot checkpoints and run CPU inference on generated images**; others use
tiny constructed tensors/weights, metadata or saved counts only. They are not all
no-forward tests. Preserve the shipped assets/fixture byte hashes. Synthetic
`split=test` fixtures exercise held-out dispatch without reading the real TEST
partition. Running the reviewed TEST CLI on real data is a different, explicitly
gated operation; these test commands do not authorize it.

## Every current test module

### Main package and scientific contracts

| Module | Defs | Param | Purpose | Reproduce |
|---|---:|---:|---|---|
| [test_adapter_coverage.py](../tests/test_adapter_coverage.py) | 3 | 0 | Prevent legacy dataset migrations from silently disappearing from the CLI. | `T tests/test_adapter_coverage.py` |
| [test_adapters_additional.py](../tests/test_adapters_additional.py) | 6 | 0 | Native annotation semantics, unknown supervision and official split guards. | `T tests/test_adapters_additional.py` |
| [test_adapters_auxiliary.py](../tests/test_adapters_auxiliary.py) | 13 | 2 | Adapter contracts: auxiliary labels must never become negative segmentations. | `T tests/test_adapters_auxiliary.py` |
| [test_adapters_segmentation.py](../tests/test_adapters_segmentation.py) | 10 | 0 | Annotation semantics and geometry regressions for legacy adapters. | `T tests/test_adapters_segmentation.py` |
| [test_augment.py](../tests/test_augment.py) | 6 | 1 | Scientific geometry, ignored-support and reproducibility checks for augmentation. | `T tests/test_augment.py` |
| [test_benchmark_manifest.py](../tests/test_benchmark_manifest.py) | 7 | 0 | Scientific eligibility must not be inferred from missing provenance. | `T tests/test_benchmark_manifest.py` |
| [test_biomisa_policy.py](../tests/test_biomisa_policy.py) | 3 | 0 | Fixed source filename roles must never become per-image area sorting. | `T tests/test_biomisa_policy.py` |
| [test_bn_recalibration.py](../tests/test_bn_recalibration.py) | 9 | 0 | Tiny CPU synthetic forward only; no actual TRAIN/VAL/TEST arrays or weights. | `T tests/test_bn_recalibration.py` |
| [test_bn_research_validation.py](../tests/test_bn_research_validation.py) | 2 | 1 | Derived metadata/byte checks only; no actual calibration or VAL run. | `T tests/test_bn_research_validation.py` |
| [test_bundled_models.py](../tests/test_bundled_models.py) | 3 | 0 | HPO original/pilot catalog bytes and generated-image CPU prediction using bundled weights. | `T tests/test_bundled_models.py` |
| [test_cli.py](../tests/test_cli.py) | 2 | 0 | Training TOML/explicit-argument precedence and rejection before execution. | `T tests/test_cli.py` |
| [test_compare_standalone_val.py](../tests/test_compare_standalone_val.py) | 4 | 1 | Standalone BEST/LAST own-coverage and paired-report provenance; immutable inputs. | `T tests/test_compare_standalone_val.py` |
| [test_comparison.py](../tests/test_comparison.py) | 10 | 3 | Paired group bootstrap, per-source/all-head support and provenance rejection. | `T tests/test_comparison.py` |
| [test_completed_val_comparison.py](../tests/test_completed_val_comparison.py) | 5 | 1 | Metadata contract checks without opening real reports, GT, or models. | `T tests/test_completed_val_comparison.py` |
| [test_compose_disc_reference.py](../tests/test_compose_disc_reference.py) | 3 | 0 | Count-only composition must not change GT support or any other head. | `T tests/test_compose_disc_reference.py` |
| [test_control_e060_verifier.py](../tests/test_control_e060_verifier.py) | 3 | 1 | Tiny metadata/tensor fixtures only: no real checkpoints, photos or inference. | `T tests/test_control_e060_verifier.py` |
| [test_corpus_v2_proposal.py](../tests/test_corpus_v2_proposal.py) | 1 | 1 | A proposal cannot turn frozen holdout photo/patient aliases into training. | `T tests/test_corpus_v2_proposal.py` |
| [test_cup_containment.py](../tests/test_cup_containment.py) | 6 | 0 | Postprocessing changes cup predictions only and preserves reference support. | `T tests/test_cup_containment.py` |
| [test_cup_continuation.py](../tests/test_cup_continuation.py) | 9 | 1 | Synthetic count/mask fixtures; no production photos/checkpoints or forwards. | `T tests/test_cup_continuation.py` |
| [test_curated_completion.py](../tests/test_curated_completion.py) | 17 | 9 | Synthetic tiny constructed weights + metadata; no real weights/GT/NN forward. | `T tests/test_curated_completion.py` |
| [test_curated_final_context.py](../tests/test_curated_final_context.py) | 4 | 3 | Synthetic metadata wiring only; genuine original-chain tests live separately. | `T tests/test_curated_final_context.py` |
| [test_derived_cache.py](../tests/test_derived_cache.py) | 8 | 1 | Lossless derivatives stay explicit, portable, pinned and fail closed. | `T tests/test_derived_cache.py` |
| [test_engine.py](../tests/test_engine.py) | 9 | 1 | Generated-data model geometry, masked gradients, tiled coverage and hierarchy decoding. | `T tests/test_engine.py` |
| [test_fixed27_proposal.py](../tests/test_fixed27_proposal.py) | 6 | 1 | Synthetic CPU metadata only; native Optuna and native8777 validation, no NN/GT. | `C tests/test_fixed27_proposal.py` |
| [test_freeze_eligibility.py](../tests/test_freeze_eligibility.py) | 4 | 2 | Synthetic metadata-only invariants; no model forward, real GT or TEST. | `T tests/test_freeze_eligibility.py` |
| [test_frozen_completed_val_comparison.py](../tests/test_frozen_completed_val_comparison.py) | 8 | 2 | Synthetic770 VAL metadata only; no real photos, GT, weights or TEST reads. | `T tests/test_frozen_completed_val_comparison.py` |
| [test_frozen_original_reference.py](../tests/test_frozen_original_reference.py) | 9 | 2 | Synthetic only: no actual image, annotation, original model or CUDA access. | `T tests/test_frozen_original_reference.py` |
| [test_geometry_policy.py](../tests/test_geometry_policy.py) | 14 | 5 | Portable preparation policy, original native dimensions, and old model paths. | `T tests/test_geometry_policy.py` |
| [test_gui_validation_continuation.py](../tests/test_gui_validation_continuation.py) | 4 | 1 | Fake process metadata; no real /proc, GPU, checkpoint or model operations. | `T tests/test_gui_validation_continuation.py` |
| [test_heldout_curated_dispatch_metadata.py](../tests/test_heldout_curated_dispatch_metadata.py) | 1 | 1 | Stage2 curated TTA metadata dispatch only: no weights, GT, forward or CUDA. | `T tests/test_heldout_curated_dispatch_metadata.py` |
| [test_highlevel_predict_tta.py](../tests/test_highlevel_predict_tta.py) | 7 | 3 | High-level TTA wiring with generated RGB and tiny CPU forwards; no real weights/GT. | `T tests/test_highlevel_predict_tta.py` |
| [test_hpo.py](../tests/test_hpo.py) | 6 | 1 | Validate search decisions without reading held-out test data or training models. | `T tests/test_hpo.py` |
| [test_hpo_context.py](../tests/test_hpo_context.py) | 10 | 1 | Context, normalization and crop changes stay opt-in and keep label geometry. | `T tests/test_hpo_context.py` |
| [test_hpo_photometry.py](../tests/test_hpo_photometry.py) | 8 | 6 | Synthetic full-image photometry plus strict HPO TTA/provenance integration. | `T tests/test_hpo_photometry.py` |
| [test_integrity.py](../tests/test_integrity.py) | 3 | 1 | Data identity must survive relocation and detect annotation changes on resume. | `T tests/test_integrity.py` |
| [test_legacy_annotation_completion.py](../tests/test_legacy_annotation_completion.py) | 4 | 0 | Secondary observers, disc-only annotations and external source overrides. | `T tests/test_legacy_annotation_completion.py` |
| [test_legacy_av.py](../tests/test_legacy_av.py) | 10 | 0 | Original A/V topology, safe conversion, decoding and reference-numeric checks. | `T tests/test_legacy_av.py` |
| [test_legacy_cli.py](../tests/test_legacy_cli.py) | 12 | 4 | Original/pilot CLI dispatch, explicit raw proxy and generated-input output contracts. | `T tests/test_legacy_cli.py` |
| [test_legacy_disc_evaluation.py](../tests/test_legacy_disc_evaluation.py) | 3 | 0 | Protect the original strict threshold and reference support contract. | `T tests/test_legacy_disc_evaluation.py` |
| [test_legacy_fov_optimization.py](../tests/test_legacy_fov_optimization.py) | 2 | 2 | The CPU optimization must preserve the historical disk morphology exactly. | `T tests/test_legacy_fov_optimization.py` |
| [test_legacy_inference.py](../tests/test_legacy_inference.py) | 6 | 1 | Bundled-original CPU integration on generated inputs: geometry, AV decoding and restoration. | `T tests/test_legacy_inference.py` |
| [test_legacy_preprocessing.py](../tests/test_legacy_preprocessing.py) | 10 | 2 | Golden compatibility with original Python preprocessing and stored DICOM units. | `T tests/test_legacy_preprocessing.py` |
| [test_legacy_single_net.py](../tests/test_legacy_single_net.py) | 6 | 1 | Numerical compatibility with the original architecture and checkpoint format. | `T tests/test_legacy_single_net.py` |
| [test_matlab_av.py](../tests/test_matlab_av.py) | 7 | 2 | Public-API MATLAB numerical oracles and strict calibrated profile contracts. | `T tests/test_matlab_av.py` |
| [test_matlab_clahe.py](../tests/test_matlab_clahe.py) | 7 | 3 | Synthetic MATLAB-oracle comparisons; real-photo arrays remain outside Git. | `T tests/test_matlab_clahe.py` |
| [test_matlab_fov.py](../tests/test_matlab_fov.py) | 5 | 3 | Synthetic public-MATLAB oracle tests; no MATLAB or real photos at test runtime. | `T tests/test_matlab_fov.py` |
| [test_native_v3_count_analysis.py](../tests/test_native_v3_count_analysis.py) | 5 | 1 | Small metadata-only checks; genuine completed inputs supply the positive proof. | `T tests/test_native_v3_count_analysis.py` |
| [test_partial_supervision.py](../tests/test_partial_supervision.py) | 2 | 0 | Partial optic-disc labels must never produce a supposedly trained cup output. | `T tests/test_partial_supervision.py` |
| [test_photometry_cache.py](../tests/test_photometry_cache.py) | 13 | 2 | Small immutable RGB derivatives; no full corpus, real-model training or GPU. | `T tests/test_photometry_cache.py` |
| [test_pool_completion_adapter.py](../tests/test_pool_completion_adapter.py) | 12 | 5 | Synthetic terminal metadata only; no weights, model/GT loads or CUDA calls. | `T tests/test_pool_completion_adapter.py` |
| [test_prepare.py](../tests/test_prepare.py) | 12 | 2 | Regression tests for scientific label semantics and source separation. | `T tests/test_prepare.py` |
| [test_prepare_corpus_v2.py](../tests/test_prepare_corpus_v2.py) | 10 | 1 | Parent immutability, alias holdouts and nominal camera provenance for v2. | `T tests/test_prepare_corpus_v2.py` |
| [test_prepare_unified_data.py](../tests/test_prepare_unified_data.py) | 6 | 0 | Canonical physical frames, multi-annotator identity and split-safety regressions. | `T tests/test_prepare_unified_data.py` |
| [test_reference_evaluation.py](../tests/test_reference_evaluation.py) | 3 | 0 | Reference metrics must preserve unknown labels and missed-vessel errors. | `T tests/test_reference_evaluation.py` |
| [test_refresh_original_av.py](../tests/test_refresh_original_av.py) | 2 | 1 | A new A/V reference must preserve every other original prediction count. | `T tests/test_refresh_original_av.py` |
| [test_reviewed_heldout.py](../tests/test_reviewed_heldout.py) | 21 | 5 | Tiny synthetic stage2 forwards; no actual held-out data or GPU experiments. | `T tests/test_reviewed_heldout.py` |
| [test_reviewed_test_comparison.py](../tests/test_reviewed_test_comparison.py) | 12 | 5 | Synthetic saved counts/metadata only; no real TEST data or inference. | `T tests/test_reviewed_test_comparison.py` |
| [test_sampling_exposure_audit.py](../tests/test_sampling_exposure_audit.py) | 4 | 0 | Analytic small-corpus checks for pre-crop expected exposure, never ML training. | `T tests/test_sampling_exposure_audit.py` |
| [test_selection_tta.py](../tests/test_selection_tta.py) | 16 | 8 | Reviewed TTA metadata/math only: no network forward, actual GT or TEST scoring. | `T tests/test_selection_tta.py` |
| [test_sources.py](../tests/test_sources.py) | 5 | 0 | Resource identity and read-only archive behavior. | `T tests/test_sources.py` |
| [test_streaming_thresholds.py](../tests/test_streaming_thresholds.py) | 2 | 1 | Native threshold counters, ignored support and same-forward producer parity. | `T tests/test_streaming_thresholds.py` |
| [test_strong_pipeline_benchmark.py](../tests/test_strong_pipeline_benchmark.py) | 22 | 7 | Exact four-network composition, reference/TTA/threshold forwarding, phase scopes and retained output checks. | `T tests/test_strong_pipeline_benchmark.py` |
| [test_terminal60_threshold_watch.py](../tests/test_terminal60_threshold_watch.py) | 6 | 2 | Synthetic metadata only; no real checkpoint/model/GPU/manifest operations. | `T tests/test_terminal60_threshold_watch.py` |
| [test_terminal60_val_comparison.py](../tests/test_terminal60_val_comparison.py) | 3 | 1 | Synthetic all2529/5700/770 metadata only; no real model/data/TEST reads. | `T tests/test_terminal60_val_comparison.py` |
| [test_unified.py](../tests/test_unified.py) | 25 | 4 | Scientific contracts for one partially supervised five-output RGB network. | `T tests/test_unified.py` |
| [test_unified_av_residual.py](../tests/test_unified_av_residual.py) | 12 | 4 | CPU numerical/gradient contracts for optional AV capacity, not quality evidence. | `T tests/test_unified_av_residual.py` |
| [test_unified_cli.py](../tests/test_unified_cli.py) | 13 | 2 | Public unified CLI round trips use generated functionality data, never quality claims. | `T tests/test_unified_cli.py` |
| [test_unified_hpo.py](../tests/test_unified_hpo.py) | 5 | 0 | Full-data search guards and unchanged scheduling across screen/resume stages. | `T tests/test_unified_hpo.py` |
| [test_unified_loss_balance.py](../tests/test_unified_loss_balance.py) | 7 | 2 | Optional v2 vessel emphasis: masked gradients and v1 numerical compatibility. | `T tests/test_unified_loss_balance.py` |
| [test_unified_mixed_precision.py](../tests/test_unified_mixed_precision.py) | 9 | 1 | CPU contracts; no real CUDA, real data, or speed/quality claims. | `T tests/test_unified_mixed_precision.py` |
| [test_unified_photometry.py](../tests/test_unified_photometry.py) | 13 | 5 | Opt-in full-image intensity contracts; synthetic fixtures, no scientific scores. | `T tests/test_unified_photometry.py` |
| [test_unified_selection.py](../tests/test_unified_selection.py) | 29 | 9 | Selection metadata checks use CPU toy snapshots, never any network forward. | `T tests/test_unified_selection.py` |
| [test_unified_warmstart.py](../tests/test_unified_warmstart.py) | 6 | 2 | Original-head fidelity, rectangular geometry, and explicit exposure safeguards. | `T tests/test_unified_warmstart.py` |
| [test_v3_completion.py](../tests/test_v3_completion.py) | 3 | 1 | Synthetic pool72 full60 task/receipt/milestone chain; reject partial or retagged evidence. | `T tests/test_v3_completion.py` |
| [test_val_bridge.py](../tests/test_val_bridge.py) | 11 | 3 | Synthetic encoded assets only: no GT decoding, NumPy, Torch or model forward. | `T tests/test_val_bridge.py` |
| [test_validation_evidence.py](../tests/test_validation_evidence.py) | 10 | 3 | Fresh tiny CPU forwards, never actual corpus/model evaluation or test GT reads. | `T tests/test_validation_evidence.py` |
| [test_vascular2_bn_comparison.py](../tests/test_vascular2_bn_comparison.py) | 2 | 1 | Real completed research metadata; no model/GT/forward access. | `T tests/test_vascular2_bn_comparison.py` |
| [test_vascular2_bn_recalibration.py](../tests/test_vascular2_bn_recalibration.py) | 9 | 0 | Tiny CPU synthetic forward only; no actual TRAIN/VAL/TEST arrays or weights. | `T tests/test_vascular2_bn_recalibration.py` |
| [test_vascular2_bn_validation.py](../tests/test_vascular2_bn_validation.py) | 3 | 1 | Derived metadata/byte checks only; no actual calibration or VAL run. | `T tests/test_vascular2_bn_validation.py` |
| [test_vascular2_comparison.py](../tests/test_vascular2_comparison.py) | 4 | 2 | Actual completed standalone report metadata; no GT/model access. | `T tests/test_vascular2_comparison.py` |
| [test_vascular2_val_readiness.py](../tests/test_vascular2_val_readiness.py) | 4 | 1 | Frozen CLI delegation and nested metadata only; no Torch/model/GT/NN work. | `T tests/test_vascular2_val_readiness.py` |
| [test_vascular_sampling_weight.py](../tests/test_vascular_sampling_weight.py) | 11 | 4 | Frozen6b bit/RNG compatibility and isolated vascular-key weighting; no GT files. | `T tests/test_vascular_sampling_weight.py` |

### Native coordinator (separate uv project)

| Module | Defs | Param | Purpose | Reproduce |
|---|---:|---:|---|---|
| [test_baseline_rung_reference.py](../tools/optuna_coordinator/tests/test_baseline_rung_reference.py) | 1 | 1 | Optuna5 native rung membership, demonstrated in memory; no live study access. | `C tools/optuna_coordinator/tests/test_baseline_rung_reference.py` |
| [test_core.py](../tools/optuna_coordinator/tests/test_core.py) | 7 | 0 | Temporary native Optuna lifecycle, leases, receipts, full60-only completion and config bounds. | `C tools/optuna_coordinator/tests/test_core.py` |
| [test_sampler_state.py](../tools/optuna_coordinator/tests/test_sampler_state.py) | 6 | 1 | Real pinned Optuna5/SQLite tests; no model, data reader, transport or GPU. | `C tools/optuna_coordinator/tests/test_sampler_state.py` |
| [test_v2_finite.py](../tools/optuna_coordinator/tests/test_v2_finite.py) | 11 | 5 | Real Optuna/SQLite protocol2, synthetic receipts only; no model/data/GPU. | `C tools/optuna_coordinator/tests/test_v2_finite.py` |

### Historical report-local regression

| Module | Defs | Param | Purpose | Reproduce |
|---|---:|---:|---|---|
| [checks.py](../reports/dense-vessel-support-20261112/checks.py) | 6 | 0 | Tiny metadata-only negative cases for the supplementary support analysis. | `U reports/dense-vessel-support-20261112/checks.py` |
| [test_cohort.py](../reports/fixed-timing-input-cohort-20261112/test_cohort.py) | 3 | 0 | Synthetic IO/identity guards only; no models, Torch or real annotation inputs. | `U reports/fixed-timing-input-cohort-20261112/test_cohort.py` |
| [test_bootstrap_extension.py](../reports/unified/paired-avweight-e008/test_bootstrap_extension.py) | 1 | 1 | Synthetic paired draws shared by pooled and macro comparisons; closed-form difference. | `T reports/unified/paired-avweight-e008/test_bootstrap_extension.py` |

### Similarly named tools, not test modules

| File | AST test definitions | What it does |
|---|---:|---|
| [scripts/unified_test.py](../scripts/unified_test.py) |0|Explicit reviewed held-out stage2 CLI; never treat its filename as a unit-test result. |
| [scripts/compare_reviewed_test.py](../scripts/compare_reviewed_test.py) |0|Count-only comparator of already reviewed TEST reports; does not run a model. |

Use `uv run --frozen --no-sync python scripts/unified_test.py --help` or the
corresponding comparator `--help` to inspect interfaces. No actual held-out
execution is implied. Their dispatch/gate regressions are listed above.

## Recompute static counts without importing tests

This reads tracked Python source with `ast`; it does not collect or execute tests.
Checking all tracked Python files also finds nonstandard `checks.py` modules.

```sh
uv run --frozen --no-sync python - <<'PYCODE'
import ast, subprocess
from pathlib import Path
modules = definitions = 0
for name in subprocess.check_output(["git", "ls-files", "*.py"], text=True).splitlines():
    tree = ast.parse(Path(name).read_bytes(), filename=name)
    tests = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test")]
    if tests:
        print(name, len(tests))
        modules += 1
        definitions += len(tests)
print("modules", modules, "AST definitions", definitions)
PYCODE
```

## Recorded software verification: actual runs, not this inventory

These are selected existing execution proofs covering inference, benchmarking,
final gates, held-out dispatch, data and training contracts. Their numbers are
**executed cases at their recorded source snapshots**, not AST definitions.
Subsets overlap: do not add them into a unique total, treat them as a current
full-suite pass, or transfer a historical green result to changed source bytes.
Every other module above remains available; this page does not invent an
individual execution receipt where none is linked.

| Area and available modules | Recorded actual evidence | Scope and limitation |
|---|---|---|
| Historical complete then-current suite, including unified training/CLI, legacy/bundled inference and stage2 | [687 passed / exit0 at8c56594](../reports/heldout/README.md), [exact receipt](../reports/heldout/stage2-fullsuite-proof.json) and [log](../reports/heldout/stage2-fullsuite-pytest.log) | Synthetic data plus available original/bundled weights, CUDA uninitialized; not the current89-module inventory. Earlier682-pass/4-fail archived-tree/backend-fixture attempt and correction remain recorded. |
| Inference TTA and typed selection: `test_selection_tta`, `test_unified_selection`, `test_reviewed_heldout`, comparison/eligibility modules | [161 distinct focused checks across bounded runs, including header-type fix](../reports/final-selection-tta/README.md), [source/implementation proof](../reports/final-selection-tta/implementation-proof.json) | Exact NONE schema/arithmetic/RNG; odd rectangles/inverse operators; vessel-weighted joint AV fusion, zero-vessel fallback; FP32 despite BF16 metadata; strict bool/int descriptors. No actual corpus or NN forward in this proof. |
| Public prediction/CLI TTA: `test_highlevel_predict_tta`, CLI/selection regressions | [18+47 original cases](../reports/highlevel-predict-tta/README.md), [independent91-case exit0 at428755a](../reports/highlevel-predict-tta/root-verification.json) | Generated RGB/FOV and tiny CPU networks; custom/alias forwarding, metadata, unsupported legacy modes, default NONE bit/RNG parity. No actual model/VAL/TEST. |
| Full-image photometry/cache, HPO/config/training integration | [fd44 independent204 cases/exit0/28.51s](../reports/photometry-cache-root-review-20261111/fd44-verification.json); [HPO port106/92/33 recorded subsets](../reports/hpo-unified-photometry/verification.json); [merged3339038:61 cases/exit0](../reports/hpo-unified-photometry/root-merged-integration.json) | Synthetic operator/cache and tiny wiring fixtures; strict saved descriptor/backend/fingerprint, full-image-before-crop/TTA and preprocessing once. Old raw behavior preserved; no full training/quality inference from these tests. |
| Genuine curated completion adapter/family and old-family regression | [308 distinct focused cases and retained fixture corrections](../reports/curated-fd44-selection-proposal/README.md); [independent integrated105 cases plus actual LC CPU eligibility0](../reports/curated-fd44-selection-proposal/root-genuine-LC-eligibility.json) | Constructed original-chain tests separate from genuine LC model/index audit; actual LC operation118.04s/exit0 checks encoded bytes/role/config, **no forward**. Does not itself select/freeze/evaluate TEST. |
| Final stage2 curated dispatch and original4/reference coherence | [readiness audit](../reports/heldout-readiness-20261112/README.md), [exact test record](../reports/heldout-readiness-20261112/test-run-record.json) | Initial118 existing cases passed while6 new fixtures correctly failed output-path safety; fixture path corrected, six dispatch cases then passed1.90s/exit0.124 distinct successful cases, not a single rerun of124. No model/GT/TEST/GPU/runtime changes. |
| Frozen VAL bridge/comparators and saved TEST comparator | [64 synthetic bridge cases](../reports/data/val-domain-bridge-v1-v2/tests.json), [105 integrated comparison cases](../reports/completed-frozen-val-integration/verification.json), [70 saved-TEST comparator cases](../reports/reviewed-test-comparison/verification.json) | Exact photo/source/support/role and original-byte pins, paired count math and immutable inputs. Saved/synthetic count comparison is not actual held-out evaluation. |
| Selection/heldout/completion integration | [223 cases/exit0 atf5e1d5b](../reports/freeze-v3-pool72-actual-partial-20261111/root-merged-verification.json) | Synthetic temporary positive contexts and source/protocol negatives. That historical proof's then-pending V3 completion wording is not the current completion status. |
| Geometry, adapters and data identity | [87 CPU policy cases plus metadata inventory](../reports/v2-geometry/verification.json), [policy validation](../reports/data/corpus-v2-geometry-policy/validation.json), [archive audit](../reports/data/corpus-v2-archive-audit/verification.json) |87-case run does not read actual RGB/GT or train. Separate native TRAIN decoder/archive audit is preparation evidence, not a quality run or proof of all unknown patient identities. |
| Loss/sampling/training math | [57 CPU loss cases and2 native in-memory ASHA cases](../reports/v2-loss/verification.json); [cache E1 preservation](CACHE_WIDTH16_PROTOCOL.md) | Masked loss/default bitwise math versus a real-data training intervention are different evidence. Historical paired E1 model/optimizer/RNG/coverage equality uses its own2221-photo corpus; do not relabel it as current2529/5700. |
| Native proposal/sampler contracts | [11 cases with Optuna5 and no live-study change](../reports/fixed27-proposal/verification.json); coordinator modules above | Temporary native SQLite/Optuna tests are not an actual ask/tell or model training. Original native studies/sampler decisions remain operational evidence outside this inventory. |
| MATLAB/legacy numerical oracles | [CLAHE stored comparisons](../reports/legacy/matlab-clahe/comparison.json), [AV profile comparisons](../reports/legacy/matlab-av-profile/comparison.json), [FOV calibration](../reports/legacy/matlab-fov-calibration/comparison.json) | Fixture/oracle numerical evidence, not universal raw-path MATLAB certification or new quality evidence. Public tests use retained synthetic/golden inputs; MATLAB is not needed at test runtime. Real-photo full-chain residual qualifications remain. |
| Dense-count and fixed timing-cohort guards | [Six metadata checks](../reports/dense-vessel-support-20261112/README.md); [three synthetic cohort IO/identity tests](../reports/fixed-timing-input-cohort-20261112/test-exit.json), [cohort preparation scope](../reports/fixed-timing-input-cohort-20261112/README.md) | Dense README records six successful checks without a separate numeric tester-exit artifact. Cohort tester has observed exit0/3 cases; generated IO/framed identity and official-FOV selection guards. Preparing two TRAIN inputs is not a benchmark/model/GT/TEST run. |
| Final benchmark instrumentation | [51 targeted benchmark/manifest cases plus actual frozen182 metadata check](../reports/final-benchmark-preparation-20261112/frozen-reference-schema-check.json); historical34-case preparation is separately recorded in [the recipe](FINAL_BENCHMARK_RECIPE.md); [actual prior CPU tradeoff](../reports/benchmark/tpe72-tta-cpu-drive21/README.md) | Mocked CUDA synchronization/call counts and tiny RGB/FOV fixtures do not measure real latency. The separate actual CPU run has16 operations, child/outer0, no extra NN checks; see speed scope below. |

## Scientific validation evidence: separate from software tests

The current canonical domain is manifest17a2277c / content00d9ef19 /
geometry2591d846: **21 TRAIN sources,2529 photos,5700 annotation records**;
VAL770 photos/993 records/612 known groups. A run completing60 epochs is not proof
that its selected BEST independently owns all5700 contributions. Each role's
coverage must come from its own original checkpoint/proof. Known groups do not
establish complete patient independence. Actual held-out TEST has a separately
reviewed route; no final optimized candidate/TEST result is established here.

Source labels below retain original identities: source72/training3725ad32,
fd44/training2b707675, and the specific evaluation builds recorded in each report.
Public redacted/aggregate projections are **not original eligibility/gate bytes**.
Saved training VAL, independently fresh VAL and count-only postanalysis are
separate operations; same support does not imply bitwise-equal predictions.

| Model/protocol and original family | TRAIN/selected-role evidence | Actual validation and exits | Source/context and limitations |
|---|---|---|---|
| [Fixed original nearest reference182](../reports/frozen-original-val-nearest/README.md), [coherence with four-network528](../reports/heldout-readiness-20261112/README.md) | Original training file exposure unresolved; never assign current2529/5700 coverage to historical originals. | Full770 VAL report; [recorded producer return0](../reports/frozen-original-val-nearest/terminal.json). No new run here. | Same fixed dedicated vessel/Disc0/universal-cup∩Disc0/calibrated-AV operators.182 and528 are different complete data/source/backend envelopes; A/V has measured tiny drift. Historical raw/MATLAB/exposure qualifications remain. |
| [V2 sampling1 fixed terminal control](../reports/v2-control60-terminal-val/README.md) |60epochs/122880updates/491520samples; LAST60 owns2529/5700/21. BEST52 owns5699 and is not substituted. | Fresh CPU770 VAL, [producer0](../reports/v2-control60-terminal-val/producer-terminal.json). | Original TRAIN digest90752468, evaluationf0081907, own reviewed terminal context. Vessel/artery regress versus original despite higher macro. |
| [Vascular sampling2](../reports/vascular2-paired-val/README.md) and [sampling4](../reports/vascular4-paired-val/README.md), standalone | Sampling2 BEST59/LAST60; sampling4 BEST55/LAST60; each own2529/5700, completed60. | Original completion/QA and both fresh770-role producers/wrapper0 retained in linked summaries. | Source0b8762e/training85441afa; source/runtime dictionaries remain distinct. Matched E60 and VAL-selected BEST are different estimands; vessel deficit and adverse sources retained. |
| [Sampling4 inference TTA](../reports/inference-tta-vascular4/README.md) | Same immutable BEST55/991b117a; no new training. | Four full770 cases none/2/4/8, child0 and external0. | Historical experimental source, FP32/512/64/.5. Same-device none counts/support match prior512/64; no saved probability-bit parity. DRHAGIS AV adverse points; no current-gate retag or default change. |
| [First native TPE5](../reports/tpe72-completed-fresh-val/README.md), standalone72 | Full60/generic4bQA0/outer0. BEST58/20d4caab and LAST60 each own2529/5700/21. | Both fresh770 source72/b269 producers and outer0. | Saved/fresh small count drift, exact GT support; vessel below original. Higher macro/AV is not all-head superiority. |
| [First-TPE5 Old TTA](../reports/tpe72-inference-tta-val/README.md) and [rot4 threshold study](../reports/first-TPE-BEST58-rot4-threshold-20261111/README.md) | Same full-coverage BEST58; no weight updates. | Four modes full770/child+outer0; scalar sweep full770/child+outer0, same-array threshold contrasts. | Historical1f evaluation differs from native72 and later typed builds. Rot4/.35 is adaptive VAL, not TEST; RIMONE cup/source negatives and original vessel gap remain. NONE CPU latency is not TTA latency. |
| [Native V3 control0/trial4 saved counts](../reports/v3-completed-native-val-comparison/README.md), pool72 | Genuine original pool COMPLETE chains; control BEST59/LAST60 and trial4 BEST54/LAST60 each own5700. | Training/completion/selected-role eligibility verified; saved770 sidecar counts compared **without a new forward**. | source72/training3725. Saved counts are not a newly enriched final-runtime report or a standalone envelope. Original vessel deficits remain. |
| [V3 trial4 fresh TTA](../reports/v3-trial4-inference-tta-val/README.md), [rot4/.4 threshold sweep](../reports/native-v3-best54-rot4-thresholds/README.md) | Genuine pool BEST54/21c1b3c6 plus separate LAST60; each own2529/5700. | Four BEST modes plus LAST-none, each770/child0/external0. Corrected threshold run full770/child+outer0; initial import failure retained. | Fresh318e342/evaluation198fdd47 separate from training3725.8−4 macro point negative/CI crosses0, IOSTAR AV adverse; .4 cup/source regressions. No equivalence/selection/default claim. |
| [Raw256](../reports/curated-raw256-fresh-val-20261111/README.md), curated_fd44 | Original full60/PBS0; BEST52/132c7879 and LAST60 each own2529/5700, both CPU audits0. | Both typed fresh770 cases/child0/outer0. | fd44 training2b707675, evaluation1475c264. Full saved raw config retained; parent deficits and tiny saved/fresh drift disclosed, not isolated patch causality. |
| [Raw384](../reports/curated-raw384-typed-fresh-VAL-20261111/README.md), curated_fd44 | Original full60; BEST59/f6ef880c and LAST60 each own2529/5700; original index/QA and current CPU proofs. | Both fresh770 cases0; e3b outer0; source/input pins unchanged. |6d3593f/f13 evaluation versus fd44 training. Vessel below parent/original although AV/disc improve; saved/fresh counts differ. No report/source retag. |
| [Raw512 NONE](../reports/raw512-typed-fresh-val-20261112/README.md), curated_fd44 | Original full60/PBS0, BEST50/521b8881 and LAST60 each own2529/5700, CPU proofs0. | Both fresh770 cases0 and original outer0. | Same typed6d/f13/NONE/.5 runtime as raw384. Macro declines while dense-vessel support improves versus raw384; dense/global vessel still below original; AVRDB/source tradeoffs retained. |
| [Raw512 BEST50 rot4/.35](../reports/raw512-BEST50-rot4-035-fresh-val-20261112/README.md), curated_fd44 | Same immutable own-full BEST50, no retraining/architecture change. | One full770 case0 and outer0; stage621.89s. | Typed6d/f13. TTA and threshold both change: joint contrast only. Gains versus same NONE do not remove original dense/global-vessel deficit; RIMONE/CHASE/DRHAGIS/AVRDB adverses remain. Invocation duration is not a latency benchmark. |
| [CLAHE384](../reports/clahe384-typed-fresh-val-20261112/README.md), curated_fd44 | Full60/PBS0; BEST58 and LAST60 each own2529/5700; both CPU role checks0. | Both fresh770 cases0, outer0; initial path-only staging failure preserved. | Full original CLAHE descriptor/backend/cache config retained, online full-image transform;6d/f13. Every pooled head and AV regresses against matched raw384; no promotion. |
| [LC384](../reports/curated-lc384-best60-fresh-val-20261111/README.md), curated_fd44 | Training completed60; BEST60/dfef53e4 and LAST60 both own2529/5700; [genuine CPU role audit](../reports/curated-fd44-completion/genuine-lc384-two-role-audit.json). | **BEST full770 child0 only. Planned LAST timed out, no complete report; whole outer1.** | Source fd44/training2b; complete nonraw LC config retained. Vessel/cup regress, disc improves. Timeout does not undo training, and BEST0 does not make the two-role operation successful. No inferred LAST quality. |
| [Native TPE6](../reports/native-TPE6-matched-fresh-VAL-20261111/README.md), standalone72 | Actual full60/QA0/outer0; BEST52/LAST60 each own2529/5700. | Both matched native source72/b269 fresh770 cases/outer0. | Own complete config/protocol retained; multiple LR/WD/sampling changes and one seed. Regressions versus TPE5, no model promotion. |
| [Native TPE7 LAST60](../reports/TPE7-LAST60-matched-val-20261112/README.md), standalone72 | Full60/122880/491520 and QA/outer0. LAST60 owns5700; BEST46 owns5699 and is **ineligible/not freshly evaluated here**. | LAST-only770 child0/outer0. | Native72/b269/FP32/NONE. Dense gains versus TPE5/6 but original dense/global vessel intervals remain negative; higher macro is not all-task recovery. |
| [Controlled vessel-loss2](../reports/loss2-LAST60-matched-val-20261112/README.md), standalone72 | Full60/QA0/outer0; BEST58/LAST60 each own2529/5700/21. LAST is the selected comparison role. | LAST-only770 child0/outer0 at05:21:54UTC. Internal planned0 is not the outer witness. | Matched source72/b269; only vessel_loss_weight1→2 versus TPE5 LAST. Prespecified dense gain+.00555 has positive paired CI, but original dense gap remains negative and cup source-macro regresses. No promotion. |

Full-domain paired analyses use2000 source-stratified known-group draws/seed42,
unless the linked study declares otherwise. Dense-vessel supplement excludes
only positive-only AVRDB:87photos/55groups/8sources, **not** all612 groups. Pooled
Dice, source/head macro, conditional AV and vessel-gated end-to-end AV measure
different things. Source intervals are exploratory/unadjusted for multiplicity;
one seed is not training-seed uncertainty or clinical equivalence. Preserve all
adverse heads/sources, exact integer counts/support and original byte provenance.

The [17-row all-head overview](../reports/all-head-candidate-comparison-20261112/README.md)
and [current status](EXPERIMENTS_CURRENT.md) are the compact navigation layer.
Native8 and width24 remain unfinished operational experiments at this documentation
snapshot; they are not successful validation rows or selected models. Earlier
pilot/partial-budget reports remain historical evidence in `reports/`, never
substitutes for full60/current-domain/own-role eligibility. Product release parity
and private Meta operational tests have separate catalogs.

## Speed and size provenance (read-only audit)

The historical [resident CPU tradeoff](../reports/benchmark/tpe72-tta-cpu-drive21/README.md)
used **the strong four networks**, not the old two-network proxy. The actually
executed strong script hashes3a0324e0…ac1aae; the ML package was independently
pinned1f723ad9. Its three single-net loads are dedicated vessel/Disc0/universal,
plus a separate AV load. Cup intersects the SAME-forward Disc0 before nearest
restoration; AV is calibrated from predicted vessels and supplied FOV. Original
private producer commit53be29a is referenced by the [sanitized terminal](../reports/benchmark/tpe72-tta-cpu-drive21/producer-terminal.json),
with exact driver/launch/runtime/timing SHA pins.

Same-run resident medians are original59.594548s, TPE NONE8.748764s,
rot433.199672s and dihedral865.531443s; actual sixteen operations are four groups
of1warmup+3measured calls. Saved source counts **all four** originals for
5,943,312 parameters/25,068,700 checkpoint bytes, versus TPE1,965,637/8,894,002.
These are actual component/checkpoint totals, not two-proxy sizes, model
compression, whole-wheel size or RAM reduction. Full universal compute/parameters
count even though only its cup output is used. Five output masks are not five nets.

`benchmark_latency.py`'s old entrypoint is a universal+AV two-net proxy. Only its
generic synchronized `measure` helper timed the strong closure; the proxy
entrypoint was not called. Fixed order/warm caches/one DRIVE TRAIN photograph,
one CPU thread/FP32 and three timed samples limit this result. IO, model load,
output serialization/probability arrays and CLI startup are excluded. The product
API does additional work; no full-product speed claim follows. The .5 clocks are
not a new .35 measurement, a TTA quality guarantee or timing of a later candidate.
See the [separate final recipe](FINAL_BENCHMARK_RECIPE.md); it is prepared code,
not a benchmark of a final selected checkpoint. No source/doc speed correction
was needed after checking the saved clocks, inventories and original source.


Additional real-data verification 2026-11-12: [final-source original full770VAL reproduction](../reports/original-f13-Old-VAL-20261112/README.md), actual child0/outer0,
exact GTsupport unchanged;807.9985s. This is validation, not TEST or a timing benchmark.
