"""Synthetic saved counts/metadata only; no real TEST data or inference."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "reviewed_test_comparison", ROOT / "scripts/compare_reviewed_test.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case(tmp_path):
    rows = []
    for split, photo, task in [
        ("train", "tr", "av"),
        ("val", "va", "av"),
        ("test", "te0", "av"),
        ("test", "te0", "disc_cup"),
        ("test", "te1", "av"),
        ("test", "te1", "disc_cup"),
    ]:
        rows.append(
            {
                "id": photo + "-" + task,
                "photo_id": photo,
                "source_id": photo,
                "split_group_id": photo,
                "dataset": "source",
                "split": split,
                "task": task,
                "preprocessing_profile": "legacy25_raw",
                "pixels_per_degree": 25,
            }
        )
    manifest = m.canonical(rows)
    groups, counts, original_inventory, unified_inventory = m.inventory(manifest)
    data = {
        "manifest_sha256": m.sha(manifest),
        "prepared_content_sha256": "d" * 64,
        "fingerprint_schema": 2,
        "eligible_photo_counts": counts,
        "split_inventory_sha256": unified_inventory,
    }
    configuration = {
        "purpose": "full_data",
        "epochs": 60,
        "epoch_samples": 8192,
        "coverage_first": True,
        "legacy_initialization": None,
        "allow_historical_exposure": False,
        "max_steps": None,
        "batch_size": 4,
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25,
        "train_view": "patch",
        "tile_size": 512,
        "overlap": 64,
    }
    config = {"values": configuration, "sha256": m.sha(m.canonical(configuration))}
    protocol = {
        "threshold": 0.5,
        "geometry": "canonical_prepared",
        "normalization": "RGB/255-0.5",
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25,
        "channels": list(m.CHANNELS),
        "mode": "tiled",
        "tile_size": 512,
        "overlap": 64,
        "metric_support": {"test": "fixture"},
        "annotation_policy": "fixture-first-sorted",
    }
    selected = {
        "schema_version": 1,
        "kind": "unified_heldout_selection",
        "decision": {
            "recorded_at_utc": "2026-01-02T00:00:00Z",
            "reviewer": "fixture",
            "rationale": "synthetic fixture",
        },
        "checkpoint": {
            "sha256": "e" * 64,
            "epoch": 56,
            "steps": 56 * 2048,
            "lineage": {"fixture": "unified"},
            "training_selection_score": 0.7,
            "configuration": config,
            "channels": list(m.CHANNELS),
        },
        "source": {"evaluation_source_sha256": "1" * 64, "training_source_sha256": "2" * 64},
        "canonical_data": data,
        "inference_protocol": protocol,
        "training_completion": {
            "planned_epochs": 60,
            "completed_epochs": 60,
            "last_epoch": 60,
            "last_checkpoint_sha256": "f" * 64,
            "last_steps": 122880,
            "best_epoch": 56,
            "best_training_selection_score": 0.7,
            "metrics_csv_sha256": "3" * 64,
            "training_summary_sha256": "4" * 64,
            "coverage_sha256": "5" * 64,
        },
        "validation_evidence": {
            "sha256": "6" * 64,
            "evaluated_at_utc": "2026-01-01T00:00:00Z",
            "selection_metric": "macro_dataset_mean_dice",
            "selection_score": 0.7,
        },
        "boundaries": copy.deepcopy(m.BOUNDARIES),
    }
    selection_bytes = m.canonical(selected) + b"\n"
    models = {
        k: c * 64 for k, c in [("universal", "7"), ("vessel", "8"), ("disc", "9"), ("av", "a")]
    }
    policy = {
        "schema_version": 1,
        "kind": "frozen_original_reference_policy",
        "policy": {"name": "fixture-original-fixed-recipe", "model_sha256": models},
        "data": {
            "manifest_sha256": data["manifest_sha256"],
            "prepared_content_sha256": data["prepared_content_sha256"],
            "photos_sha256": "b" * 64,
            "geometry_policy_sha256": "c" * 64,
            "eligible_photo_counts": counts,
            "inventory_sha256": original_inventory,
        },
        "source": {
            "runtime": {"evaluation_source_sha256": "0" * 64},
            "scripts": {"fixture": "1" * 64},
        },
        "validation_report_sha256": "2" * 64,
        "selected_unified_selection_sha256": m.sha(selection_bytes),
        "rationale": "synthetic original policy",
        "frozen_at_utc": "2026-01-03T00:00:00Z",
    }
    entries = []
    for pid, records in groups["test"].items():
        channels = {name: {"tp": 6, "fp": 1, "fn": 4, "valid_pixels": 20} for name in m.CHANNELS}
        entries.append(
            {
                "photo_id": pid,
                "record_ids": [r["id"] for r in records],
                "dataset": "source",
                "channel_datasets": dict.fromkeys(m.CHANNELS, "source"),
                "channels": channels,
                "av_conditional_counts": {
                    "correct": 16,
                    "scored_pixels": 20,
                    "eligible_pixels": 20,
                    "end_to_end_correct": 12,
                },
            }
        )
    original = {
        "schema_version": 1,
        "family": "frozen_original_reference",
        "preview_only": False,
        "split": "test",
        **policy["data"],
        "source": copy.deepcopy(policy["source"]),
        "policy_spec": copy.deepcopy(policy["policy"]),
        "photos": 2,
        "checkpoint_sha256": models["universal"],
        "vessel_checkpoint_sha256": models["vessel"],
        "disc_checkpoint_sha256": models["disc"],
        "av_checkpoint_sha256": models["av"],
        "per_photo": copy.deepcopy(entries),
        "reviewed_policy_sha256": m.sha(m.canonical(policy)),
        "evaluated_at_utc": "2026-01-04T00:00:00Z",
    }
    candidate = {
        "schema_version": 1,
        "kind": "unified_reviewed_heldout_evaluation",
        "selection_sha256": m.sha(selection_bytes),
        "execution_context": {
            "route": "explicit_reviewed_stage2_script",
            "explicit_execution_requested": True,
            "stage1_test_evaluation_enabled": False,
            "prior_blindness_proven": False,
            "reviewer_identity_authenticated": False,
            "new_model_selection_performed": False,
        },
        "selection_decision": copy.deepcopy(selected["decision"]),
        "training_completion": copy.deepcopy(selected["training_completion"]),
        "validation_evidence": copy.deepcopy(selected["validation_evidence"]),
        "checkpoint": {
            k: selected["checkpoint"][k] for k in ("sha256", "epoch", "steps", "lineage")
        },
        "configuration": copy.deepcopy(config),
        "source": copy.deepcopy(selected["source"]),
        "canonical_data": copy.deepcopy(data),
        "inference_protocol": copy.deepcopy(protocol),
        "evaluated_at_utc": "2026-01-04T00:00:00Z",
        "metrics": {
            "family": "unified",
            "purpose": "full_data",
            "split": "test",
            "photos": 2,
            "records": 4,
            "threshold": 0.5,
            "supervised_channels": [True] * 5,
            "metric_support": protocol["metric_support"],
            "evaluation_policy": protocol["annotation_policy"],
            "per_photo": copy.deepcopy(entries),
        },
    }
    paths = {
        name: tmp_path / (name + ".json")
        for name in ["original", "policy", "candidate", "selection", "manifest"]
    }
    values = {
        "original": original,
        "policy": policy,
        "candidate": candidate,
        "selection": selected,
        "manifest": rows,
    }
    for name, value in values.items():
        paths[name].write_bytes(selection_bytes if name == "selection" else m.canonical(value))
    return values, paths


def run(case, **pins):
    _, p = case
    return m.compare(
        original_report=p["original"],
        original_policy=p["policy"],
        candidate_report=p["candidate"],
        selection=p["selection"],
        manifest=p["manifest"],
        trusted_original_policy_sha256=pins.get("policy", m.sha(p["policy"].read_bytes())),
        trusted_selection_sha256=pins.get("selection", m.sha(p["selection"].read_bytes())),
    )


def save(case, name):
    values, paths = case
    paths[name].write_bytes(m.canonical(values[name]))


def test_consistent_differing_pipeline_sources_compare_test_and_preserve_input_bytes(case):
    values, paths = case
    before = {k: p.read_bytes() for k, p in paths.items()}
    assert values["original"]["source"] != values["candidate"]["source"]
    result = run(case)
    assert result["split"] == "test" and result["photos"] == 2
    assert result["bootstrap"]["draws"] == 2000 and result["bootstrap"]["seed"] == 42
    assert result["whole_input_sha256"]["candidate_report"] == m.sha(before["candidate"])
    assert result["candidate_wrapped_report_sha256"] != result["derived_candidate_view_sha256"]
    assert result["candidate_report_sha256"] == result["derived_candidate_view_sha256"]
    assert "by_dataset" in result and "av_accuracy" in result
    assert all(p.read_bytes() == before[k] for k, p in paths.items())


@pytest.mark.parametrize(
    "target,path,bad",
    [
        ("original", ["split"], "val"),
        ("candidate", ["metrics", "split"], "val"),
        ("candidate", ["selection_sha256"], "0" * 64),
        ("policy", ["selected_unified_selection_sha256"], "0" * 64),
        ("original", ["reviewed_policy_sha256"], "0" * 64),
        ("original", ["source"], {}),
        ("candidate", ["source"], {}),
        ("original", ["av_checkpoint_sha256"], "0" * 64),
        ("original", ["policy_spec"], {}),
        ("candidate", ["checkpoint", "epoch"], 60),
        ("candidate", ["configuration", "sha256"], "0" * 64),
        ("candidate", ["metrics", "threshold"], 0.4),
        ("candidate", ["training_completion", "last_epoch"], 27),
        ("candidate", ["selection_decision", "reviewer"], "other"),
        ("candidate", ["canonical_data", "prepared_content_sha256"], "0" * 64),
        ("candidate", ["metrics", "photos"], 1),
        ("candidate", ["metrics", "records"], 3),
        ("candidate", ["execution_context", "explicit_execution_requested"], False),
    ],
)
def test_mismatched_review_policy_selection_source_and_test_domain_rejected(
    case, target, path, bad
):
    values, _ = case
    node = values[target]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = bad
    save(case, target)
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize("name", ["policy", "selection"])
def test_external_original_artifact_byte_sha_required(case, name):
    with pytest.raises(ValueError, match="Externally reviewed"):
        run(case, **{name: "0" * 64})


def test_record_identity_and_source_attribution_match_trusted_manifest(case):
    values, _ = case
    values["candidate"]["metrics"]["per_photo"][0]["record_ids"] = ["wrong"]
    save(case, "candidate")
    with pytest.raises(ValueError, match="record IDs"):
        run(case)


def test_cli_preserves_existing_output(case, tmp_path, monkeypatch):
    _, paths = case
    output = tmp_path / "comparison.json"
    output.write_text("preserve")
    args = ["compare"]
    for key, name in [
        ("original-report", "original"),
        ("original-policy", "policy"),
        ("candidate-report", "candidate"),
        ("selection", "selection"),
        ("manifest", "manifest"),
    ]:
        args.extend(["--" + key, str(paths[name])])
    args.extend(
        [
            "--trusted-original-policy-sha256",
            m.sha(paths["policy"].read_bytes()),
            "--trusted-selection-sha256",
            m.sha(paths["selection"].read_bytes()),
            "--output",
            str(output),
        ]
    )
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "preserve"


def rebind_reviewed_artifacts(case):
    values, paths = case
    save(case, "selection")
    digest = m.sha(paths["selection"].read_bytes())
    values["policy"]["selected_unified_selection_sha256"] = digest
    values["candidate"]["selection_sha256"] = digest
    save(case, "policy")
    values["original"]["reviewed_policy_sha256"] = m.sha(paths["policy"].read_bytes())
    save(case, "original")
    save(case, "candidate")


@pytest.mark.parametrize(
    "defect", ["partial", "steps", "threshold", "config", "manifest", "ids", "source_digest"]
)
def test_self_consistent_rebound_metadata_cannot_evade_completion_protocol_and_domain_checks(
    case, defect
):
    values, _ = case
    selected, candidate = values["selection"], values["candidate"]
    if defect == "partial":
        selected["training_completion"]["last_epoch"] = candidate["training_completion"][
            "last_epoch"
        ] = 27
    elif defect == "steps":
        selected["checkpoint"]["steps"] = candidate["checkpoint"]["steps"] = 1
    elif defect == "threshold":
        selected["inference_protocol"]["threshold"] = candidate["inference_protocol"][
            "threshold"
        ] = candidate["metrics"]["threshold"] = 0
    elif defect == "config":
        config = selected["checkpoint"]["configuration"]
        config["values"]["epochs"] = 3
        config["sha256"] = m.sha(m.canonical(config["values"]))
        candidate["configuration"] = copy.deepcopy(config)
    elif defect == "manifest":
        selected["canonical_data"]["manifest_sha256"] = candidate["canonical_data"][
            "manifest_sha256"
        ] = "0" * 64
    elif defect == "ids":
        candidate["metrics"]["per_photo"][0]["photo_id"] = "foreign-test-photo"
    elif defect == "source_digest":
        selected["source"]["training_source_sha256"] = candidate["source"][
            "training_source_sha256"
        ] = "missing"
    rebind_reviewed_artifacts(case)
    with pytest.raises(ValueError):
        run(case)


def test_paired_positive_gt_support_mismatch_rejected_without_gt_access(case):
    values, _ = case
    values["candidate"]["metrics"]["per_photo"][0]["channels"]["disc"]["fn"] += 1
    save(case, "candidate")
    with pytest.raises(ValueError, match="ground-truth support"):
        run(case)


def test_helper_imports_without_torch_or_execution_gate_modules(monkeypatch):
    import builtins
    import runpy

    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name in {"deepretina.selection", "deepretina.heldout"}
        ):
            raise AssertionError("Metadata helper imported execution code: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    runpy.run_path(
        str(ROOT / "scripts/compare_reviewed_test.py"), run_name="metadata_only_import_probe"
    )


def test_existing_metric_identity_cannot_be_silently_retagged(case):
    values, _ = case
    values["candidate"]["metrics"]["checkpoint_sha256"] = "0" * 64
    save(case, "candidate")
    with pytest.raises(ValueError, match="contradicts reviewed wrapper"):
        run(case)
