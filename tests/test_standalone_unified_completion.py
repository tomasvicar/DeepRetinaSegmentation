"""Synthetic E60 metadata/tensors, no forward, GT decoder or actual TEST input."""

import copy
import importlib.util
import json
import math
import runpy
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from deepretina import selection as gate
from deepretina.integrity import prepared_content_fingerprint
from deepretina.unified import UnifiedConfig, _training_source_sha256, build_unified

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "standalone_completion", ROOT / "scripts/verify_unified_completion.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def forbid_forward_gt_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No forward/CUDA/GT decoder permitted")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    import numpy as np

    monkeypatch.setattr(np, "load", forbidden)
    torch.set_num_threads(1)


@pytest.fixture
def case(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    # Deliberately invalid image/NPZ bytes: any decoder would fail this fixture.
    for name in ["image.bin", "labels.bin", "fov.bin"]:
        (canonical / name).write_bytes(b"encoded fixture, never decode")
    rows = []
    for i in range(5700):
        photo = i % 2529
        rows.append(
            {
                "id": f"r{i}",
                "photo_id": f"p{photo}",
                "source_id": f"s{photo}",
                "dataset": f"d{photo % 21}",
                "task": "av" if photo % 2 else "disc_cup",
                "split": "train",
                "image": "image.bin",
                "mask": "labels.bin",
                "preprocessing_profile": "legacy25_raw",
                "pixels_per_degree": 25,
                "preparation_max_size": None,
            }
        )
    for split in ["val", "test"]:
        rows.append(
            {
                "id": split,
                "photo_id": split,
                "source_id": split,
                "dataset": "d0",
                "task": "av",
                "split": split,
                "image": "image.bin",
                "mask": "labels.bin",
                "preprocessing_profile": "legacy25_raw",
                "pixels_per_degree": 25,
                "preparation_max_size": None,
            }
        )
    manifest = canonical / "manifest.json"
    manifest.write_bytes(m.canonical(rows))
    (canonical / "photos.json").write_bytes(m.canonical([{"image": "image.bin", "fov": "fov.bin"}]))
    monkeypatch.setattr(m, "MANIFEST", m.sha(manifest.read_bytes()))
    (canonical / "_SUCCESS").write_text(m.MANIFEST)
    content = prepared_content_fingerprint(manifest)
    monkeypatch.setattr(m, "CONTENT", content)
    (canonical / "integrity.json").write_bytes(
        m.canonical(
            {
                "fingerprint_schema": 2,
                "manifest_sha256": m.MANIFEST,
                "prepared_content_sha256": content,
                "preview_only": False,
            }
        )
    )
    cfg = UnifiedConfig(
        base_channels=2,
        depth=1,
        norm="group",
        epochs=60,
        epoch_samples=8192,
        batch_size=4,
        workers=0,
        patch_size=32,
        tile_size=32,
        overlap=8,
        device="cpu",
        purpose="full_data",
        selection_metric="macro_dataset_mean_dice",
        vascular_sampling_weight=2,
    )
    model = build_unified(cfg)
    photo_visits = {f"p{i}": 60 for i in range(2529)}
    photo_visits["p0"] += 491520 - sum(photo_visits.values())
    by_dataset = {
        f"d{i}": {
            "eligible_photos": len(
                {r["photo_id"] for r in rows if r["split"] == "train" and r["dataset"] == f"d{i}"}
            ),
            "seen_photos": len(
                {r["photo_id"] for r in rows if r["split"] == "train" and r["dataset"] == f"d{i}"}
            ),
        }
        for i in range(21)
    }
    coverage = {
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "photo_visits": photo_visits,
        "annotation_visits": {f"r{i}": 1 for i in range(5700)},
        "by_dataset": by_dataset,
    }
    policy_raw = (ROOT / "reports/data/corpus-v2-geometry-policy/policy.json").read_bytes()
    policy = {
        "format_version": 1,
        "sha256": m.sha(policy_raw),
        "document_json": policy_raw.decode(),
    }
    common = {
        "format_version": 1,
        "family": "unified",
        "lineage": model.lineage,
        "architecture": model.architecture_spec,
        "channels": list(gate.CHANNELS),
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "manifest_sha256": m.MANIFEST,
        "prepared_content_sha256": content,
        "training_source_sha256": _training_source_sha256(),
        "supervised_channels": [True] * 5,
        "inference_max_size": None,
        "normalization": "RGB/255-0.5",
        "purpose": "full_data",
        "probability_fusion": "joint_av_mixture_then_conditional",
        "coverage": coverage,
        "derived_cache": None,
        "geometry_policy": policy,
    }

    def validation(score):
        return {
            "family": "unified",
            "split": "val",
            "selection_metric": cfg.selection_metric,
            "selection_score": score,
            "macro_dataset_mean_dice": score,
        }

    best = dict(
        common,
        epoch=56,
        steps=56 * 2048,
        val_selection_score=0.8,
        validation_metrics=validation(0.8),
    )
    last = dict(
        common, epoch=60, steps=122880, val_selection_score=0.6, validation_metrics=validation(0.6)
    )
    optimizer = {
        "state": {
            i: {
                "step": torch.tensor(122880.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }
            for i, p in enumerate(model.parameters())
        },
        "param_groups": [
            {
                "params": list(range(len(list(model.parameters())))),
                "lr": cfg.lr * cfg.min_lr_ratio,
                "weight_decay": cfg.weight_decay,
            }
        ],
    }
    last["training_state"] = {
        "coverage": coverage,
        "optimizer": optimizer,
        "scheduler": {"T_max": 60, "last_epoch": 60},
        "best_validation_score": 0.8,
        "selection_metric": cfg.selection_metric,
    }
    run = tmp_path / "run"
    run.mkdir()
    torch.save(best, run / "best.pt")
    torch.save(last, run / "last.pt")
    csv = "epoch,steps,lr,val_selection_score,val_macro_dataset_mean_dice\n"
    for epoch in range(1, 61):
        score = 0.8 if epoch == 56 else 0.6
        lr = (
            cfg.lr * cfg.min_lr_ratio
            + (cfg.lr - cfg.lr * cfg.min_lr_ratio) * (1 + math.cos(math.pi * (epoch - 1) / 60)) / 2
        )
        csv += f"{epoch},{epoch * 2048},{lr},{score},{score}\n"
    (run / "metrics.csv").write_text(csv)
    for name, value in [
        (
            "training_summary.json",
            {
                "completed_budget": True,
                "steps": 122880,
                "best_validation_score": 0.8,
                "selection_metric": cfg.selection_metric,
            },
        ),
        (
            "run.json",
            {
                key: best[key]
                for key in [
                    "config",
                    "manifest_sha256",
                    "prepared_content_sha256",
                    "training_source_sha256",
                    "derived_cache",
                    "geometry_policy",
                ]
            },
        ),
        ("coverage.json", coverage),
        ("validation.json", best["validation_metrics"]),
    ]:
        (run / name).write_bytes(m.canonical(value))
    source = ROOT.resolve()
    protocol = {
        "schema_version": 1,
        "kind": "reviewed_standalone_unified_protocol",
        "source": {
            "root": str(source),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
            "execution_code_sha256": runpy.run_path(
                str(source / "scripts/unified_hpo.py"), run_name="synthetic_fixture"
            )["code_digest"](source),
            "training_source_sha256": _training_source_sha256(),
        },
        "run": str(run),
        "config": asdict(cfg),
        "data": {
            "manifest_sha256": m.MANIFEST,
            "prepared_content_sha256": content,
            "geometry_policy_sha256": m.POLICY,
            "derived_cache": None,
        },
        "budget": copy.deepcopy(m.BUDGET),
        "inference": {"threshold": 0.5, "tile_size": 32, "overlap": 8},
    }
    terminal = {
        "schema_version": 1,
        "kind": "bounded_unified_supervisor_terminal",
        "status": "completed",
        "returncode": 0,
        "timed_out": False,
        "test_evaluation_performed": False,
        "completed_epochs": 60,
        "source": protocol["source"],
        "run": str(run),
        "budget": protocol["budget"],
        "data": protocol["data"],
        "config_sha256": m.sha(m.canonical(protocol["config"])),
        "steps": 122880,
        "delivered_samples": 491520,
    }
    paths = {
        "protocol": tmp_path / "protocol.json",
        "terminal": tmp_path / "original-supervisor-terminal.json",
        "output": tmp_path / "completion.json",
    }
    result = {
        "protocol": protocol,
        "terminal": terminal,
        "paths": paths,
        "source": source,
        "run": run,
        "manifest": manifest,
    }
    rebind(result)
    return result


def rebind(case):
    p = case["paths"]
    p["protocol"].write_bytes(m.canonical(case["protocol"]) + b"\n")
    case["terminal"]["protocol_sha256"] = m.sha(p["protocol"].read_bytes())
    case["terminal"]["artifact_sha256"] = {
        name: m.sha((case["run"] / name).read_bytes()) for name in m.ARTIFACTS
    }
    p["terminal"].write_bytes(m.canonical(case["terminal"]) + b"\n")


def run(case, **overrides):
    p = case["paths"]
    return m.verify(
        source=case["source"],
        run=case["run"],
        manifest=case["manifest"],
        protocol_path=p["protocol"],
        trusted_protocol_sha256=overrides.get("protocol", m.sha(p["protocol"].read_bytes())),
        terminal_path=p["terminal"],
        trusted_terminal_sha256=overrides.get("terminal", m.sha(p["terminal"].read_bytes())),
        output=p["output"],
    )


def test_synthetic_finite_e060_passes_real_safe_load_encoded_hash_and_existing_flat_gate(case):
    result = run(case)
    assert (
        result["kind"] == "standalone_unified_completion_qa"
        and result["all_eligible_annotations_contributed"] is True
    )
    assert (
        result["contributing_annotation_records"] == 5700 and result["minimum_photo_visits"] == 60
    )
    assert json.loads(result["original_supervisor_terminal_text"]) == case["terminal"]
    spec = importlib.util.spec_from_file_location(
        "compare_complete", ROOT / "scripts/compare_completed_val.py"
    )
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    evidence = {
        "checkpoint": {"sha256": result["checkpoint_sha256"]["best.pt"], "epoch": 56},
        "data": {
            "manifest_sha256": m.MANIFEST,
            "prepared_content_sha256": m.CONTENT,
            "eligible_photo_counts": {"train": 2529},
        },
        "source": {"training_source_sha256": result["checkpoint_training_source_sha256"]},
    }
    comparison.verify_completion(result, evidence, expected_train_photo_count=2529)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("returncode", 1),
        ("timed_out", True),
        ("completed_epochs", 27),
        ("test_evaluation_performed", True),
        ("kind", "pool_receipt"),
        ("run", "/foreign/run"),
    ],
)
def test_supervisor_terminal_must_be_actual_clean_e060_and_bound(case, field, bad):
    case["terminal"][field] = bad
    rebind(case)
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize("artifact", ["protocol", "terminal"])
def test_external_original_byte_hash_is_required(case, artifact):
    with pytest.raises(ValueError, match="Externally reviewed"):
        run(case, **{artifact: "0" * 64})


@pytest.mark.parametrize(
    "defect",
    [
        "partial",
        "nonfinite",
        "missing_annotation",
        "foreign_annotation",
        "short_photo",
        "optimizer_shape",
        "optimizer_step",
        "scheduler",
        "config",
    ],
)
def test_even_rebound_supervisor_cannot_override_inconsistent_checkpoint_evidence(case, defect):
    path = case["run"] / "last.pt"
    last = torch.load(path, weights_only=True)
    if defect == "partial":
        last["epoch"] = 27
    elif defect == "nonfinite":
        next(iter(last["state_dict"].values())).view(-1)[0] = float("nan")
    elif defect in {"missing_annotation", "foreign_annotation"}:
        visits = last["coverage"]["annotation_visits"]
        value = visits.pop("r0")
        if defect == "foreign_annotation":
            visits["foreign"] = value
        last["coverage"]["contributing_annotation_records"] = len(visits)
    elif defect == "short_photo":
        last["coverage"]["photo_visits"]["p1"] = 59
    elif defect == "optimizer_shape":
        last["training_state"]["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif defect == "optimizer_step":
        last["training_state"]["optimizer"]["state"][0]["step"] = torch.tensor(1.0)
    elif defect == "scheduler":
        last["training_state"]["scheduler"]["last_epoch"] = 59
    elif defect == "config":
        last["config"]["vascular_sampling_weight"] = 1
    torch.save(last, path)
    rebind(case)
    with pytest.raises(ValueError):
        run(case)


def test_real_e027_pool_stage_cannot_become_standalone_terminal_without_opening_checkpoints():
    raw = (ROOT / "tests/reference/standalone_completion/actual-e027-stripped.json").read_bytes()
    terminal = json.loads(raw)
    with pytest.raises(ValueError, match="standalone supervisor"):
        m.check_contracts(
            {"schema_version": 1, "kind": "reviewed_standalone_unified_protocol"},
            terminal,
            protocol_sha="none",
            source=Path("/unused"),
            run=Path("/unused"),
        )


def test_existing_output_is_preserved(case):
    case["paths"]["output"].write_text("preserve")
    with pytest.raises(ValueError, match="existing completion"):
        run(case)
    assert case["paths"]["output"].read_text() == "preserve"


def test_actual_git_head_and_execution_byte_guards(case):
    pins = copy.deepcopy(case["protocol"]["source"])
    pins["commit"] = "0" * 40
    with pytest.raises(ValueError, match="source commit"):
        m.check_source(case["source"], pins)
    pins = copy.deepcopy(case["protocol"]["source"])
    pins["execution_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="executable source"):
        m.check_source(case["source"], pins)


@pytest.mark.parametrize(
    "key", ["manifest_sha256", "prepared_content_sha256", "geometry_policy_sha256"]
)
def test_rebound_reviewed_protocol_cannot_change_fixed_v2_corpus_policy(case, key):
    case["protocol"]["data"][key] = "0" * 64
    rebind(case)
    with pytest.raises(ValueError, match="canonical/policy"):
        run(case)


def test_final_artifact_hash_is_bound_before_checkpoint_load(case):
    with (case["run"] / "best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        run(case)


def test_pool_or_waiter_receipt_is_never_a_supervisor_terminal(case):
    case["terminal"]["kind"] = "unified_pool_waiter_terminal"
    rebind(case)
    with pytest.raises(ValueError, match="standalone supervisor"):
        run(case)


def test_output_cannot_be_a_dangling_symlink(case, tmp_path):
    case["paths"]["output"].symlink_to(tmp_path / "absent-target")
    with pytest.raises(ValueError, match="existing completion"):
        run(case)
    assert not (tmp_path / "absent-target").exists()


def test_actual_git_clean_guard_uses_a_real_temporary_git_repository(tmp_path):
    source = tmp_path / "source-git"
    source.mkdir()
    (source / "src").mkdir()
    (source / "scripts").mkdir()
    (source / "src/model.py").write_text("value = 1\n")
    (source / "pyproject.toml").write_text("")
    (source / "uv.lock").write_text("")
    for args in [
        ["init", "-q"],
        ["config", "user.email", "fixture@example.invalid"],
        ["config", "user.name", "fixture"],
        ["add", "."],
        ["commit", "-qm", "fixture"],
    ]:
        subprocess.run(["git", "-C", str(source), *args], check=True)
    pins = {
        "commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "execution_code_sha256": "0" * 64,
    }
    (source / "src/model.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="dirty"):
        m.check_source(source, pins)


def test_optional_pinned_cache_identity_is_preserved_without_cache_payload_scan(case):
    cache = {
        "schema": 1,
        "fingerprint": m.CACHE,
        "canonical_fingerprint": m.CONTENT,
        "allow_canonical_fallback": False,
    }
    case["protocol"]["data"]["derived_cache"] = cache
    for name in ["best.pt", "last.pt"]:
        path = case["run"] / name
        payload = torch.load(path, weights_only=True)
        payload["derived_cache"] = cache
        torch.save(payload, path)
    path = case["run"] / "run.json"
    metadata = json.loads(path.read_text())
    metadata["derived_cache"] = cache
    path.write_bytes(m.canonical(metadata))
    rebind(case)
    assert run(case)["derived_cache"] == cache


def test_protocol_configuration_must_include_actual_source_defaults_explicitly(case):
    case["protocol"]["config"].pop("vessel_loss_weight")
    case["terminal"]["config_sha256"] = m.sha(m.canonical(case["protocol"]["config"]))
    rebind(case)
    with pytest.raises(ValueError, match="materialize"):
        run(case)


def test_cache_fallback_is_not_allowed_in_reviewed_protocol(case):
    case["protocol"]["data"]["derived_cache"] = {
        "schema": 1,
        "fingerprint": m.CACHE,
        "canonical_fingerprint": m.CONTENT,
        "allow_canonical_fallback": True,
    }
    rebind(case)
    with pytest.raises(ValueError, match="pinned V2 cache"):
        run(case)


def test_run_completion_can_pass_with_partial_selected_best_but_must_disclose_it(case):
    path = case["run"] / "best.pt"
    best = torch.load(path, weights_only=True)
    best["coverage"]["annotation_visits"].pop("r0")
    best["coverage"]["contributing_annotation_records"] = 5699
    torch.save(best, path)
    rebind(case)
    result = run(case)
    assert result["terminal"] is True and result["all_eligible_annotations_contributed"] is True
    assert result["annotation_coverage_scope"] == "terminal_last_checkpoint_run_completion"
    assert result["selected_best_full_annotations"] is False
    assert result["best_checkpoint_coverage"]["epoch"] == 56
    assert result["best_checkpoint_coverage"]["contributing_annotation_records"] == 5699
    assert result["best_checkpoint_coverage"]["missing_annotation_ids"] == ["r0"]
    assert result["last_checkpoint_coverage"]["epoch"] == 60
    assert result["last_checkpoint_coverage"]["contributing_annotation_records"] == 5700
    assert result["last_checkpoint_coverage"]["all_eligible_annotations_contributed"] is True
