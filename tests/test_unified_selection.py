"""Selection metadata checks use CPU toy snapshots, never any network forward."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina import selection as s
from deepretina.integrity import prepared_content_fingerprint
from deepretina.unified import UnifiedConfig, _training_source_sha256, build_unified


@pytest.fixture(autouse=True)
def no_forward_or_cuda(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No model forward or CUDA initialization in selection checks")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    rows, photos = [], []
    av = np.zeros((3, 2, 4), np.float32)
    av[0] = 1
    av[1].flat[:4] = 1
    av[2].flat[4:] = 1
    dc = np.zeros((2, 2, 4), np.float32)
    dc[0].flat[:3] = 1
    dc[1].flat[:1] = 1
    for photo, split in [("t0", "train"), ("t1", "train"), ("v0", "val"), ("x0", "test")]:
        image, fov = f"{photo}.png", f"{photo}-fov.png"
        Image.fromarray(np.full((2, 4, 3), 128, np.uint8)).save(root / image)
        Image.fromarray(np.full((2, 4), 255, np.uint8)).save(root / fov)
        photos.append({"image": image, "fov": fov})
        for task, target in [("av", av), ("disc_cup", dc)]:
            mask = f"{photo}-{task}.npz"
            np.savez(
                root / mask,
                target=target,
                valid=np.ones_like(target, bool),
                fov=np.ones((2, 4), bool),
            )
            rows.append(
                {
                    "id": f"{photo}-{task}",
                    "photo_id": photo,
                    "source_id": f"subject-{photo}",
                    "dataset": "synthetic",
                    "split": split,
                    "task": task,
                    "image": image,
                    "mask": mask,
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25,
                    "preparation_max_size": None,
                }
            )
    manifest = root / "manifest.json"
    manifest.write_bytes(s.canonical(rows))
    (root / "photos.json").write_bytes(s.canonical(photos))
    manifest_sha = s.digest(manifest.read_bytes())
    (root / "_SUCCESS").write_text(manifest_sha)
    content = prepared_content_fingerprint(manifest)
    (root / "integrity.json").write_bytes(
        s.canonical(
            {
                "fingerprint_schema": 2,
                "manifest_sha256": manifest_sha,
                "prepared_content_sha256": content,
                "preview_only": False,
            }
        )
    )
    config = UnifiedConfig(
        base_channels=2,
        depth=1,
        norm="group",
        epochs=60,
        epoch_samples=2,
        batch_size=2,
        workers=0,
        patch_size=32,
        tile_size=32,
        overlap=8,
        device="cpu",
        purpose="full_data",
        selection_metric="macro_dataset_mean_dice",
        augmentation="basic",
    )
    model = build_unified(config)  # CPU construction only; _call_impl is forbidden.
    coverage = {
        "eligible_photos": 2,
        "seen_photos": 2,
        "eligible_annotation_records": 4,
        "contributing_annotation_records": 4,
        "photo_visits": {"t0": 60, "t1": 60},
        "annotation_visits": {
            f"{pid}-{task}": 60 for pid in ("t0", "t1") for task in ("av", "disc_cup")
        },
        "by_dataset": {"synthetic": {"eligible_photos": 2, "seen_photos": 2}},
        "counting": "actual supervised samples delivered to successful optimizer updates",
    }
    scores = [0.8 if epoch == 5 else 0.1 + epoch / 1000 for epoch in range(1, 61)]
    common = {
        "format_version": 1,
        "family": "unified",
        "lineage": model.lineage,
        "architecture": model.architecture_spec,
        "channels": list(s.CHANNELS),
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "manifest_sha256": manifest_sha,
        "prepared_content_sha256": content,
        "training_source_sha256": _training_source_sha256(),
        "supervised_channels": [True] * 5,
        "inference_max_size": None,
        "normalization": "RGB/255-0.5",
        "purpose": "full_data",
        "probability_fusion": "joint_av_mixture_then_conditional",
        "coverage": coverage,
    }
    best = dict(
        common,
        epoch=5,
        steps=5,
        val_selection_score=scores[4],
        validation_metrics={
            "family": "unified",
            "split": "val",
            "selection_metric": config.selection_metric,
            "selection_score": scores[4],
            config.selection_metric: scores[4],
        },
    )
    last = dict(
        common,
        epoch=60,
        steps=60,
        val_selection_score=scores[-1],
        validation_metrics={
            "family": "unified",
            "split": "val",
            "selection_metric": config.selection_metric,
            "selection_score": scores[-1],
            config.selection_metric: scores[-1],
        },
        training_state={
            "coverage": coverage,
            "optimizer": {
                "state": {
                    0: {
                        "step": torch.tensor(60.0),
                        "exp_avg": torch.zeros(1),
                        "exp_avg_sq": torch.zeros(1),
                    }
                },
                "param_groups": [{"lr": 0.00005}],
            },
            "scheduler": {"T_max": 60, "last_epoch": 60},
            "best_validation_score": 0.8,
            "selection_metric": config.selection_metric,
        },
    )
    best_path, last_path = tmp_path / "best.pt", tmp_path / "last.pt"
    torch.save(best, best_path)
    torch.save(last, last_path)
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open("w") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "epoch",
                "steps",
                "lr",
                "train_loss",
                "val_mean_dice",
                "val_macro_dataset_mean_dice",
                "val_selection_score",
                "train_seconds",
                "validation_seconds",
            ],
        )
        writer.writeheader()
        for epoch, score in enumerate(scores, 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "steps": epoch,
                    "lr": config.lr * config.min_lr_ratio
                    + (config.lr - config.lr * config.min_lr_ratio)
                    * (1 + math.cos(math.pi * (epoch - 1) / 60))
                    / 2,
                    "train_loss": 1.0,
                    "val_mean_dice": score,
                    "val_macro_dataset_mean_dice": score,
                    "val_selection_score": score,
                    "train_seconds": 0.1,
                    "validation_seconds": 0.1,
                }
            )
    summary = tmp_path / "training_summary.json"
    summary.write_bytes(
        s.canonical(
            {
                "completed_budget": True,
                "steps": 60,
                "best_validation_score": 0.8,
                "selection_metric": config.selection_metric,
            }
        )
    )
    protocol = s.inference_protocol(best)
    data, groups = s.data_identity(manifest, best)
    counts = [[6, 0, 2, 8], [3, 0, 1, 8], [2, 1, 2, 8], [2, 1, 1, 8], [1, 0, 0, 8]]
    channels = {
        name: s._metrics(np.array(value)) for name, value in zip(s.CHANNELS, counts, strict=True)
    }
    macro = {name: value["dice"] for name, value in channels.items()}
    score = float(np.mean(list(macro.values())))
    metrics = {
        "family": "unified",
        "split": "val",
        "purpose": "full_data",
        "lineage": model.lineage,
        "supervised_channels": [True] * 5,
        "threshold": 0.5,
        "metric_support": s.SUPPORT,
        "evaluation_policy": s.POLICY,
        "photos": 1,
        "records": 2,
        "channels": channels,
        "by_dataset": {"synthetic": channels},
        "macro_dataset_channel_dice": macro,
        "macro_dataset_mean_dice": score,
        "mean_dice": score,
        "av_labeled_pixels": 8,
        "av_accuracy": 5 / 8,
        "av_conditional_accuracy": 7 / 8,
        "selection_metric": config.selection_metric,
        "selection_score": score,
        "per_photo": [
            {
                "photo_id": "v0",
                "dataset": "synthetic",
                "record_ids": [r["id"] for r in groups["val"]["v0"]],
                "channel_datasets": {name: "synthetic" for name in s.CHANNELS},
                "channels": channels,
                "av_conditional_counts": {
                    "correct": 7,
                    "scored_pixels": 8,
                    "eligible_pixels": 8,
                    "end_to_end_correct": 5,
                },
            }
        ],
    }
    evidence = {
        "schema_version": 1,
        "kind": "unified_validation_evidence",
        "checkpoint": {
            "sha256": s.digest(best_path.read_bytes()),
            "epoch": 5,
            "steps": 5,
            "training_selection_score": 0.8,
        },
        "configuration": s.config_identity(best),
        "source": {
            **s.evaluation_source_identity(),
            "training_source_sha256": best["training_source_sha256"],
        },
        "data": data,
        "inference_protocol": protocol,
        "evaluated_at_utc": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        "metrics": metrics,
    }
    evidence_path = tmp_path / "enriched-val.json"
    evidence_path.write_bytes(s.canonical(evidence))
    return {
        "checkpoint": best_path,
        "last_checkpoint": last_path,
        "manifest": manifest,
        "validation_evidence": evidence_path,
        "metrics_csv": metrics_path,
        "training_summary": summary,
    }


def create(case, tmp_path):
    path = tmp_path / "selection.json"
    receipt = s.create_selection(
        output=path, reviewer="reviewer", rationale="Frozen validation choice", **case
    )
    return path, receipt["selection_sha256"]


def check(case, path, expected, **overrides):
    return s.check_selection(
        selection=path, expected_selection_sha256=expected, **case, **overrides
    )


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_bytes(s.canonical(value))


def mutate_checkpoint(path, change):
    value = torch.load(path, map_location="cpu", weights_only=True)
    change(value)
    torch.save(value, path)


def test_valid_metadata_never_decodes_test_gt_or_forwards(case, tmp_path, monkeypatch):
    loaded, original = [], s.np.load

    def observe(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(s.np, "load", observe)
    path, expected = create(case, tmp_path)
    checked = check(case, path, expected)
    assert checked.test_evaluation_enabled is False
    assert checked.checkpoint_sha256 == hashlib.sha256(case["checkpoint"].read_bytes()).hexdigest()
    assert loaded and all(name.startswith("v0-") for name in loaded)
    assert checked.protocol["threshold"] == 0.5
    altered_copy = checked.protocol
    altered_copy["threshold"] = 0.1
    assert checked.protocol["threshold"] == 0.5


def test_snapshot_retained_even_when_same_checkpoint_path_is_replaced(case, tmp_path):
    path, expected = create(case, tmp_path)
    checked = check(case, path, expected)
    original = checked.checkpoint_snapshot
    case["checkpoint"].write_bytes(case["last_checkpoint"].read_bytes())
    assert checked.checkpoint_snapshot == original
    assert torch.load(io.BytesIO(original), weights_only=True)["epoch"] == 5
    with pytest.raises(s.SelectionError, match="checkpoint/configuration"):
        check(case, path, expected)


@pytest.mark.parametrize("kwargs", [{"threshold": 0.4}, {"tile_size": 48}, {"overlap": 2}])
def test_requested_protocol_override_cannot_reuse_selection(case, tmp_path, kwargs):
    path, expected = create(case, tmp_path)
    with pytest.raises(s.SelectionError, match="protocol"):
        check(case, path, expected, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["checkpoint"].update(sha256="f" * 64),
        lambda r: r["configuration"]["values"].update(lr=0.3),
        lambda r: r["source"].update(evaluation_source_sha256="f" * 64),
        lambda r: r["data"].update(prepared_content_sha256="f" * 64),
        lambda r: r["metrics"]["per_photo"][0].update(photo_id="x0"),
        lambda r: r["metrics"]["per_photo"][0]["channel_datasets"].update(disc="other"),
        lambda r: r["metrics"]["channels"]["disc"].update(tp=999),
        lambda r: r.update(evaluated_at_utc=(datetime.now(UTC) + timedelta(days=1)).isoformat()),
    ],
)
def test_mismatched_enriched_evidence_rejected(case, tmp_path, change):
    mutate_json(case["validation_evidence"], change)
    with pytest.raises((s.SelectionError, ValueError)):
        create(case, tmp_path)
    assert not (tmp_path / "selection.json").exists()


def test_arbitrary_filename_hash_or_old_thin_val_report_is_not_evidence(case, tmp_path):
    case["validation_evidence"].write_text('{"split":"val","score":1}')
    with pytest.raises(s.SelectionError, match="enriched validation evidence"):
        create(case, tmp_path)
    (tmp_path / "selection.json").write_text('{"approved":true}')
    expected = hashlib.sha256((tmp_path / "selection.json").read_bytes()).hexdigest()
    with pytest.raises(s.SelectionError, match="selection artifact"):
        check(case, tmp_path / "selection.json", expected)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(epoch=59),
        lambda r: r["training_state"]["scheduler"].update(T_max=8),
        lambda r: r["training_state"]["optimizer"]["state"][0].update(
            exp_avg=torch.tensor([float("nan")])
        ),
        lambda r: r["coverage"]["photo_visits"].pop("t1"),
    ],
)
def test_incomplete_or_corrupt_training_budget_rejected(case, tmp_path, change):
    mutate_checkpoint(case["last_checkpoint"], change)
    with pytest.raises(s.SelectionError):
        create(case, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["config"].update(purpose="functionality"),
        lambda r: r["lineage"].update(historical_exposure="unknown"),
        lambda r: r["supervised_channels"].__setitem__(4, False),
        lambda r: r["config"].update(allow_historical_exposure=True),
        lambda r: r["state_dict"][next(iter(r["state_dict"]))].fill_(float("nan")),
    ],
)
def test_pilot_teacher_unsupported_nonfinite_checkpoints_rejected(case, tmp_path, change):
    mutate_checkpoint(case["checkpoint"], change)
    with pytest.raises((s.SelectionError, ValueError)):
        create(case, tmp_path)


def test_changed_actual_canonical_bytes_fail_even_with_same_path(case, tmp_path):
    path, expected = create(case, tmp_path)
    image = case["manifest"].parent / "v0.png"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="content changed"):
        check(case, path, expected)


def test_decision_is_required_pinned_and_not_backdated_before_val(case, tmp_path):
    path, _expected = create(case, tmp_path)
    with pytest.raises(s.SelectionError, match="review digest"):
        check(case, path, "f" * 64)
    mutate_json(path, lambda r: r["decision"].update(recorded_at_utc="2000-01-01T00:00:00+00:00"))
    updated = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(s.SelectionError, match="postdates"):
        check(case, path, updated)


def test_actual_val_gt_support_is_checked_not_only_report_headers(case, tmp_path):
    evidence = json.loads(case["validation_evidence"].read_text())
    row = evidence["metrics"]["per_photo"][0]
    row["channels"]["disc"] = s._metrics(np.array([1, 1, 1, 8]))
    case["validation_evidence"].write_bytes(s.canonical(evidence))
    with pytest.raises(s.SelectionError, match="actual canonical annotations"):
        create(case, tmp_path)


def test_public_test_route_stays_blocked_even_after_valid_selection(case, tmp_path, capsys):
    from deepretina.cli import main

    create(case, tmp_path)
    report = tmp_path / "test-metrics.json"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--checkpoint",
                str(case["checkpoint"]),
                "--manifest",
                str(case["manifest"]),
                "--split",
                "test",
                "--device",
                "cpu",
                "--output",
                str(report),
            ]
        )
    assert error.value.code == 2
    assert "val only" in capsys.readouterr().err
    assert not report.exists()


def test_gate_is_not_removed_by_python_optimization(case, tmp_path):
    path = tmp_path / "arbitrary.json"
    path.write_text('{"approval":true}')
    script = Path(__file__).parents[1] / "scripts/unified_selection.py"
    args = [
        sys.executable,
        "-O",
        str(script),
        "check",
        "--selection",
        str(path),
        "--expected-selection-sha256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ]
    for key, value in case.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parents[1] / "src"),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
    )
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "Invalid selection artifact fields" in result.stderr


def test_optimizer_steps_must_agree_with_completed_run(case, tmp_path):
    mutate_checkpoint(
        case["last_checkpoint"],
        lambda r: r["training_state"]["optimizer"]["state"][0].update(step=torch.tensor(59.0)),
    )
    with pytest.raises(s.SelectionError, match="Optimizer update count"):
        create(case, tmp_path)


def test_lr_history_must_follow_frozen_config(case, tmp_path):
    rows = list(csv.DictReader(case["metrics_csv"].open()))
    rows[10]["lr"] = "0.1"
    with case["metrics_csv"].open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(s.SelectionError, match="LR history"):
        create(case, tmp_path)


def test_missing_epoch_does_not_count_as_full_budget(case, tmp_path):
    lines = case["metrics_csv"].read_text().splitlines()
    case["metrics_csv"].write_text("\n".join(lines[:20] + lines[21:]) + "\n")
    with pytest.raises(s.SelectionError, match="60 unique epochs"):
        create(case, tmp_path)


def test_incomplete_summary_rejected(case, tmp_path):
    mutate_json(case["training_summary"], lambda r: r.update(completed_budget=False))
    with pytest.raises(s.SelectionError, match="completion summary"):
        create(case, tmp_path)


def test_active_autocast_is_not_the_recorded_float32_protocol(case, tmp_path):
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(s.SelectionError, match="autocast"),
    ):
        create(case, tmp_path)


def test_duplicate_json_fields_and_nonfinite_json_are_not_accepted():
    with pytest.raises(s.SelectionError, match="Duplicate"):
        s.parse_json(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(s.SelectionError, match="Nonfinite"):
        s.parse_json(b'{"value":NaN}')


def test_artifact_cannot_claim_it_enables_test_execution(case, tmp_path):
    path, _expected = create(case, tmp_path)
    mutate_json(path, lambda r: r["boundaries"].update(test_evaluation_enabled=True))
    updated = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(s.SelectionError, match="stage1 boundary"):
        check(case, path, updated)
