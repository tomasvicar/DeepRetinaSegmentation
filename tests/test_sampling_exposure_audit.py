"""Analytic small-corpus checks for pre-crop expected exposure, never ML training."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "sampling_exposure_audit", Path(__file__).parents[1] / "scripts/audit_sampling_exposure.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_aliased_vascular_choice_and_disc_only_addition_change_expected_exposure():
    rows = [
        {"id": "drive1", "dataset": "drive", "photo_id": "same", "task": "vessels"},
        {"id": "rite1", "dataset": "rite", "photo_id": "same", "task": "av"},
        {"id": "refuge1", "dataset": "refuge", "photo_id": "disc", "task": "disc_cup"},
    ]
    flags = {"drive1": {"vessel"}, "rite1": {"vessel", "av"}, "refuge1": {"disc", "cup"}}
    old = audit.exposure(rows, flags, flags, 8192)
    assert old["photos"] == 2 and old["coverage_first_draws"] == 2
    assert old["tasks"]["vessel"]["expected_selected_supervision_total"] == 4096
    assert old["tasks"]["av"]["expected_selected_supervision_total"] == 2048
    assert old["tasks"]["av"]["expected_photo_opportunities"] == 4096
    assert old["tasks"]["cup"]["expected_selected_supervision_total"] == 4096
    assert {r["sampling_owner"] for r in old["sampling_sources"]} == {"drive", "refuge"}
    assert old["expected_selected_annotation_bundles_by_source"]["rite"] == 2048
    rows.append({"id": "drions1", "dataset": "drions", "photo_id": "new", "task": "disc_cup"})
    flags["drions1"] = {"disc"}
    new = audit.exposure(rows, flags, flags, 8192)
    assert new["weighted_extra_draws"] == 8189
    assert new["tasks"]["vessel"]["expected_selected_supervision_total"] == pytest.approx(3276.6)
    assert new["tasks"]["av"]["expected_selected_supervision_total"] == pytest.approx(1638.3)
    assert new["tasks"]["disc"]["expected_selected_supervision_total"] == pytest.approx(4915.4)
    assert sum(r["expected_photo_draws"] for r in new["sampling_sources"]) == pytest.approx(8192)


def test_unknown_av_and_absent_cup_do_not_create_sampling_tasks():
    target = np.zeros((3, 2, 2), np.float32)
    target[0, 0, 0] = 1
    valid = np.ones_like(target, bool)
    fov = np.ones((2, 2), bool)
    row = {"id": "ambiguous", "task": "av"}
    tasks, positive = audit.task_flags(row, target, valid, fov)
    assert tasks == positive == {"vessel"}
    target[1, 0, 0] = 1
    tasks, positive = audit.task_flags(row, target, valid, fov)
    assert tasks == positive == {"vessel", "av"}
    valid[2] = False
    assert audit.task_flags(row, target, valid, fov)[0] == {"vessel"}
    disc = np.ones((2, 2, 2), bool)
    support = np.ones_like(disc)
    support[1] = False
    assert audit.task_flags({"id": "disc", "task": "disc_cup"}, disc, support, fov)[0] == {"disc"}


def test_scenarios_separate_vascular_weight_from_av_bundle_choice():
    rows = [
        {"id": "a", "dataset": "drive", "photo_id": "same", "task": "vessels"},
        {"id": "b", "dataset": "rite", "photo_id": "same", "task": "av"},
        {"id": "c", "dataset": "refuge", "photo_id": "disc", "task": "disc_cup"},
    ]
    flags = {"a": {"vessel"}, "b": {"vessel", "av"}, "c": {"disc", "cup"}}
    baseline = audit.exposure(rows, flags, flags, 8192)
    explicit = audit.exposure(rows, flags, flags, 8192, vascular_priority=1, av_probability=0.5)
    assert baseline == explicit
    more_av = audit.exposure(rows, flags, flags, 8192, vascular_priority=1, av_probability=0.75)
    assert more_av["tasks"]["av"]["expected_selected_supervision_total"] == 3072
    for task in ("vessel", "disc", "cup"):
        assert more_av["tasks"][task] == baseline["tasks"][task]
    more_vascular = audit.exposure(
        rows, flags, flags, 8192, vascular_priority=2, av_probability=0.5
    )
    assert more_vascular["tasks"]["vessel"]["expected_selected_supervision_total"] == 5461
    assert more_vascular["tasks"]["av"]["expected_selected_supervision_total"] == 2730.5
    assert more_vascular["tasks"]["disc"]["expected_selected_supervision_total"] == 2731
    with pytest.raises(ValueError, match="predeclared"):
        audit.exposure(rows, flags, flags, 8192, vascular_priority=3)


def test_task_flag_export_is_train_only_pinned_metadata_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    import hashlib
    import io
    import json
    import sys

    row = {
        "id": "train",
        "dataset": "drive",
        "photo_id": "p",
        "task": "vessels",
        "split": "train",
        "mask": "train.npz",
    }
    heldout = dict(row, id="heldout", photo_id="other", split="test", mask="never-open.npz")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([row, heldout]))
    cache = tmp_path / "cache"
    cache.mkdir()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        target=np.ones((1, 2, 2), bool),
        valid=np.ones((1, 2, 2), bool),
        fov=np.ones((2, 2), bool),
    )
    raw = buffer.getvalue()
    (cache / "label.npz").write_bytes(raw)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    index = {
        "canonical_manifest_sha256": digest(manifest),
        "canonical_fingerprint": "fixture",
        "construction": {"exact_equality_verified": True},
        "source_to_derived": {"train.npz": {"derived": "label.npz"}},
        "files": {"label.npz": {"bytes": len(raw), "sha256": digest(cache / "label.npz")}},
    }
    (cache / "index.json").write_text(json.dumps(index))
    source = tmp_path / "frozen.py"
    source.write_text(
        "class UnifiedDataset:\n    def sampling_weights(self):\n        return [1.0]\n"
    )
    output, flags = tmp_path / "report.json", tmp_path / "flags.json"
    args = [
        "audit",
        "--manifest",
        str(manifest),
        "--parent-manifest",
        str(manifest),
        "--cache-root",
        str(cache),
        "--frozen-unified",
        str(source),
        "--output",
        str(output),
        "--task-flags-output",
        str(flags),
        "--expected-manifest-sha256",
        digest(manifest),
        "--expected-cache-index-sha256",
        digest(cache / "index.json"),
        "--expected-frozen-unified-sha256",
        digest(source),
    ]
    monkeypatch.setattr(sys, "argv", args)
    audit.main()
    metadata = json.loads(flags.read_text())
    assert metadata["record_tasks"] == {"train": ["vessel"]}
    assert metadata["manifest_sha256"] == digest(manifest)
    report = json.loads(output.read_text())
    assert report["train_task_flags_sha256"] == digest(flags)
    assert report["heldout_masks_decoded"] == 0
    fresh = tmp_path / "new-report.json"
    args[args.index("--output") + 1] = str(fresh)
    with pytest.raises(FileExistsError):
        audit.main()
    assert not fresh.exists()
