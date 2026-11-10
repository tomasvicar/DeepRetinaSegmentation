"""Full-data search guards and unchanged scheduling across screen/resume stages."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/unified_hpo.py"
    spec = importlib.util.spec_from_file_location("unified_hpo", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def inventory(root, *, preview=False, omit_source=None):
    helper = module()
    rows = [
        {"dataset": name, "photo_id": f"{name}:1", "split": "train"}
        for name in sorted(helper.REQUIRED_SOURCES)
        if name != omit_source
    ]
    path = root / "manifest.json"
    path.write_text(json.dumps(rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "_SUCCESS").write_text(digest)
    audit = {
        "manifest_sha256": digest,
        "prepared_content_sha256": "synthetic-metadata-only",
        "preview_only": preview,
        "annotations_checked": len(rows),
        "photos_checked": len(rows),
        "head_counts": [
            {"split": split, "channel": channel, "valid_pixels": 10, "positive_photos": 1}
            for split in ("train", "val")
            for channel in ("vessel", "artery", "vein", "disc", "cup")
        ],
    }
    (root / "integrity.json").write_text(json.dumps(audit))
    return path


def test_full_search_rejects_previews_and_missing_source(tmp_path):
    helper = module()
    manifest = inventory(tmp_path, preview=True)
    with pytest.raises(ValueError, match="not a preview"):
        helper.make_plan(manifest)
    inventory(tmp_path, omit_source="rite")
    with pytest.raises(ValueError, match="missing training sources"):
        helper.make_plan(manifest)


def test_full_search_requires_real_support_for_every_validation_head(tmp_path):
    helper = module()
    manifest = inventory(tmp_path)
    audit_path = tmp_path / "integrity.json"
    audit = json.loads(audit_path.read_text())
    audit["head_counts"] = [r for r in audit["head_counts"] if r["channel"] != "cup"]
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="supervision for cup"):
        helper.make_plan(manifest)


def test_search_keeps_full_scheduler_horizon_and_refuses_source_changes(tmp_path, monkeypatch):
    helper = module()
    manifest = inventory(tmp_path)
    plan = helper.make_plan(manifest, screen_epochs=8, total_epochs=60, epoch_samples=1)
    assert all(c["epochs"] == 60 and c["epoch_samples"] == 17 for c in plan["candidates"].values())
    assert all(
        c["selection_metric"] == "macro_dataset_mean_dice" for c in plan["candidates"].values()
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    calls = []

    def train(manifest, output, config, **kwargs):
        calls.append((config.epochs, kwargs["stop_after_epoch"], kwargs["resume"]))
        return output / "best.pt"

    monkeypatch.setattr(helper, "train_unified", train)
    output = tmp_path / "trial"
    helper.run_trial(plan_path, "baseline", output, device="cpu")
    helper.run_trial(plan_path, "baseline", output, device="cpu", resume=True, final=True)
    assert calls == [(60, 8, None), (60, 60, output / "last.pt")]
    monkeypatch.setattr(helper, "code_digest", lambda root: "changed")
    with pytest.raises(ValueError, match="Execution code changed"):
        helper.run_trial(plan_path, "baseline", output, device="cpu", resume=True)


def test_trial_provenance_survives_training_failure(tmp_path, monkeypatch):
    helper = module()
    manifest = inventory(tmp_path)
    plan = helper.make_plan(manifest)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated worker interruption")

    monkeypatch.setattr(helper, "train_unified", interrupted)
    with pytest.raises(RuntimeError, match="worker interruption"):
        helper.run_trial(plan_path, "baseline", tmp_path / "trial")
    proof = json.loads((tmp_path / "trial.search_trial.json").read_text())
    assert proof["manifest_sha256"] == plan["manifest_sha256"]
    assert proof["trial"] == "baseline"


def test_independent_search_never_accepts_exposure_qualified_teacher(tmp_path):
    helper = module()
    manifest = inventory(tmp_path)
    plan = helper.make_plan(manifest)
    plan["candidates"]["baseline"].update(
        architecture="legacy_universal",
        legacy_initialization="historical-teacher.pt",
        allow_historical_exposure=True,
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="excludes historical teacher"):
        helper.run_trial(path, "baseline", tmp_path / "trial")
    assert not (tmp_path / "trial").exists()
