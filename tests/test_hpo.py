"""Validate search decisions without reading held-out test data or training models."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "hpo_under_test", Path(__file__).parents[1] / "scripts/hpo.py"
)
hpo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hpo
SPEC.loader.exec_module(hpo)


def write_run(path, manifest, config, score):
    from dataclasses import asdict

    path.mkdir(parents=True)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    settings = asdict(config)
    (path / "run.json").write_text(json.dumps({"config": settings, "manifest_sha256": digest}))
    (path / "validation.json").write_text(json.dumps({"split": "val", "mean_dice": score}))
    with (path / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_mean_dice"])
        writer.writeheader()
        writer.writerow({"epoch": config.epochs, "val_mean_dice": score})
    torch.save(
        {
            "config": settings,
            "manifest_sha256": digest,
            "epoch": config.epochs,
            "steps": config.epochs * 2,
            "val_mean_dice": score,
        },
        path / "best.pt",
    )
    return path / "best.pt"


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"task":"disc_cup"}]')
    config = hpo.SearchConfig("disc_cup", device="cpu", cpu_threads=1)
    baseline = tmp_path / "baseline"
    write_run(baseline, manifest, config.trial(0.001, 30), 0.8)
    # Any accidental test read would fail JSON decoding.
    (baseline / "test.json").write_text("{never read heldout data during search")
    calls = []

    def fake_train(manifest, path, trial):
        calls.append((trial.lr, trial.epochs))
        scores = {
            8: {0.0003: 0.95, 0.001: 0.99, 0.003: 0.7},
            30: {0.0003: 0.82, 0.001: 0.8, 0.003: 0.79},
        }
        return write_run(path, manifest, trial, scores[trial.epochs][trial.lr])

    monkeypatch.setattr(hpo, "train", fake_train)
    monkeypatch.setattr(
        hpo, "evaluate", lambda *a, **kw: pytest.fail("Test evaluation during search")
    )
    return manifest, tmp_path / "search", baseline, config, calls


def test_promotes_nonbaseline_candidates_and_preserves_baseline(scenario):
    manifest, output, baseline, config, calls = scenario
    previous = {p.name: p.read_bytes() for p in baseline.iterdir()}
    result = hpo.search(manifest, output, baseline, config)
    assert calls == [(0.0003, 8), (0.001, 8), (0.003, 8), (0.0003, 30), (0.003, 30)]
    assert result["winner"]["lr"] == 0.0003
    assert result["validation_delta_vs_baseline"] == pytest.approx(0.02)
    assert result["test_access_during_search"] is False
    assert result["selection_split"] == "val"
    assert len(result["eligible_finalists"]) == 3
    assert previous == {p.name: p.read_bytes() for p in baseline.iterdir()}
    assert len(list(csv.DictReader((output / "candidates.csv").open()))) == 6
    assert len(list((output / "configs").glob("*.json"))) == 5


def test_baseline_wins_even_if_short_stage_is_bad(scenario, monkeypatch):
    manifest, output, baseline, config, calls = scenario

    def worse(manifest, path, trial):
        calls.append((trial.lr, trial.epochs))
        return write_run(path, manifest, trial, 0 if trial.lr == 0.001 else 0.7)

    monkeypatch.setattr(hpo, "train", worse)
    result = hpo.search(manifest, output, baseline, config)
    assert result["winner"]["phase"] == "baseline"
    assert result["winner"]["reused_baseline"] is True
    assert result["validation_delta_vs_baseline"] == 0


def test_completed_resume_does_not_train_again(scenario):
    manifest, output, baseline, config, calls = scenario
    original = hpo.search(manifest, output, baseline, config)
    calls.clear()
    assert hpo.search(manifest, output, baseline, config, resume=True) == original
    assert calls == []
    with pytest.raises(FileExistsError):
        hpo.search(manifest, output, baseline, config)
    changed = hpo.SearchConfig("disc_cup", device="cpu", cpu_threads=1, lrs=(0.0001, 0.001, 0.003))
    with pytest.raises(ValueError, match="changed configuration"):
        hpo.search(manifest, output, baseline, changed, resume=True)


def test_baseline_manifest_or_configuration_mismatch_fails_before_training(scenario):
    manifest, output, baseline, config, calls = scenario
    info = json.loads((baseline / "run.json").read_text())
    info["config"]["seed"] = 99
    (baseline / "run.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="seed"):
        hpo.search(manifest, output, baseline, config)
    assert calls == [] and not output.exists()
    info["config"]["seed"] = 42
    info["manifest_sha256"] = "wrong"
    (baseline / "run.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Manifest hash"):
        hpo.search(manifest, output, baseline, config)


def test_test_evaluation_requires_frozen_selection_and_can_run_only_once(scenario, monkeypatch):
    manifest, output, baseline, config, _calls = scenario
    with pytest.raises(FileNotFoundError):
        hpo.evaluate_selected_test(manifest, output, "cpu")
    selected = hpo.search(manifest, output, baseline, config)
    frozen = (output / "selection.json").read_bytes()
    evaluations = []

    def evaluate(checkpoint, manifest, **kwargs):
        evaluations.append((checkpoint, kwargs))
        return {"split": kwargs["split"], "mean_dice": 0.75}

    monkeypatch.setattr(hpo, "evaluate", evaluate)
    result = hpo.evaluate_selected_test(manifest, output, "cpu")
    assert evaluations[0][0] == Path(selected["winner"]["checkpoint"])
    assert evaluations[0][1]["split"] == "test"
    assert evaluations[0][1]["threshold"] == 0.5
    assert result["test_previously_reported_in_original_pilot"] is True
    assert frozen == (output / "selection.json").read_bytes()
    with pytest.raises(FileExistsError):
        hpo.evaluate_selected_test(manifest, output, "cpu")
    assert len(evaluations) == 1


@pytest.mark.parametrize(
    "rates", [(0, 0.001), (0.001, 0.001), (float("nan"), 0.001), (float("inf"), 0.001)]
)
def test_invalid_search_rates_fail(rates):
    with pytest.raises(ValueError):
        hpo.SearchConfig("disc_cup", lrs=rates).validate()
