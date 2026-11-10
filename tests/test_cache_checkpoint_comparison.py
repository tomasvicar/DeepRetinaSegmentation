"""Tiny CPU-only synthetic checkpoint comparison fixtures; no training/GPU/data."""

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports/unified/cache-width16/compare_checkpoint_states.py"
)
spec = importlib.util.spec_from_file_location("cache_comparison", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    def payload():
        coverage = {
            "photo_visits": {"sample": 3},
            "by_dataset": {"synthetic": {"seen_photos": 1}},
        }
        counts = {"tp": 2, "fp": 1, "fn": 1, "valid_pixels": 5}
        validation = {
            "mean_dice": 0.5,
            "av_labeled_pixels": 2,
            "channels": {"vessel": counts},
            "per_photo": [
                {
                    "photo_id": "sample",
                    "channels": {"vessel": dict(counts)},
                    "dataset": "synthetic",
                }
            ],
        }
        return {
            "family": "unified",
            "config": {"device": "cuda", "seed": 42},
            "state_dict": {
                "weight": torch.tensor([1.0, 2.0]),
                "running_mean": torch.tensor([0.1]),
                "num_batches_tracked": torch.tensor(4),
            },
            "training_state": {
                "optimizer": {
                    "state": {
                        0: {
                            "step": torch.tensor(1.0),
                            "exp_avg": torch.tensor([0.01, 0.02]),
                        }
                    }
                },
                "scheduler": {"last_epoch": 1},
                "rng": {
                    "torch": torch.tensor([2, 3], dtype=torch.uint8),
                    "python": (3, (1, 2), None),
                },
                "coverage": coverage,
            },
            "coverage": coverage,
            "epoch": 1,
            "steps": 2,
            "manifest_sha256": "manifest",
            "prepared_content_sha256": "data",
            "training_source_sha256": "old-source",
            "validation_metrics": validation,
        }

    old, new = payload(), payload()
    new["config"]["device"] = "cpu"
    new["training_source_sha256"] = "new-source"
    new["derived_cache"] = {
        "fingerprint": "cache",
        "canonical_fingerprint": "data",
        "allow_canonical_fallback": False,
    }
    new["validation_metrics"]["per_photo"][0]["av_conditional_counts"] = {
        "correct": 2,
        "scored_pixels": 2,
    }
    paths = [
        tmp_path / "old.pt",
        tmp_path / "new.pt",
        tmp_path / "old.csv",
        tmp_path / "new.csv",
    ]
    torch.save(old, paths[0])
    torch.save(new, paths[1])
    for path, seconds in zip(paths[2:], [4.0, 2.0], strict=True):
        with path.open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "steps",
                    "train_loss",
                    "train_seconds",
                    "validation_seconds",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": 1,
                    "steps": 2,
                    "train_loss": 0.4,
                    "train_seconds": seconds,
                    "validation_seconds": 1.0,
                }
            )
    return paths


def test_exact_states_and_expected_provenance_differences(tmp_path):
    paths = fixture(tmp_path)
    report = module.build_report(*paths)
    assert report["preservation_equal"] and report["strict_tensor_equal"]
    assert report["sections"]["state_dict"]["tensors"] == 3
    assert report["sections"]["training_state"]["tensors"] == 3
    assert report["provenance"]["derived_cache"]["old_present"] is False
    assert report["provenance"]["training_source_sha256"]["new"] == "new-source"
    assert report["timing"]["train_speedup"] == 2
    json.dumps(report, allow_nan=False)


def test_changed_buffers_optimizer_rng_missing_keys_and_nan_are_visible(tmp_path):
    paths = fixture(tmp_path)
    new = torch.load(paths[1], weights_only=True)
    new["state_dict"]["running_mean"] += 0.25
    del new["state_dict"]["num_batches_tracked"]
    new["training_state"]["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    new["training_state"]["rng"]["torch"][0] = 4
    torch.save(new, paths[1])
    report = module.build_report(*paths)
    assert not report["preservation_equal"] and not report["strict_tensor_equal"]
    differences = report["sections"]["state_dict"]["differences"]
    assert any("num_batches_tracked" in d["path"] for d in differences)
    assert next(d for d in differences if "running_mean" in d["path"])[
        "max_abs_finite"
    ] == pytest.approx(0.25)
    assert report["sections"]["training_state"]["nonfinite"]
    json.dumps(report, allow_nan=False)


def test_validation_photo_sets_counts_and_csv_loss_are_not_hidden(tmp_path):
    paths = fixture(tmp_path)
    new = torch.load(paths[1], weights_only=True)
    new["validation_metrics"]["per_photo"][0]["channels"]["vessel"]["tp"] = 0
    torch.save(new, paths[1])
    paths[3].write_text(paths[3].read_text().replace("0.4", "0.5"))
    report = module.build_report(*paths)
    assert not report["sections"]["validation"]["equal"]
    assert any(
        "per_photo_channel_counts" in item["path"]
        for item in report["sections"]["validation"]["differences"]
    )
    assert not report["sections"]["csv_loss_steps"]["equal"]
    assert not report["preservation_equal"]


def test_snapshot_hash_and_weights_read_same_bytes_even_path_replaced(tmp_path, monkeypatch):
    paths = fixture(tmp_path)
    data = paths[0].read_bytes()
    actual = torch.load

    def replacement(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO) and kwargs["weights_only"] is True
        paths[0].write_bytes(paths[1].read_bytes())
        return actual(source, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", replacement)
    payload, identity = module.snapshot(paths[0])
    assert identity["sha256"] == module.hashlib.sha256(data).hexdigest()
    assert payload["training_source_sha256"] == "old-source"


def test_exclusive_output_preserves_existing_before_reading_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("keep")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing report must fail before checkpoint IO")

    monkeypatch.setattr(module, "snapshot", forbidden)
    with pytest.raises(FileExistsError):
        module.main(
            [
                "--old",
                "old",
                "--new",
                "new",
                "--old-csv",
                "old.csv",
                "--new-csv",
                "new.csv",
                "--output",
                str(path),
            ]
        )
    assert path.read_text() == "keep"


def test_nonfinite_and_both_missing_do_not_pass(tmp_path):
    same_inf = module.compare(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert same_inf["strict_tensor_equal"] and not same_inf["equal"] and same_inf["nonfinite"]
    paths = fixture(tmp_path)
    for path in paths[:2]:
        payload = torch.load(path, weights_only=True)
        del payload["training_state"]["scheduler"]
        torch.save(payload, path)
    report = module.build_report(*paths)
    assert not report["preservation_equal"]
    assert any("scheduler" in issue for issue in report["issues"])
    assert not module.compare(module.MISSING, module.MISSING)["equal"]
