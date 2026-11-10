"""Tiny metadata/tensor fixtures only: no real checkpoints, photos or inference."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "e060_qa", Path(__file__).parents[1] / "scripts/verify_control_e060.py"
)
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


@pytest.fixture
def example():
    config = {
        "epochs": 60,
        "epoch_samples": 8192,
        "batch_size": 4,
        "coverage_first": True,
        "selection_metric": "macro_dataset_mean_dice",
        "lr": 0.0003,
        "min_lr_ratio": 0.05,
        "device": "cuda:0",
    }
    identity = {
        "schema": 1,
        "fingerprint": qa.CACHE,
        "canonical_fingerprint": qa.CONTENT,
        "allow_canonical_fallback": False,
    }
    plan = {
        "candidates": {qa.TRIAL: config},
        "derived_cache": identity,
        "datasets": ["toy_a", "toy_b"],
    }
    rows = [
        {
            "id": f"annotation-{i}",
            "source_id": f"patient-{i}",
            "photo_id": f"photo-{i}",
            "dataset": name,
            "task": "disc_cup",
            "split": "train",
        }
        for i, name in enumerate(plan["datasets"])
    ]
    coverage = {
        "eligible_photos": 2,
        "seen_photos": 2,
        "photo_visits": {"photo-0": 245760, "photo-1": 245760},
        "by_dataset": {name: {"eligible_photos": 1, "seen_photos": 1} for name in plan["datasets"]},
    }

    def payload(epoch, score):
        metrics = {"macro_dataset_mean_dice": score, "selection_score": score}
        return {
            "format_version": 1,
            "family": "unified",
            "config": copy.deepcopy(config),
            "manifest_sha256": qa.MANIFEST,
            "prepared_content_sha256": qa.CONTENT,
            "training_source_sha256": qa.TRAINING,
            "derived_cache": copy.deepcopy(identity),
            "channels": ["vessel", "artery", "vein", "disc", "cup"],
            "normalization": "RGB/255-0.5",
            "supervised_channels": [True] * 5,
            "state_dict": {"weight": torch.tensor([0.1])},
            "validation_metrics": metrics,
            "val_selection_score": score,
            "epoch": epoch,
            "steps": epoch * 2048,
            "coverage": copy.deepcopy(coverage),
        }

    last, best = payload(60, 0.9), payload(58, 0.9)
    last["training_state"] = {
        "best_validation_score": 0.9,
        "selection_metric": config["selection_metric"],
        "coverage": copy.deepcopy(coverage),
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([0.01])}},
            "param_groups": [{"lr": 0.000015}],
        },
        "scheduler": {"T_max": 60, "last_epoch": 60},
        "rng": {
            "torch": torch.tensor([1], dtype=torch.uint8),
            "python": (3, (1, 2), None),
            "numpy": ["MT19937", [1, 2], 0, 0, 0.0],
            "sampler": torch.tensor([1], dtype=torch.uint8),
            "augmentation": None,
            "cuda": torch.tensor([1], dtype=torch.uint8),
        },
    }
    metrics = []
    for e in range(1, 61):
        score = 0.9 if e in [58, 60] else e / 1000
        metrics.append(
            {
                "epoch": str(e),
                "steps": str(e * 2048),
                "lr": str(
                    config["lr"] * (0.05 + 0.95 * (1 + math.cos(math.pi * (e - 1) / 60)) / 2)
                ),
                "train_loss": ".2",
                "val_mean_dice": str(score),
                "val_macro_dataset_mean_dice": str(score),
                "val_selection_score": str(score),
                "train_seconds": "1",
                "validation_seconds": "1",
            }
        )
    summary = {
        "completed_budget": True,
        "steps": 122880,
        "best_validation_score": 0.9,
        "selection_metric": config["selection_metric"],
    }
    run = {
        k: copy.deepcopy(last[k])
        for k in [
            "config",
            "manifest_sha256",
            "prepared_content_sha256",
            "derived_cache",
            "training_source_sha256",
        ]
    }
    provenance = {
        "plan_sha256": qa.PLAN,
        "trial": qa.TRIAL,
        "execution_code_sha256": qa.EXECUTION,
        "manifest_sha256": qa.MANIFEST,
    }
    return {
        "last": last,
        "best": best,
        "metrics": metrics,
        "summary": summary,
        "run": run,
        "coverage_sidecar": copy.deepcopy(coverage),
        "validation_sidecar": copy.deepcopy(best["validation_metrics"]),
        "rows": rows,
        "plan": plan,
        "provenance": provenance,
    }


def test_terminal_first_maximum_and_parent_augmentation_none(example):
    result = qa.validate_terminal(**example)
    assert result["last_epoch"] == 60 and result["best_epoch"] == 58
    assert result["delivered_samples"] == 491520 and result["known_train_source_groups"] == 2
    assert result["best_checkpoint_has_optimizer"] is False


@pytest.mark.parametrize(
    "broken",
    [
        "unfinished",
        "later_tie",
        "csv_gap",
        "wrong_steps",
        "wrong_lr",
        "nonfinite_weight",
        "nonfinite_optimizer",
        "missing_cuda_rng",
        "wrong_scheduler",
        "wrong_cache",
        "wrong_source",
        "low_coverage",
        "source_count",
        "split_leak",
        "sidecar",
        "wrong_plan",
    ],
)
def test_rejects_broken_evidence(example, broken):
    if broken == "unfinished":
        example["summary"]["completed_budget"] = False
    elif broken == "later_tie":
        example["best"]["epoch"] = 60
        example["best"]["steps"] = 122880
    elif broken == "csv_gap":
        example["metrics"][10]["epoch"] = "12"
    elif broken == "wrong_steps":
        example["last"]["steps"] = 122879
    elif broken == "wrong_lr":
        example["metrics"][20]["lr"] = ".001"
    elif broken == "nonfinite_weight":
        example["best"]["state_dict"]["weight"][0] = float("nan")
    elif broken == "nonfinite_optimizer":
        example["last"]["training_state"]["optimizer"]["state"][0]["exp_avg"][0] = float("inf")
    elif broken == "missing_cuda_rng":
        example["last"]["training_state"]["rng"].pop("cuda")
    elif broken == "wrong_scheduler":
        example["last"]["training_state"]["scheduler"]["T_max"] = 8
    elif broken == "wrong_cache":
        example["best"]["derived_cache"]["fingerprint"] = "wrong"
    elif broken == "wrong_source":
        example["last"]["training_source_sha256"] = "87-source"
    elif broken == "low_coverage":
        for item in [
            example["last"]["coverage"],
            example["last"]["training_state"]["coverage"],
            example["coverage_sidecar"],
        ]:
            item["photo_visits"] = {"photo-0": 59, "photo-1": 491461}
    elif broken == "source_count":
        example["coverage_sidecar"]["by_dataset"]["toy_a"]["seen_photos"] = 0
    elif broken == "split_leak":
        example["rows"].append(
            dict(example["rows"][0], id="alias", photo_id="other-eye", split="val")
        )
    elif broken == "sidecar":
        example["validation_sidecar"]["selection_score"] = 0.1
    elif broken == "wrong_plan":
        example["provenance"]["plan_sha256"] = "9077"
    with pytest.raises(ValueError):
        qa.validate_terminal(**example)


def test_independent_copy_is_exact_and_refuses_overwrite(tmp_path):
    destination = tmp_path / "epoch-060"
    qa.copy_milestone(destination, {"last.pt": b"fixture bytes", "summary.json": b"{}\n"})
    assert (destination / "last.pt").read_bytes() == b"fixture bytes"
    with pytest.raises(ValueError, match="preserved"):
        qa.copy_milestone(destination, {"last.pt": b"changed"})
    assert (destination / "last.pt").read_bytes() == b"fixture bytes"
