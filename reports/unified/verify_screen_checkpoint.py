"""Read-only screen QA plus independent checkpoint milestone; never submits work."""

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import torch

from deepretina import unified
from deepretina.integrity import verify_prepared_content


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Nonfinite checkpoint tensor"
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value), "Nonfinite checkpoint scalar"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "run", "plan", "manifest", "milestone", "report"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        os.kill(args.training_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Original training PID is still alive")
    assert Path(unified.__file__).resolve() == (args.source / "src/deepretina/unified.py").resolve()
    assert sha(args.plan) == "907713f6c4a94c7281e2a5e226777fb02a5cdf613b0d79c20e0fe14831cf5e14"
    plan = json.loads(args.plan.read_text())
    digest = hashlib.sha256()
    for path in sorted(
        [
            *args.source.glob("src/**/*.py"),
            *args.source.glob("scripts/*.py"),
            args.source / "pyproject.toml",
            args.source / "uv.lock",
        ]
    ):
        digest.update(path.relative_to(args.source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    assert (
        digest.hexdigest()
        == plan["execution_code_sha256"]
        == "8172e06bfa2c61497d3a34d443027f2f0193834090e95530322e3a6f795254fe"
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"]
    ).strip()
    assert sha(args.manifest) == plan["manifest_sha256"]
    assert (
        verify_prepared_content(args.manifest, plan["prepared_content_sha256"])
        == plan["prepared_content_sha256"]
    )
    train_photos = {
        r["photo_id"] for r in json.loads(args.manifest.read_text()) if r["split"] == "train"
    }
    last = torch.load(args.run / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    config = dict(plan["candidates"][args.trial])
    config.pop("device")
    training_sha = unified._training_source_sha256()
    for saved in [last, best]:
        observed = dict(saved["config"])
        observed.pop("device")
        assert observed == config
        assert saved["manifest_sha256"] == plan["manifest_sha256"]
        assert saved["prepared_content_sha256"] == plan["prepared_content_sha256"]
        assert saved["training_source_sha256"] == training_sha
        finite(saved["state_dict"])
    assert last["epoch"] == 8 and last["steps"] == 16384
    assert config["epochs"] == 60 and config["batch_size"] == 4 and config["epoch_samples"] == 8192
    finite(last["training_state"]["optimizer"])
    scheduler = last["training_state"]["scheduler"]
    assert scheduler["T_max"] == 60 and scheduler["last_epoch"] == 8
    coverage = last["coverage"]
    assert coverage["eligible_photos"] == coverage["seen_photos"] == len(train_photos) == 2221
    assert set(coverage["photo_visits"]) == train_photos
    assert sum(coverage["photo_visits"].values()) == 65536
    assert all(
        row["eligible_photos"] == row["seen_photos"] for row in coverage["by_dataset"].values()
    )
    provenance = json.loads((args.run.parent / (args.run.name + ".search_trial.json")).read_text())
    assert provenance == {
        "plan_sha256": sha(args.plan),
        "trial": args.trial,
        "execution_code_sha256": plan["execution_code_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
    }
    args.milestone.mkdir(parents=True, exist_ok=True)
    assert not any(args.milestone.iterdir()), "Existing milestone preserved"
    for name in ["best.pt", "last.pt"]:
        os.link(args.run / name, args.milestone / name)
    for name in [
        "metrics.csv",
        "validation.json",
        "coverage.json",
        "run.json",
        "training_summary.json",
    ]:
        shutil.copy2(args.run / name, args.milestone / name)
    (args.milestone / "validation-last.json").write_text(
        json.dumps(last["validation_metrics"], indent=2) + "\n"
    )

    def metrics(saved):
        value = saved["validation_metrics"]
        return {
            "epoch": saved["epoch"],
            "macro": value["macro_dataset_mean_dice"],
            "pooled_mean": value["mean_dice"],
            "pooled_dice": {k: v["dice"] for k, v in value["channels"].items()},
            "macro_channel_dice": value["macro_dataset_channel_dice"],
            "av_conditional_accuracy": value["av_conditional_accuracy"],
            "av_end_to_end_accuracy": value["av_accuracy"],
        }

    report = {
        "verified_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "trial": args.trial,
        "original_training_pid": args.training_pid,
        "terminal": True,
        "last": metrics(last),
        "best": metrics(best),
        "finite_weights_optimizer": True,
        "coverage_all2221": True,
        "delivered_samples": 65536,
        "steps": 16384,
        "scheduler_horizon": 60,
        "scheduler_last_epoch": 8,
        "source_commit": plan["source_commit"],
        "execution_code_sha256": digest.hexdigest(),
        "checkpoint_training_source_sha256": training_sha,
        "canonical_fingerprint": plan["prepared_content_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "plan_sha256": sha(args.plan),
        "milestone": str(args.milestone),
        "checkpoint_sha256": {name: sha(args.milestone / name) for name in ["best.pt", "last.pt"]},
        "no_test_evaluation": True,
        "torch": str(torch.__version__),
    }
    (args.milestone / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
