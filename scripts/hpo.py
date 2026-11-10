"""Bounded validation-only learning-rate search on an immutable prepared manifest.

Search writes an immutable selection before the separate --evaluate-selected-test
operation may access the test split. Existing stable baselines are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from deepretina.engine import TrainConfig, evaluate, train


@dataclass(frozen=True)
class SearchConfig:
    task: str
    lrs: tuple[float, ...] = (0.0003, 0.001, 0.003)
    stage_epochs: int = 8
    final_epochs: int = 30
    top_k: int = 2
    seed: int = 42
    batch_size: int = 4
    patch_size: int = 256
    base_channels: int = 16
    depth: int = 3
    samples_per_image: int = 8
    workers: int = 2
    loss: str = "bce_dice"
    device: str = "auto"
    tile_size: int = 512
    overlap: int = 64
    cpu_threads: int = 2

    def validate(self):
        if self.task not in {"vessels", "av", "disc_cup"}:
            raise ValueError("Unsupported task")
        if (
            not self.lrs
            or len(set(self.lrs)) != len(self.lrs)
            or any(not math.isfinite(v) or v <= 0 for v in self.lrs)
        ):
            raise ValueError("Learning rates must be unique finite positive numbers")
        if not 1 <= self.top_k <= len(self.lrs) or not 1 <= self.stage_epochs <= self.final_epochs:
            raise ValueError(
                "Require 1 <= top_k <= candidates and 1 <= stage_epochs <= final_epochs"
            )
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")

    def trial(self, lr, epochs):
        return TrainConfig(
            task=self.task,
            epochs=epochs,
            lr=lr,
            seed=self.seed,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            base_channels=self.base_channels,
            depth=self.depth,
            samples_per_image=self.samples_per_image,
            workers=self.workers,
            loss=self.loss,
            device=self.device,
            tile_size=self.tile_size,
            overlap=self.overlap,
            max_steps=None,
        )


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _manifest_hash(manifest):
    return hashlib.sha256(Path(manifest).read_bytes()).hexdigest()


def _config_equal(actual, expected, *, baseline=False):
    # Device placement is runtime infrastructure, not a model hyperparameter.
    ignored = {"device", "lr"} if baseline else {"device"}
    new_defaults = {
        "norm": "group",
        "train_view": "patch",
        "view_size": 512,
        "foreground_probability": 0.0,
        "augmentation": "basic",
        "scheduler": "constant",
        "min_lr_ratio": 0.05,
        "epoch_samples": None,
        "sampling": "uniform",
    }
    for key, value in expected.items():
        if key not in ignored and actual.get(key, new_defaults.get(key)) != value:
            raise ValueError(
                f"Run configuration differs in {key}: {actual.get(key)!r} versus {value!r}"
            )


def _completed(run: Path, expected: TrainConfig, manifest_hash: str, phase: str, *, baseline=False):
    metadata = json.loads((run / "run.json").read_text())
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Manifest hash differs for {run}")
    _config_equal(metadata["config"], asdict(expected), baseline=baseline)
    validation = json.loads((run / "validation.json").read_text())
    if validation.get("split") != "val":
        raise ValueError(f"Expected validation metrics, not held-out test data: {run}")
    score = validation["mean_dice"]
    if score is None or not math.isfinite(score):
        raise ValueError(f"Undefined validation score: {run}")
    with (run / "metrics.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history or int(history[-1]["epoch"]) != expected.epochs:
        raise ValueError(f"Run incomplete: {run}")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if saved.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Checkpoint manifest differs: {run}")
    _config_equal(saved["config"], asdict(expected), baseline=baseline)
    if not math.isclose(float(saved["val_mean_dice"]), score, abs_tol=1e-12):
        raise ValueError(f"Checkpoint/validation scores differ: {run}")
    return {
        "phase": phase,
        "lr": metadata["config"]["lr"],
        "epochs": expected.epochs,
        "validation_mean_dice": score,
        "best_epoch": saved["epoch"],
        "steps": saved["steps"],
        "run": str(run.resolve()),
        "checkpoint": str((run / "best.pt").resolve()),
        "reused_baseline": baseline,
    }


def _csv(output, candidates):
    fields = [
        "phase",
        "lr",
        "epochs",
        "validation_mean_dice",
        "best_epoch",
        "steps",
        "run",
        "checkpoint",
        "reused_baseline",
    ]
    with (output / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)


def search(manifest: Path, output: Path, baseline_run: Path, config: SearchConfig, *, resume=False):
    """Run three short trials, promote top-k, then compare against a full baseline.

    No test metrics are read or computed. A completed matching baseline is always
    eligible, regardless of the short-stage ranking. Each trial trains from scratch.
    """
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    manifest, output, baseline_run = (
        Path(manifest).resolve(),
        Path(output).resolve(),
        Path(baseline_run).resolve(),
    )
    if output == baseline_run or baseline_run.is_relative_to(output):
        raise ValueError("HPO output must be separate from the read-only baseline")
    manifest_hash = _manifest_hash(manifest)
    settings = {
        "config": asdict(config),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "baseline_run": str(baseline_run),
        "selection_split": "val",
        "test_access_during_search": False,
    }
    # JSON normalizes tuples, enabling stable equality across process restarts.
    settings = json.loads(json.dumps(settings))
    baseline = _completed(
        baseline_run,
        config.trial(config.lrs[0], config.final_epochs),
        manifest_hash,
        "baseline",
        baseline=True,
    )
    if output.exists() and any(output.iterdir()):
        if not resume or not (output / "search.json").is_file():
            raise FileExistsError(
                f"Choose a new output directory, or --resume the identical search: {output}"
            )
        if json.loads((output / "search.json").read_text()) != settings:
            raise ValueError("Cannot resume a search with changed configuration, paths or manifest")
        if (output / "selection.json").is_file():
            return json.loads((output / "selection.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "configs").mkdir()
        _write(output / "search.json", settings)
        _write(
            output / "runtime.json",
            {
                "host": socket.gethostname(),
                "torch": str(torch.__version__),
                "started_utc": datetime.now(UTC).isoformat(),
                "cpu_threads": config.cpu_threads,
            },
        )
    candidates = [baseline]
    _csv(output, candidates)

    def candidate(lr, epochs, phase):
        trial = config.trial(lr, epochs)
        name = f"{phase}_lr_{lr:.8g}"
        _write(output / "configs" / (name + ".json"), asdict(trial))
        if phase == "final" and lr == baseline["lr"]:
            result = dict(baseline, phase="final", reused_baseline=True)
        else:
            path = output / name
            if path.exists() and any(path.iterdir()):
                try:
                    result = _completed(path, trial, manifest_hash, phase)
                except (FileNotFoundError, ValueError, KeyError):
                    archive = output / "interrupted" / f"{name}_{time.time_ns()}"
                    archive.parent.mkdir(exist_ok=True)
                    shutil.move(str(path), str(archive))
                    print(json.dumps({"archived_incomplete_trial": str(archive)}), flush=True)
                    train(manifest, path, trial)
                    result = _completed(path, trial, manifest_hash, phase)
            else:
                train(manifest, path, trial)
                result = _completed(path, trial, manifest_hash, phase)
        candidates.append(result)
        _csv(output, candidates)
        _write(
            output / "progress.json",
            {"completed": candidates, "last_completed_utc": datetime.now(UTC).isoformat()},
        )
        print(json.dumps({"candidate_completed": result}), flush=True)
        return result

    stage = [candidate(lr, config.stage_epochs, "stage") for lr in config.lrs]
    # Stable tie-breaking follows the caller's LR order; no test data participates.
    promoted = sorted(
        [trial for trial in stage if trial["lr"] != baseline["lr"]],
        key=lambda r: -r["validation_mean_dice"],
    )[: config.top_k]
    final = [candidate(item["lr"], config.final_epochs, "final") for item in promoted]
    eligible = [baseline, *final]
    winner = max(eligible, key=lambda r: r["validation_mean_dice"])
    selection = {
        "task": config.task,
        "manifest_sha256": manifest_hash,
        "selection_split": "val",
        "test_access_during_search": False,
        "baseline": baseline,
        "promoted_learning_rates": [r["lr"] for r in promoted],
        "eligible_finalists": eligible,
        "winner": winner,
        "validation_delta_vs_baseline": winner["validation_mean_dice"]
        - baseline["validation_mean_dice"],
        "threshold": 0.5,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "completed_utc": datetime.now(UTC).isoformat(),
        "caveat": "Single seed and original pilot split; validation selection is not proof of generalization. This test split was already reported in the original pilot.",
    }
    _write(output / "selection.json", selection)
    return selection


def evaluate_selected_test(manifest: Path, output: Path, device="auto"):
    """One explicit test evaluation after the validation selection is persisted."""
    selection_path = Path(output) / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("selection_split") != "val"
        or selection.get("test_access_during_search") is not False
    ):
        raise ValueError("A completed validation-only selection is required first")
    if _manifest_hash(manifest) != selection["manifest_sha256"]:
        raise ValueError("Test evaluation must use the identical prepared manifest")
    result_path = Path(output) / "selected_test.json"
    if result_path.exists():
        raise FileExistsError("Selected test was already evaluated; existing results are preserved")
    torch.set_num_threads(2)
    scores = evaluate(
        Path(selection["winner"]["checkpoint"]),
        manifest,
        split="test",
        device=device,
        tile_size=selection["tile_size"],
        overlap=selection["overlap"],
        threshold=selection["threshold"],
    )
    result = {
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selection": selection["winner"],
        "test_previously_reported_in_original_pilot": True,
        "metrics": scores,
    }
    _write(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["vessels", "av", "disc_cup"])
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.0003, 0.001, 0.003])
    parser.add_argument("--stage-epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--loss", choices=["bce", "bce_dice"], default="bce_dice")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-selected-test", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate_selected_test:
        result = evaluate_selected_test(args.manifest, args.output, args.device)
    else:
        if args.task is None or args.baseline_run is None:
            parser.error("search requires --task and --baseline-run")
        config = SearchConfig(
            task=args.task,
            lrs=tuple(args.lrs),
            stage_epochs=args.stage_epochs,
            final_epochs=args.final_epochs,
            top_k=args.top_k,
            loss=args.loss,
            device=args.device,
            cpu_threads=args.cpu_threads,
        )
        result = search(args.manifest, args.output, args.baseline_run, config, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
