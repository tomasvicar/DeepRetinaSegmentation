"""Bounded validation-only architecture, augmentation and training-budget search.

Run from the experiments/leipzig-hpo worktree. Plans are JSON, original baselines
are read-only, and a frozen selection is required before explicit test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from deepretina.engine import TrainConfig, evaluate, train
from deepretina.inference import load_checkpoint


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(config):
    """Fill newly introduced fields with their backward-compatible defaults."""
    return asdict(TrainConfig(**config))


def equal_config(actual, expected):
    actual, expected = normalized(actual), normalized(expected)
    for name in expected:
        if name != "device" and actual[name] != expected[name]:
            raise ValueError(
                f"Configuration differs in {name}: {actual[name]!r} != {expected[name]!r}"
            )


def completed(run, expected, manifest_hash, phase, name):
    run = Path(run)
    metadata = json.loads((run / "run.json").read_text())
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Manifest changed for {run}")
    equal_config(metadata["config"], asdict(expected))
    validation = json.loads((run / "validation.json").read_text())
    if validation.get("split") != "val":
        raise ValueError(f"Expected validation-only evidence for {run}")
    score = validation["mean_dice"]
    if score is None or not math.isfinite(score):
        raise ValueError(f"Undefined validation score for {run}")
    with (run / "metrics.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history or int(history[-1]["epoch"]) != expected.epochs:
        raise ValueError(f"Incomplete {expected.epochs}-epoch run: {run}")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if saved["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Checkpoint manifest changed for {run}")
    equal_config(saved["config"], asdict(expected))
    if not math.isclose(saved["val_mean_dice"], score, abs_tol=1e-12):
        raise ValueError(f"Checkpoint and validation report disagree for {run}")
    return {
        "name": name,
        "phase": phase,
        "epochs": expected.epochs,
        "config": asdict(expected),
        "validation_mean_dice": score,
        "best_epoch": int(saved["epoch"]),
        "steps": int(saved["steps"]),
        "run": str(run.resolve()),
        "checkpoint": str((run / "best.pt").resolve()),
        "checkpoint_sha256": digest(run / "best.pt"),
    }


def resolve_plan(manifest, plan_path, baseline_run, incumbent_run, device):
    manifest = Path(manifest).resolve()
    plan = json.loads(Path(plan_path).read_text())
    baseline_run = Path(baseline_run).resolve()
    metadata = json.loads((baseline_run / "run.json").read_text())
    base = TrainConfig(**metadata["config"])
    if base.epochs != 30 or base.max_steps is not None:
        raise ValueError("The original baseline must be a completed full 30-epoch run")
    if plan["task"] != base.task:
        raise ValueError("Plan and baseline tasks differ")
    stage_epochs, final_epochs = (
        int(plan.get("stage_epochs", 12)),
        int(plan.get("final_epochs", 60)),
    )
    top_k = int(plan.get("top_k", 2))
    if not 1 <= stage_epochs < final_epochs or final_epochs < base.epochs:
        raise ValueError("Require 1 <= screening epochs < final epochs and final budget >=30")
    if not 1 <= top_k <= len(plan["candidates"]):
        raise ValueError("Invalid finalist count")
    names = list(plan["candidates"])
    if any(not name or not all(c.isalnum() or c in "_-" for c in name) for name in names):
        raise ValueError("Candidate names must contain only letters, numbers, underscore or dash")
    manifest_hash = digest(manifest)
    baseline = completed(
        baseline_run, base, manifest_hash, "original_baseline", "original_baseline_30"
    )
    incumbent = None
    candidate_lr = base.lr
    if incumbent_run is not None:
        incumbent_run = Path(incumbent_run).resolve()
        incumbent_metadata = json.loads((incumbent_run / "run.json").read_text())
        incumbent_config = TrainConfig(**incumbent_metadata["config"])
        # The previous bounded LR search may change LR only, not architecture or labels.
        equal_config(asdict(replace(incumbent_config, lr=base.lr)), asdict(base))
        incumbent = completed(
            incumbent_run, incumbent_config, manifest_hash, "incumbent", "previous_lr_winner"
        )
        candidate_lr = incumbent_config.lr
    candidate_base = replace(base, device=device, lr=candidate_lr)
    candidates = {}
    forbidden = {
        "task",
        "epochs",
        "device",
        "seed",
        "max_steps",
        "workers",
        "samples_per_image",
        "batch_size",
        "tile_size",
        "overlap",
    }
    for name, overrides in plan["candidates"].items():
        if forbidden.intersection(overrides):
            raise ValueError(
                f"{name}: candidate cannot override fixed protocol fields {sorted(forbidden.intersection(overrides))}"
            )
        candidates[name] = asdict(replace(candidate_base, epochs=stage_epochs, **overrides))
    return {
        "task": base.task,
        "plan": plan,
        "plan_sha256": digest(plan_path),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "baseline": baseline,
        "incumbent": incumbent,
        "matched_baseline_config": asdict(replace(base, epochs=final_epochs, device=device)),
        "candidates": candidates,
        "stage_epochs": stage_epochs,
        "final_epochs": final_epochs,
        "top_k": top_k,
        "selection_split": "val",
        "test_access_during_search": False,
    }


def summary_csv(output, rows):
    fields = [
        "name",
        "phase",
        "epochs",
        "validation_mean_dice",
        "best_epoch",
        "steps",
        "run",
        "checkpoint",
        "checkpoint_sha256",
        "train_seconds",
    ]
    with (output / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def search(settings, output, *, resume=False, cpu_threads=4):
    output = Path(output).resolve()
    read_only_runs = [settings["baseline"]]
    if settings["incumbent"] is not None:
        read_only_runs.append(settings["incumbent"])
    for item in read_only_runs:
        protected = Path(item["run"])
        if (
            output == protected
            or protected.is_relative_to(output)
            or output.is_relative_to(protected)
        ):
            raise ValueError("Output must be separate from every read-only baseline/incumbent")
    if output.exists() and any(output.iterdir()):
        if not resume or not (output / "search.json").exists():
            raise FileExistsError("Choose a fresh output or resume an identical search")
        if json.loads((output / "search.json").read_text()) != settings:
            raise ValueError("Plan, baseline, manifest or resolved configuration changed")
        if (output / "selection.json").exists():
            return json.loads((output / "selection.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "configs").mkdir()
        write_json(output / "search.json", settings)
        write_json(
            output / "runtime.json",
            {
                "host": socket.gethostname(),
                "started_utc": datetime.now(UTC).isoformat(),
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "cpu_threads": cpu_threads,
                "environment": {
                    key: os.environ.get(key)
                    for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "runner_sha256": digest(__file__),
            },
        )
    torch.set_num_threads(cpu_threads)
    rows = [settings["baseline"]]
    if (
        settings["incumbent"]
        and settings["incumbent"]["checkpoint"] != settings["baseline"]["checkpoint"]
    ):
        rows.append(settings["incumbent"])
    failures = []
    summary_csv(output, rows)

    def candidate(name, phase, config_dict):
        config = TrainConfig(**config_dict)
        run = output / f"{phase}_{name}"
        write_json(output / "configs" / f"{phase}_{name}.json", asdict(config))
        if run.exists() and any(run.iterdir()):
            try:
                result = completed(run, config, settings["manifest_sha256"], phase, name)
            except (FileNotFoundError, ValueError, KeyError):
                archive = output / "interrupted" / f"{phase}_{name}_{time.time_ns()}"
                archive.parent.mkdir(exist_ok=True)
                shutil.move(str(run), str(archive))
            else:
                rows.append(result)
                summary_csv(output, rows)
                return result
        started = time.monotonic()
        try:
            train(Path(settings["manifest"]), run, config)
        except torch.cuda.OutOfMemoryError as error:
            failure = {
                "name": name,
                "phase": phase,
                "config": asdict(config),
                "error": str(error),
                "run": str(run),
            }
            failures.append(failure)
            write_json(output / "failures.json", failures)
            gc.collect()
            torch.cuda.empty_cache()
            return None
        result = completed(run, config, settings["manifest_sha256"], phase, name)
        result["train_seconds"] = time.monotonic() - started
        write_json(
            run / "trial_timing.json",
            {"seconds": result["train_seconds"], "includes_validation": True},
        )
        rows.append(result)
        summary_csv(output, rows)
        write_json(
            output / "progress.json",
            {"completed": rows, "failures": failures, "updated_utc": datetime.now(UTC).isoformat()},
        )
        print(json.dumps({"candidate_completed": result}), flush=True)
        return result

    stage = [candidate(name, "stage", config) for name, config in settings["candidates"].items()]
    stage = [item for item in stage if item is not None]
    promoted = sorted(stage, key=lambda r: -r["validation_mean_dice"])[: settings["top_k"]]
    # A longer budget alone can improve the old architecture; measure that control explicitly.
    matched = candidate("matched_baseline", "final", settings["matched_baseline_config"])
    if matched is None:
        raise RuntimeError("Matched-budget baseline failed; no fair longer-budget comparison")
    final = []
    for item in promoted:
        cfg = dict(item["config"], epochs=settings["final_epochs"])
        final.append(candidate(item["name"], "final", cfg))
    eligible = [
        item
        for item in [settings["baseline"], settings["incumbent"], matched, *final]
        if item is not None
    ]
    eligible = list({item["checkpoint"]: item for item in eligible}.values())
    winner = max(eligible, key=lambda r: r["validation_mean_dice"])
    selection = {
        "task": settings["task"],
        "manifest_sha256": settings["manifest_sha256"],
        "selection_split": "val",
        "test_access_during_search": False,
        "promoted_names": [item["name"] for item in promoted],
        "eligible_finalists": eligible,
        "winner": winner,
        "validation_delta_vs_original_30": winner["validation_mean_dice"]
        - settings["baseline"]["validation_mean_dice"],
        "validation_delta_vs_matched_budget": winner["validation_mean_dice"]
        - matched["validation_mean_dice"],
        "completed_utc": datetime.now(UTC).isoformat(),
        "threshold": 0.5,
        "tile_size": 512,
        "overlap": 64,
        "caveat": "Single seed, small original pilot validation set and previously reported test set. Multiple simultaneous configuration changes are not individual-factor ablations.",
    }
    write_json(output / "selection.json", selection)
    return selection


def benchmark_selection(manifest, output, device, *, warmup=2, repeats=5):
    """Benchmark complete inference on one fixed validation photograph, excluding load/export."""
    from deepretina.dataset import read_manifest, read_sample
    from deepretina.inference import probabilities_for_config

    output = Path(output)
    selection = json.loads((output / "selection.json").read_text())
    if digest(manifest) != selection["manifest_sha256"]:
        raise ValueError("Benchmark manifest changed")
    if (output / "benchmark.json").exists():
        return json.loads((output / "benchmark.json").read_text())
    row = min(read_manifest(manifest, selection["task"], "val"), key=lambda r: r["id"])
    image, _, _ = read_sample(Path(manifest).parent, row)
    measurements = []

    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for finalist in selection["eligible_finalists"]:
        model, saved, actual_device = load_checkpoint(Path(finalist["checkpoint"]), device)
        parameters = sum(parameter.numel() for parameter in model.parameters())

        infer = partial(
            probabilities_for_config,
            model,
            image,
            actual_device,
            saved["config"],
            tile_size=512,
            overlap=64,
        )

        for _ in range(warmup):
            infer()
        synchronize(actual_device)
        if actual_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actual_device)
        times = []
        for _ in range(repeats):
            synchronize(actual_device)
            start = time.perf_counter()
            result = infer()
            synchronize(actual_device)
            times.append(time.perf_counter() - start)
            if result.shape[-2:] != image.shape[-2:] or not np.isfinite(result).all():
                raise ValueError("Benchmark inference returned invalid image geometry/values")
        measurements.append(
            {
                "name": finalist["name"],
                "checkpoint": finalist["checkpoint"],
                "validation_mean_dice": finalist["validation_mean_dice"],
                "parameters": parameters,
                "checkpoint_bytes": Path(finalist["checkpoint"]).stat().st_size,
                "seconds": times,
                "median_seconds": statistics.median(times),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(actual_device)
                if actual_device.type == "cuda"
                else None,
            }
        )
        del model, infer
        gc.collect()
        if actual_device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "host": socket.gethostname(),
        "device": str(actual_device),
        "input_id": row["id"],
        "input_sha256": digest(Path(manifest).parent / row["image"]),
        "shape": list(image.shape),
        "warmup": warmup,
        "repeats": repeats,
        "tile_size": 512,
        "overlap": 64,
        "scope": "prepared validation image; complete probability inference including CPU blending/resize; excludes checkpoint load and file export",
        "measurements": measurements,
    }
    write_json(output / "benchmark.json", result)
    return result


def evaluate_selected_test(manifest, output, device):
    output = Path(output)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("selection_split") != "val"
        or selection.get("test_access_during_search") is not False
    ):
        raise ValueError("Frozen validation-only selection required")
    if digest(manifest) != selection["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if (output / "selected_test.json").exists():
        raise FileExistsError("Selected test already evaluated")
    scores = evaluate(
        Path(selection["winner"]["checkpoint"]),
        Path(manifest),
        split="test",
        device=device,
        tile_size=512,
        overlap=64,
        threshold=0.5,
    )
    result = {
        "selection_sha256": digest(selection_path),
        "selection": selection["winner"],
        "test_previously_reported_in_original_pilot": True,
        "metrics": scores,
    }
    write_json(output / "selected_test.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--incumbent-run", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-selected-test", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.evaluate_selected_test:
        result = evaluate_selected_test(args.manifest, args.output, args.device)
    else:
        if args.plan is None or args.baseline_run is None:
            parser.error("Search requires --plan and --baseline-run")
        settings = resolve_plan(
            args.manifest, args.plan, args.baseline_run, args.incumbent_run, args.device
        )
        settings["cpu_threads"] = args.cpu_threads
        if args.dry_run:
            result = settings
        else:
            result = search(settings, args.output, resume=args.resume, cpu_threads=args.cpu_threads)
            if not args.no_benchmark:
                benchmark_selection(args.manifest, args.output, args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
