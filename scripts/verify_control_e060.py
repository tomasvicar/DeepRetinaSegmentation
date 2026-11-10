"""Terminal-only 3157 control QA. No inference, training, migration or overwrite."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path

SOURCE = "3157bb957fcf6e6f632a5c49d6df2c8a78ea3747"
EXECUTION = "8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b"
TRAINING = "dd69012a1741000ae5841634094094397c690a2371c9a91aa411344a436f1136"
PLAN = "6be48b57fc85e8ffba7772058562aa5a8bf855f9cb445c5180484111ffba5778"
MANIFEST = "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"
CONTENT = "61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e"
CACHE = "4e39e40def03d0f2859dd147107a3736a62b4fc0249dd440a1280a9803aa8591"
CACHE_INDEX = "b624df6a28b3475d48d0a0b8f3287e47a4fdbd0451b719be105db569ab5629ba"
TRIAL = "lr_0003_control_cached"
PID = 1257093
TRAINING_FILES = (
    "unified.py",
    "unified_legacy.py",
    "unified_av.py",
    "model.py",
    "dataset.py",
    "augment.py",
    "inference.py",
    "integrity.py",
    "derived_cache.py",
    "legacy/single_net.py",
    "legacy/preprocessing.py",
)
SIDECARS = ("metrics.csv", "validation.json", "coverage.json", "run.json", "training_summary.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def finite(value):
    import torch

    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite tensor")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite scalar")


def normalized(config):
    return {key: value for key, value in config.items() if key != "device"}


def manifest_groups(rows):
    partitions, photos, datasets = {}, set(), {}
    for row in rows:
        for key in ("source_id", "photo_id"):
            identity = (key, row[key])
            require(
                identity not in partitions or partitions[identity] == row["split"],
                "Source/photo split leakage",
            )
            partitions[identity] = row["split"]
        if (
            row["split"] != "train"
            or row["task"] not in {"vessels", "av", "disc_cup"}
            or not row.get("usable_for_training", True)
            or row.get("exclude_from_training", False)
        ):
            continue
        photos.add(row["photo_id"])
        datasets.setdefault(row["dataset"], set()).add(row["photo_id"])
    return photos, datasets


def validate_terminal(
    last, best, metrics, summary, run, coverage_sidecar, validation_sidecar, rows, plan, provenance
):
    """Pure payload checks: fixtures never load production checkpoints or data."""
    config = normalized(plan["candidates"][TRIAL])
    require(
        config["epochs"] == 60
        and config["epoch_samples"] == 8192
        and config["batch_size"] == 4
        and config["coverage_first"] is True,
        "Planned training budget changed",
    )
    require(
        len(metrics) == 60 and [int(row["epoch"]) for row in metrics] == list(range(1, 61)),
        "Need exactly contiguous CSV epochs1..60",
    )
    scores = []
    for epoch, row in enumerate(metrics, 1):
        for key, value in row.items():
            require(math.isfinite(float(value)), f"Nonfinite CSV {key}")
        require(int(row["steps"]) == epoch * 2048, "CSV step count differs")
        expected_lr = config["lr"] * (
            config["min_lr_ratio"]
            + (1 - config["min_lr_ratio"]) * (1 + math.cos(math.pi * (epoch - 1) / 60)) / 2
        )
        require(
            math.isclose(float(row["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-15),
            "CSV learning-rate schedule differs",
        )
        require(
            float(row["train_seconds"]) > 0 and float(row["validation_seconds"]) > 0,
            "Invalid timings",
        )
        score = float(row["val_selection_score"])
        scores.append(score)
        require(score == float(row["val_macro_dataset_mean_dice"]), "CSV selection metric differs")
    best_score = max(scores)
    best_epoch = scores.index(best_score) + 1
    require(last["epoch"] == 60 and last["steps"] == 122880, "Last is not terminal60")
    require(
        best["epoch"] == best_epoch and best["steps"] == best_epoch * 2048,
        "Best is not first strict maximum CSV epoch",
    )
    for payload in (last, best):
        require(
            payload["format_version"] == 1 and payload["family"] == "unified",
            "Wrong checkpoint family",
        )
        require(
            normalized(payload["config"]) == config, "Checkpoint config differs from exact plan"
        )
        require(
            payload["manifest_sha256"] == MANIFEST
            and payload["prepared_content_sha256"] == CONTENT
            and payload["training_source_sha256"] == TRAINING,
            "Checkpoint source/data identity differs",
        )
        require(
            payload["derived_cache"] == plan["derived_cache"], "Checkpoint cache identity differs"
        )
        require(
            payload["channels"] == ["vessel", "artery", "vein", "disc", "cup"]
            and payload["normalization"] == "RGB/255-0.5",
            "Output/input contract differs",
        )
        require(all(payload["supervised_channels"]), "All heads must be supervised")
        finite(payload["state_dict"])
        finite(payload["validation_metrics"])
        score = payload["validation_metrics"]["macro_dataset_mean_dice"]
        require(
            score == scores[payload["epoch"] - 1] == payload["val_selection_score"],
            "Checkpoint/CSV score differs",
        )
        require(
            payload["validation_metrics"]["selection_score"] == score,
            "Checkpoint selection score differs",
        )
    require(
        summary["completed_budget"] is True
        and summary["steps"] == 122880
        and summary["best_validation_score"] == best_score,
        "Terminal summary contradicts CSV/checkpoints",
    )
    require(
        summary["selection_metric"] == config["selection_metric"],
        "Summary selection metric differs",
    )
    state = last["training_state"]
    finite(state)
    require(
        state["best_validation_score"] == best_score
        and state["selection_metric"] == config["selection_metric"],
        "Resumable selection state differs",
    )
    require(
        bool(state["optimizer"]["state"]) and bool(state["optimizer"]["param_groups"]),
        "Missing optimizer state",
    )
    scheduler = state["scheduler"]
    require(
        scheduler["T_max"] == 60 and scheduler["last_epoch"] == 60,
        "Scheduler horizon/epoch differs",
    )
    rng = state["rng"]
    required = {"torch", "python", "numpy", "sampler", "augmentation", "cuda"}
    require(required <= rng.keys(), "Missing RNG component")
    require(
        rng["torch"] is not None
        and rng["python"] is not None
        and rng["numpy"] is not None
        and rng["sampler"] is not None
        and rng["cuda"] is not None,
        "Missing resumable CUDA RNG state",
    )
    # With workers2 augmentation generators live in workers, so parent augmentation=None is legitimate.
    photos, datasets = manifest_groups(rows)
    coverage = last["coverage"]
    require(coverage == state["coverage"] == coverage_sidecar, "Coverage sidecars differ")
    require(
        coverage["eligible_photos"] == coverage["seen_photos"] == len(photos),
        "Incomplete photo coverage",
    )
    visits = coverage["photo_visits"]
    require(set(visits) == photos, "Coverage photo set differs")
    require(
        all(type(value) is int and value >= 60 for value in visits.values()),
        "Minimum cumulative photo coverage below60",
    )
    require(sum(visits.values()) == 491520, "Delivered sample count differs")
    require(set(coverage["by_dataset"]) == set(datasets), "Coverage source inventory differs")
    require(set(datasets) == set(plan["datasets"]), "Plan source inventory differs")
    for name, eligible in datasets.items():
        require(
            coverage["by_dataset"][name]
            == {"eligible_photos": len(eligible), "seen_photos": len(eligible)},
            "Dataset photo coverage differs",
        )
    for key, value in {
        "config": config,
        "manifest_sha256": MANIFEST,
        "prepared_content_sha256": CONTENT,
        "derived_cache": plan["derived_cache"],
        "training_source_sha256": TRAINING,
    }.items():
        require(
            (normalized(run[key]) if key == "config" else run[key]) == value,
            "run.json provenance differs: " + key,
        )
    require(validation_sidecar == best["validation_metrics"], "Best validation sidecar differs")
    require(
        provenance
        == {
            "plan_sha256": PLAN,
            "trial": TRIAL,
            "execution_code_sha256": EXECUTION,
            "manifest_sha256": MANIFEST,
        },
        "Search trial provenance differs",
    )
    return {
        "last_epoch": 60,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "last_score": scores[-1],
        "steps": 122880,
        "delivered_samples": 491520,
        "eligible_photos": len(photos),
        "minimum_photo_visits": min(visits.values()),
        "by_dataset": coverage["by_dataset"],
        "known_train_source_groups": len(
            {
                row["source_id"]
                for row in rows
                if row["split"] == "train" and row["photo_id"] in photos
            }
        ),
        "finite_best_last_weights_and_last_optimizer": True,
        "rng_components": sorted(required),
        "best_checkpoint_has_optimizer": "training_state" in best,
    }


def snapshot(path):
    payload = Path(path).read_bytes()
    require(
        digest(Path(path).read_bytes()) == digest(payload),
        "File changed during snapshot: " + str(path),
    )
    return payload


def copy_milestone(destination, payloads):
    destination = Path(destination)
    require(
        not destination.exists() and not destination.is_symlink(), "Existing milestone preserved"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (destination / name).open("xb") as stream:
            stream.write(payload)
        require(
            digest((destination / name).read_bytes()) == digest(payload), "Copied milestone differs"
        )


def verify_source(root):
    require(
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        == SOURCE,
        "Source commit differs",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
        ).strip(),
        "Tracked immutable source changed",
    )
    actual = hashlib.sha256()
    for path in sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    ):
        actual.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    require(actual.hexdigest() == EXECUTION, "Execution byte hash differs")
    training = hashlib.sha256()
    for name in TRAINING_FILES:
        training.update(name.encode() + b"\0" + (root / "src/deepretina" / name).read_bytes())
    require(training.hexdigest() == TRAINING, "Training byte hash differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "run", "plan", "manifest", "cache", "milestone", "report"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        os.kill(PID, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Control PID1257093 still exists; terminal verification refused")
    require(
        not args.milestone.exists() and not args.report.exists(),
        "No existing output may be overwritten",
    )
    verify_source(args.source)
    plan_bytes = snapshot(args.plan)
    require(digest(plan_bytes) == PLAN, "Require exact6be48 plan")
    plan = json.loads(plan_bytes)
    require(
        plan["source_commit"] == SOURCE
        and plan["execution_code_sha256"] == EXECUTION
        and plan["training_source_sha256"] == TRAINING,
        "Plan source differs",
    )
    require(
        plan["manifest_sha256"] == MANIFEST and plan["prepared_content_sha256"] == CONTENT,
        "Plan data differs",
    )
    cache_identity = {
        "schema": 1,
        "fingerprint": CACHE,
        "canonical_fingerprint": CONTENT,
        "allow_canonical_fallback": False,
    }
    require(plan["derived_cache"] == cache_identity, "Plan cache pin differs")
    manifest_bytes = snapshot(args.manifest)
    require(digest(manifest_bytes) == MANIFEST, "Manifest bytes differ")
    rows = json.loads(manifest_bytes)
    photos, _ = manifest_groups(rows)
    require(len(photos) == 2221, "Production train inventory is not2221")
    # Import only after source and exact execution bytes are verified.
    import torch

    from deepretina import integrity
    from deepretina.integrity import verify_prepared_content

    require(
        Path(integrity.__file__).resolve()
        == (args.source / "src/deepretina/integrity.py").resolve(),
        "Wrong imported canonical verifier",
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(verify_prepared_content(args.manifest, CONTENT) == CONTENT, "Canonical bytes differ")
    index = snapshot(args.cache / "index.json")
    require(digest(index) == CACHE_INDEX, "Cache index pin differs")
    require((args.cache / "_SUCCESS").read_text().strip() == CACHE, "Cache completion pin differs")
    parsed_index = json.loads(index)
    require(
        parsed_index["canonical_fingerprint"] == CONTENT
        and parsed_index["canonical_manifest_sha256"] == MANIFEST,
        "Cache canonical identity differs",
    )
    names = ("last.pt", "best.pt", *SIDECARS)
    raw = {name: snapshot(args.run / name) for name in names}
    payload = {
        name: torch.load(io.BytesIO(raw[name]), map_location="cpu", weights_only=True)
        for name in ("last.pt", "best.pt")
    }
    provenance_path = args.run.parent / (args.run.name + ".search_trial.json")
    provenance = snapshot(provenance_path)
    result = validate_terminal(
        payload["last.pt"],
        payload["best.pt"],
        list(csv.DictReader(io.StringIO(raw["metrics.csv"].decode()))),
        json.loads(raw["training_summary.json"]),
        json.loads(raw["run.json"]),
        json.loads(raw["coverage.json"]),
        json.loads(raw["validation.json"]),
        rows,
        plan,
        json.loads(provenance),
    )
    # Refuse a late writer after validation; copied bytes are exactly those checked above.
    for name in names:
        require(
            digest((args.run / name).read_bytes()) == digest(raw[name]),
            "Run changed after QA: " + name,
        )
    verify_source(args.source)
    result.update(
        verified_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        terminal=True,
        original_pid=PID,
        source_commit=SOURCE,
        execution_code_sha256=EXECUTION,
        checkpoint_training_source_sha256=TRAINING,
        plan_sha256=PLAN,
        manifest_sha256=MANIFEST,
        canonical_fingerprint=CONTENT,
        derived_cache=cache_identity,
        cache_index_sha256=CACHE_INDEX,
        cache_payload_rehashed=False,
        cache_qa_scope="Pinned original index+completion marker and checkpoint/run/plan identity; prior complete cache byte audit retained, no new8GB payload scan",
        checkpoint_sha256={name: digest(raw[name]) for name in ("last.pt", "best.pt")},
        sidecar_sha256={name: digest(raw[name]) for name in SIDECARS},
        no_gpu_forward=True,
        no_test_evaluation=True,
        milestone=str(args.milestone),
        milestone_method="Independent copies of exactly checked byte snapshots; source hashes before/after QA and destination checks; no overwrite",
    )
    raw.update(
        {
            "search_trial.json": provenance,
            "frozen-plan.json": plan_bytes,
            "qa-summary.json": (json.dumps(result, indent=2) + "\n").encode(),
        }
    )
    copy_milestone(args.milestone, raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
