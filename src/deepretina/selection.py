"""CPU-only provenance gate for a future unified held-out evaluation.

This module neither evaluates a split nor enables the public test command.
Hashes bind current byte snapshots; unsigned records cannot prove prior blindness.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import pickle
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from .integrity import FINGERPRINT_SCHEMA, verify_prepared_content
from .unified import (
    CHANNELS,
    SOURCE_CHANNELS,
    UnifiedConfig,
    _metrics,
    inference_config,
    load_unified,
    map_record_targets,
)

VERSION = 1
EPOCHS = 60
COUNT_KEYS = ("tp", "fp", "fn", "valid_pixels")
SUPPORT = {
    name: "reference_class_labeled_vessel_pixels"
    if i in (1, 2)
    else "annotated_valid_pixels_within_field_of_view"
    for i, name in enumerate(CHANNELS)
}
POLICY = "one photo once; first sorted record per task; AV vessel annotation takes priority; all splits fixed by manifest"


class SelectionError(ValueError):
    """The frozen decision/evidence does not match the actual requested inputs."""


def require(condition, message):
    if not condition:
        raise SelectionError(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise SelectionError("Expected finite JSON-serializable evidence") from error


def digest(value):
    return hashlib.sha256(value).hexdigest()


def exact(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), f"Invalid {label} fields")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(snapshot):
    try:
        return json.loads(
            snapshot,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SelectionError(f"Nonfinite JSON: {value}")
            ),
        )
    except (ValueError, UnicodeError) as error:
        raise SelectionError(f"Invalid JSON evidence: {error}") from error


def json_snapshot(path):
    snapshot = Path(path).read_bytes()
    return snapshot, parse_json(snapshot)


def utc(value):
    try:
        timestamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise SelectionError("Decision/evaluation time must be ISO8601 UTC") from error
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0,
        "Decision/evaluation time must be timezone-aware UTC",
    )
    return timestamp


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def evaluation_source_identity():
    """Hash actual evaluation/gate implementation, not only a Git ref or filename."""
    root = Path(__file__).parent
    result = hashlib.sha256()
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    for path in root.rglob("*"):
        require(
            not (path.is_symlink() and path.is_dir()),
            "Symlinked runtime directories cannot be frozen completely",
        )
    source_files = []
    for path in files:
        relative = "src/deepretina/" + path.relative_to(root).as_posix()
        result.update(relative.encode() + b"\0" + path.read_bytes())
        source_files.append(relative)
    project = root.parents[1]
    require(
        (project / "uv.lock").is_file() and (project / "pyproject.toml").is_file(),
        "Selection freezing requires a pinned source checkout with uv.lock",
    )
    for name in ("pyproject.toml", "uv.lock"):
        result.update(name.encode() + b"\0" + (project / name).read_bytes())
    distributions = {}
    for name in (
        "torch",
        "numpy",
        "Pillow",
        "scipy",
        "scikit-image",
        "opencv-python",
        "opencv-python-headless",
        "h5py",
        "pydicom",
    ):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "evaluation_source_sha256": result.hexdigest(),
        "source_files": source_files + ["pyproject.toml", "uv.lock"],
        "source_status": "actual package Python source bytes and project metadata; not an in-memory or binary attestation",
        "hardware_status": "not queried by CPU metadata gate; fresh validation/test producer must log actual device and driver separately",
        "runtime": {
            "installed_distributions": distributions,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
    }


def config_identity(payload):
    config = asdict(UnifiedConfig(**payload["config"]))
    config.pop("device")  # Hardware choice is not a training hyperparameter.
    return {"values": config, "sha256": digest(canonical(config))}


def inference_protocol(payload, *, threshold=0.5, tile_size=None, overlap=None):
    require(
        type(threshold) in (int, float) and math.isfinite(threshold) and 0 < threshold < 1,
        "Threshold must be finite and in (0,1)",
    )
    config = inference_config(payload, tile_size=tile_size, overlap=overlap)
    return {
        "version": VERSION,
        "geometry": "canonical_prepared",
        "normalization": "RGB/255-0.5",
        "preprocessing_profile": config.preprocessing_profile,
        **(
            {"photometry": payload["preprocessing"]["photometry"]}
            if config.photometric_profile != "raw"
            else {}
        ),
        "pixels_per_degree": config.pixels_per_degree,
        "preparation_max_size": payload.get("inference_max_size"),
        "channels": list(CHANNELS),
        "mode": "whole" if config.train_view == "whole" else "tiled",
        "tile_size": config.tile_size if config.train_view == "patch" else None,
        "overlap": config.overlap if config.train_view == "patch" else None,
        "view_size": config.view_size if config.train_view == "whole" else None,
        "threshold": float(threshold),
        "precision": "float32_no_autocast",
        "activation": [
            "sigmoid",
            "conditional_softmax",
            "conditional_softmax",
            "sigmoid",
            "sigmoid",
        ],
        "av_fusion": "joint_av_mixture_then_conditional",
        "av_decoding": "argmax_tie_artery_gated_by_vessel",
        "cup_containment": "cup_and_disc",
        "metric_support": SUPPORT,
        "annotation_policy": POLICY,
        "aggregation": "pooled_counts_per_dataset_per_head_then_macro_heads",
    }


def checkpoint_snapshot(path):
    require(
        torch.get_default_dtype() == torch.float32, "Frozen protocol requires float32 default dtype"
    )
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "Frozen protocol rejects active autocast",
    )
    snapshot = Path(path).read_bytes()
    try:
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            model, payload, _ = load_unified(io.BytesIO(snapshot), device="cpu")
    except (
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise SelectionError(f"Invalid safe unified checkpoint: {error}") from error
    config = UnifiedConfig(**payload["config"])
    require(
        type(payload.get("epoch")) is int
        and 1 <= payload["epoch"] <= EPOCHS
        and type(payload.get("steps")) is int
        and payload["steps"] > 0,
        "Invalid checkpoint epoch/step metadata",
    )
    require(
        config.purpose == payload.get("purpose") == "full_data",
        "Reject functionality/pilot checkpoints",
    )
    require(
        config.epochs == EPOCHS and config.max_steps is None,
        "Require planned full60 budget without max_steps",
    )
    require(
        config.legacy_initialization is None and not config.allow_historical_exposure,
        "Historical teacher initialization is ineligible",
    )
    lineage = payload.get("lineage", {})
    require(
        lineage.get("historical_exposure") is False
        and lineage.get("independent_model_eligible") is True,
        "Require explicit independent scratch lineage",
    )
    flags = payload.get("supervised_channels")
    require(
        isinstance(flags, list) and len(flags) == 5 and all(type(v) is bool and v for v in flags),
        "Require all5 supervised heads",
    )
    require(
        payload.get("architecture") == model.architecture_spec,
        "Architecture/configuration mismatch",
    )
    require(_finite(payload["state_dict"]), "Nonfinite model weights")
    require(
        all(
            not value.is_floating_point() or value.dtype == torch.float32
            for value in payload["state_dict"].values()
        ),
        "Frozen protocol requires float32 stored weights",
    )
    require(payload.get("normalization") == "RGB/255-0.5", "Normalization mismatch")
    require(
        payload.get("probability_fusion") == "joint_av_mixture_then_conditional",
        "Fusion protocol mismatch",
    )
    require(
        isinstance(payload.get("training_source_sha256"), str)
        and len(payload["training_source_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in payload["training_source_sha256"]),
        "Missing training source fingerprint",
    )
    require(
        payload.get("checkpoint_sha256") == digest(snapshot), "Checkpoint snapshot digest mismatch"
    )
    return snapshot, payload


def _canonical_metadata(manifest):
    """Operation-scoped observations, not a promise of future storage immutability."""
    manifest = Path(manifest)
    root = manifest.parent.resolve()
    rows = parse_json(manifest.read_bytes())
    photos = parse_json((root / "photos.json").read_bytes())
    paths = {manifest.name, "photos.json", "integrity.json", "_SUCCESS"}
    for row in rows:
        paths.update((row["image"], row["mask"]))
    for photo in photos:
        paths.update((photo["image"], photo["fov"]))
    observed = {}
    for relative in sorted(paths):
        candidate = Path(relative)
        path = root / candidate
        require(
            not candidate.is_absolute()
            and ".." not in candidate.parts
            and path.resolve(strict=True).is_relative_to(root)
            and not path.is_symlink(),
            "Canonical path escaped immutable input tree",
        )
        info = path.stat()
        observed[relative] = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
        )
    return observed


def _finish_canonical_read(manifest, expected, before):
    require(
        _canonical_metadata(manifest) == before,
        "Canonical data changed during selection verification",
    )
    verify_prepared_content(manifest, expected=expected)
    require(
        _canonical_metadata(manifest) == before,
        "Canonical data changed during selection verification",
    )


def data_identity(manifest, checkpoint):
    manifest = Path(manifest)
    metadata_before = _canonical_metadata(manifest)
    before, rows = json_snapshot(manifest)
    require(
        isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
        "Expected canonical record list",
    )
    require(digest(before) == checkpoint["manifest_sha256"], "Manifest differs from checkpoint")
    fingerprint = verify_prepared_content(manifest, expected=checkpoint["prepared_content_sha256"])
    require(manifest.read_bytes() == before, "Manifest changed during identity verification")
    audit = parse_json((manifest.parent / "integrity.json").read_bytes())
    require(audit.get("preview_only") is False, "Final selection rejects preview preparation")
    groups = {split: {} for split in ("train", "val", "test")}
    seen_ids, partitions = set(), {}
    for row in rows:
        require(row["id"] not in seen_ids, "Duplicate manifest record ID")
        seen_ids.add(row["id"])
        require(row["split"] in groups, "Unknown manifest split")
        require(
            bool(row.get("photo_id")) and bool(row.get("source_id")),
            "Require explicit portable photo/source IDs",
        )
        for kind in ("photo_id", "source_id"):
            key = (kind, row[kind])
            require(
                key not in partitions or partitions[key] == row["split"],
                "Source/photo crosses held-out partitions",
            )
            partitions[key] = row["split"]
        if (
            row["task"] in SOURCE_CHANNELS
            and row.get("usable_for_training", True)
            and not row.get("exclude_from_training", False)
        ):
            metadata = row.get("metadata", {})
            require(
                not row.get("preparation_preview", metadata.get("preparation_preview", False))
                and row.get("preparation_max_size") is None,
                "Preview/capped preparation is ineligible",
            )
            require(
                row.get("preprocessing_profile", metadata.get("preprocessing_profile"))
                == "legacy25_raw"
                and row.get("pixels_per_degree", metadata.get("pixels_per_degree")) == 25,
                "Require explicit canonical legacy25 preprocessing",
            )
            groups[row["split"]].setdefault(row["photo_id"], []).append(row)
    for split, photos in groups.items():
        require(bool(photos), f"Missing eligible {split} photos")
        groups[split] = {
            pid: sorted(values, key=lambda r: r["id"]) for pid, values in sorted(photos.items())
        }
    inventory = {
        split: [
            {
                "photo_id": pid,
                "records": [
                    {key: row[key] for key in ("id", "dataset", "source_id", "task")}
                    for row in values
                ],
            }
            for pid, values in values.items()
        ]
        for split, values in groups.items()
    }
    require(
        _canonical_metadata(manifest) == metadata_before,
        "Canonical data changed during identity verification",
    )
    return {
        "manifest_sha256": digest(before),
        "prepared_content_sha256": fingerprint,
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "split_inventory_sha256": digest(canonical(inventory)),
        "eligible_photo_counts": {split: len(values) for split, values in groups.items()},
    }, groups


def _counts(value):
    require(
        isinstance(value, dict)
        and all(type(value.get(k)) is int and value[k] >= 0 for k in COUNT_KEYS),
        "Invalid confusion counts",
    )
    values = [value[key] for key in COUNT_KEYS]
    require(sum(values[:3]) <= values[3], "Confusion exceeds valid support")
    require(_metrics(values) == value, "Reported per-photo metric differs from counts")
    return np.asarray(values, np.int64)


def reference_support(rows, root):
    """Decode VAL GT only, with the exact shared mapper; never read test labels."""
    options = {task: [row for row in rows if row["task"] == task] for task in SOURCE_CHANNELS}
    tasks = ["av"] if options["av"] else ["vessels"] if options["vessels"] else []
    tasks += ["disc_cup"] if options["disc_cup"] else []
    counts = {channel: (0, 0) for channel in CHANNELS}
    for task in tasks:
        row = options[task][0]
        with np.load(root / row["mask"], allow_pickle=False) as label:
            target, valid = label["target"].astype(np.float32), label["valid"].astype(bool)
            require(
                "fov" in label and label["fov"].shape == target.shape[-2:],
                "VAL FOV geometry differs",
            )
            valid &= label["fov"][None].astype(bool)
            mapped, support = map_record_targets(row, target, valid)
        for channel in SOURCE_CHANNELS[task]:
            counts[CHANNELS[channel]] = (
                int(support[channel].sum()),
                int(((mapped[channel] >= 0.5) & support[channel]).sum()),
            )
    return counts


def validate_metrics(metrics, groups, protocol, config, root, *, expected_split="val"):
    require(expected_split in {"val", "test"}, "Unsupported checked metric split")
    require(
        metrics.get("family") == "unified"
        and metrics.get("split") == expected_split
        and metrics.get("purpose") == "full_data",
        f"Require full-data {expected_split.upper()} metrics",
    )
    require(metrics.get("supervised_channels") == [True] * 5, "VAL evidence lacks5 heads")
    require(
        metrics.get("threshold") == protocol["threshold"]
        and metrics.get("metric_support") == SUPPORT
        and metrics.get("evaluation_policy") == POLICY,
        "VAL support/protocol differs",
    )
    require(
        metrics.get("photos") == len(groups)
        and metrics.get("records") == sum(map(len, groups.values())),
        "VAL inventory counts differ",
    )
    photos = metrics.get("per_photo")
    require(isinstance(photos, list) and len(photos) == len(groups), "VAL photo inventory missing")
    require(
        len({p["photo_id"] for p in photos}) == len(photos)
        and {p["photo_id"] for p in photos} == set(groups),
        "VAL evidence contains missing/duplicate/non-val photo IDs",
    )
    totals, by_dataset = np.zeros((5, 4), np.int64), {}
    conditional = end_to_end = known = 0
    for photo in photos:
        rows = groups[photo["photo_id"]]
        gt_support = reference_support(rows, root)
        require(photo["record_ids"] == [row["id"] for row in rows], "VAL record IDs differ")
        options = {task: [row for row in rows if row["task"] == task] for task in SOURCE_CHANNELS}
        tasks = ["av"] if options["av"] else ["vessels"] if options["vessels"] else []
        tasks += ["disc_cup"] if options["disc_cup"] else []
        attribution = {}
        for task in tasks:
            for channel in SOURCE_CHANNELS[task]:
                attribution[CHANNELS[channel]] = options[task][0]["dataset"]
        require(
            photo["channel_datasets"] == attribution and photo["dataset"] == rows[0]["dataset"],
            "VAL annotation-source attribution differs",
        )
        require(set(photo["channels"]) == set(CHANNELS), "Missing VAL per-photo channel")
        for i, channel in enumerate(CHANNELS):
            values = _counts(photo["channels"][channel])
            require(
                (int(values[3]), int(values[0] + values[2])) == gt_support[channel],
                "VAL valid/positive GT support differs from actual canonical annotations",
            )
            totals[i] += values
            source = attribution.get(channel)
            require(source is not None or not values.any(), "Unsupported channel has scored pixels")
            if source is not None:
                by_dataset.setdefault(source, np.zeros((5, 4), np.int64))[i] += values
        cc = photo.get("av_conditional_counts", {})
        exact(
            cc, ("correct", "scored_pixels", "eligible_pixels", "end_to_end_correct"), "AV counts"
        )
        require(all(type(v) is int and v >= 0 for v in cc.values()), "Invalid AV counts")
        a, v = photo["channels"]["artery"], photo["channels"]["vein"]
        require(
            cc["scored_pixels"] == cc["eligible_pixels"] == a["valid_pixels"] == v["valid_pixels"],
            "AV support mismatch",
        )
        require(
            a["tp"] + v["fp"] <= a["tp"] + a["fn"] and v["tp"] + a["fp"] <= v["tp"] + v["fn"],
            "AV exclusive decoder counts impossible for positive GT support",
        )
        require(
            cc["end_to_end_correct"]
            == a["tp"] + v["tp"]
            <= cc["correct"]
            <= cc["scored_pixels"] - a["fp"] - v["fp"],
            "AV correct counts inconsistent with exclusive conditional decoding",
        )
        conditional += cc["correct"]
        end_to_end += cc["end_to_end_correct"]
        known += cc["scored_pixels"]
    channels = {name: _metrics(value) for name, value in zip(CHANNELS, totals, strict=True)}
    datasets = {
        name: {channel: _metrics(value) for channel, value in zip(CHANNELS, values, strict=True)}
        for name, values in by_dataset.items()
    }
    require(
        metrics["channels"] == channels and metrics["by_dataset"] == datasets,
        "VAL aggregate count headers differ",
    )
    require(
        all(
            value["valid_pixels"] > 0 and value["tp"] + value["fn"] > 0
            for value in channels.values()
        ),
        "VAL evidence lacks positive/valid5-head support",
    )
    macro = {
        channel: float(
            np.mean(
                [
                    values[channel]["dice"]
                    for values in datasets.values()
                    if values[channel]["dice"] is not None
                ]
            )
        )
        for channel in CHANNELS
    }
    pooled = float(np.mean([value["dice"] for value in channels.values()]))
    mean = float(np.mean(list(macro.values())))
    require(
        metrics["macro_dataset_channel_dice"] == macro
        and metrics["macro_dataset_mean_dice"] == mean
        and metrics["mean_dice"] == pooled,
        "VAL macro/pooled metrics differ from counts",
    )
    require(
        known > 0
        and metrics["av_labeled_pixels"] == known
        and metrics["av_accuracy"] == end_to_end / known
        and metrics["av_conditional_accuracy"] == conditional / known,
        "VAL AV headers differ",
    )
    if expected_split == "val":
        require(
            metrics.get("selection_metric") == config.selection_metric
            and metrics.get("selection_score") == metrics[config.selection_metric],
            "VAL selection metric/score mismatch",
        )
    else:
        require(
            "selection_metric" not in metrics and "selection_score" not in metrics,
            "Held-out metrics must not declare a new model-selection score",
        )
    return metrics[config.selection_metric]


def completion_evidence(best, last, metrics_snapshot, summary_snapshot, groups):
    require(config_identity(best) == config_identity(last), "Best/last configuration differs")
    for key in (
        "manifest_sha256",
        "prepared_content_sha256",
        "training_source_sha256",
        "derived_cache",
    ):
        require(best.get(key) == last.get(key), f"Best/last {key} differs")
    require(
        type(last.get("epoch")) is int and last["epoch"] == EPOCHS,
        "Training has not actually completed60 epochs",
    )
    require(type(best.get("epoch")) is int and 1 <= best["epoch"] <= EPOCHS, "Invalid best epoch")
    summary = parse_json(summary_snapshot)
    require(
        summary.get("completed_budget") is True and summary.get("steps") == last.get("steps"),
        "Training completion summary is incomplete/inconsistent",
    )
    state = last.get("training_state", {})
    config = UnifiedConfig(**best["config"])
    if config.scheduler == "cosine":
        require(
            state.get("scheduler", {}).get("T_max") == EPOCHS
            and state["scheduler"].get("last_epoch") == EPOCHS,
            "Scheduler did not complete planned60 horizon",
        )
    else:
        require(state.get("scheduler") is None, "Constant schedule has unexpected scheduler state")
    require(
        bool(state.get("optimizer", {}).get("state")) and _finite(state["optimizer"]),
        "Missing/nonfinite optimizer state",
    )
    optimizer_steps = [
        value["step"] for value in state["optimizer"]["state"].values() if "step" in value
    ]
    require(
        bool(optimizer_steps) and max(float(value) for value in optimizer_steps) == last["steps"],
        "Optimizer update count differs from completed history",
    )
    coverage = last.get("coverage", {})
    require(
        state.get("coverage") == coverage, "Last checkpoint coverage disagrees with training state"
    )
    require(
        set(coverage.get("photo_visits", {})) == set(groups)
        and coverage.get("eligible_photos") == coverage.get("seen_photos") == len(groups),
        "Incomplete training photo coverage",
    )
    require(
        all(type(v) is int and v > 0 for v in coverage["photo_visits"].values()),
        "Invalid photo visit counts",
    )
    records = {row["id"] for rows in groups.values() for row in rows}
    visits = coverage.get("annotation_visits", {})
    require(
        set(visits) <= records
        and all(type(v) is int and v > 0 for v in visits.values())
        and coverage.get("eligible_annotation_records") == len(records)
        and coverage.get("contributing_annotation_records") == len(visits),
        "Training annotation coverage differs",
    )
    families = {}
    for pid, rows in groups.items():
        for row in rows:
            families.setdefault(row["dataset"], set()).add(pid)
    expected_families = {
        name: {"eligible_photos": len(ids), "seen_photos": len(ids)}
        for name, ids in families.items()
    }
    require(coverage.get("by_dataset") == expected_families, "Training family coverage differs")
    require(
        sum(coverage["photo_visits"].values()) <= EPOCHS * best["config"]["epoch_samples"],
        "Training visit count exceeds planned sample budget",
    )
    rows = list(csv.DictReader(io.StringIO(metrics_snapshot.decode())))
    require(
        [int(row["epoch"]) for row in rows] == list(range(1, EPOCHS + 1)),
        "Training metrics do not cover60 unique epochs",
    )
    require(
        all(math.isfinite(float(v)) for row in rows for v in row.values()),
        "Nonfinite training metrics",
    )
    steps = [int(row["steps"]) for row in rows]
    require(
        all(a < b for a, b in zip([0, *steps[:-1]], steps, strict=True))
        and steps[-1] == last["steps"]
        and steps[best["epoch"] - 1] == best["steps"],
        "Training optimizer-step history differs",
    )
    for number, row in enumerate(rows):
        expected_lr = (
            (
                config.lr * config.min_lr_ratio
                + (config.lr - config.lr * config.min_lr_ratio)
                * (1 + math.cos(math.pi * number / EPOCHS))
                / 2
            )
            if config.scheduler == "cosine"
            else config.lr
        )
        require(
            math.isclose(float(row["lr"]), expected_lr, rel_tol=1e-10, abs_tol=1e-12),
            "Training LR history differs from frozen scheduler/configuration",
        )
        require(
            float(row["val_selection_score"]) == float(row["val_" + config.selection_metric]),
            "Training selection-score column differs from configured metric",
        )
    scores = [float(row["val_selection_score"]) for row in rows]
    require(
        best["val_selection_score"] == scores[best["epoch"] - 1] == max(scores)
        and summary.get("best_validation_score") == max(scores)
        and state.get("best_validation_score") == max(scores)
        and last["val_selection_score"] == scores[-1]
        and summary.get("selection_metric") == config.selection_metric,
        "Best checkpoint does not match completed-run selection",
    )
    return {
        "planned_epochs": EPOCHS,
        "completed_epochs": EPOCHS,
        "last_epoch": last["epoch"],
        "last_checkpoint_sha256": last["checkpoint_sha256"],
        "last_steps": last["steps"],
        "best_epoch": best["epoch"],
        "best_training_selection_score": best["val_selection_score"],
        "metrics_csv_sha256": digest(metrics_snapshot),
        "training_summary_sha256": digest(summary_snapshot),
        "coverage_sha256": digest(canonical(coverage)),
    }


def _context(
    checkpoint,
    last_checkpoint,
    manifest,
    validation_evidence,
    metrics_csv,
    training_summary,
    *,
    threshold=0.5,
    tile_size=None,
    overlap=None,
):
    snapshot, best = checkpoint_snapshot(checkpoint)
    _, last = checkpoint_snapshot(last_checkpoint)
    canonical_before = _canonical_metadata(manifest)
    data, groups = data_identity(manifest, best)
    config = UnifiedConfig(**best["config"])
    protocol = inference_protocol(best, threshold=threshold, tile_size=tile_size, overlap=overlap)
    evidence_snapshot, evidence = json_snapshot(validation_evidence)
    exact(
        evidence,
        (
            "schema_version",
            "kind",
            "checkpoint",
            "configuration",
            "source",
            "data",
            "inference_protocol",
            "evaluated_at_utc",
            "metrics",
        ),
        "enriched validation evidence",
    )
    require(
        type(evidence["schema_version"]) is int
        and evidence["schema_version"] == VERSION
        and evidence["kind"] == "unified_validation_evidence",
        "Require freshly enriched VAL evidence; thin historical reports are not eligible",
    )
    identity = {
        "sha256": best["checkpoint_sha256"],
        "epoch": best["epoch"],
        "steps": best["steps"],
        "training_selection_score": best["val_selection_score"],
    }
    require(
        evidence["checkpoint"] == identity and evidence["configuration"] == config_identity(best),
        "VAL checkpoint/configuration binding differs",
    )
    source = {
        **evaluation_source_identity(),
        "training_source_sha256": best["training_source_sha256"],
    }
    require(
        evidence["source"] == source
        and evidence["data"] == data
        and evidence["inference_protocol"] == protocol,
        "VAL source/data/inference protocol binding differs",
    )
    require(
        evidence["metrics"].get("lineage") == best["lineage"], "VAL lineage differs from checkpoint"
    )
    score = validate_metrics(
        evidence["metrics"], groups["val"], protocol, config, Path(manifest).parent
    )
    completion = completion_evidence(
        best,
        last,
        Path(metrics_csv).read_bytes(),
        Path(training_summary).read_bytes(),
        groups["train"],
    )
    _finish_canonical_read(manifest, best["prepared_content_sha256"], canonical_before)
    require(
        {**evaluation_source_identity(), "training_source_sha256": best["training_source_sha256"]}
        == source,
        "Source/runtime changed while checking selection evidence",
    )
    require(
        _canonical_metadata(manifest) == canonical_before,
        "Canonical data changed during selection verification",
    )
    checkpoint_identity = {
        **identity,
        "configuration": config_identity(best),
        "architecture": best["architecture"],
        "channels": list(CHANNELS),
        "lineage": best["lineage"],
    }
    return snapshot, {
        "checkpoint": checkpoint_identity,
        "source": source,
        "canonical_data": data,
        "inference_protocol": protocol,
        "training_completion": completion,
        "validation_evidence": {
            "sha256": digest(evidence_snapshot),
            "evaluated_at_utc": evidence["evaluated_at_utc"],
            "selection_metric": config.selection_metric,
            "selection_score": score,
        },
    }


@dataclass(frozen=True)
class CheckedSelection:
    """Immutable snapshot for future consumers: never reopen the checkpoint path."""

    selection_sha256: str
    checkpoint_sha256: str
    checkpoint_snapshot: bytes = field(repr=False)
    protocol_snapshot: bytes = field(repr=False)
    canonical_data_snapshot: bytes = field(repr=False)
    test_evaluation_enabled: bool = False

    @property
    def canonical_data(self):
        return parse_json(self.canonical_data_snapshot)

    @property
    def protocol(self):
        return parse_json(self.protocol_snapshot)


def create_selection(*, output, reviewer, rationale, **inputs):
    require(
        isinstance(reviewer, str)
        and bool(reviewer.strip())
        and isinstance(rationale, str)
        and bool(rationale.strip()),
        "Explicit reviewer and decision rationale are required",
    )
    _, context = _context(**inputs)
    now = datetime.now(UTC)
    require(
        utc(context["validation_evidence"]["evaluated_at_utc"]) <= now,
        "VAL evidence is dated after the decision",
    )
    artifact = {
        "schema_version": VERSION,
        "kind": "unified_heldout_selection",
        "decision": {
            "recorded_at_utc": now.isoformat(),
            "reviewer": reviewer,
            "rationale": rationale,
        },
        **context,
        "boundaries": {
            "test_evaluation_enabled": False,
            "prior_blindness_proven": False,
            "reviewer_identity_authenticated": False,
        },
    }
    snapshot = canonical(artifact) + b"\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(snapshot)
    return {
        "selection_sha256": digest(snapshot),
        "checkpoint_sha256": context["checkpoint"]["sha256"],
        "test_evaluation_enabled": False,
    }


def check_selection(*, selection, expected_selection_sha256, **inputs):
    snapshot, artifact = json_snapshot(selection)
    require(
        digest(snapshot) == expected_selection_sha256,
        "Selection differs from externally pinned review digest",
    )
    exact(
        artifact,
        (
            "schema_version",
            "kind",
            "decision",
            "checkpoint",
            "source",
            "canonical_data",
            "inference_protocol",
            "training_completion",
            "validation_evidence",
            "boundaries",
        ),
        "selection artifact",
    )
    require(
        type(artifact["schema_version"]) is int
        and artifact["schema_version"] == VERSION
        and artifact["kind"] == "unified_heldout_selection",
        "Unsupported frozen selection schema",
    )
    decision = artifact["decision"]
    exact(decision, ("recorded_at_utc", "reviewer", "rationale"), "decision")
    require(
        all(
            isinstance(decision[k], str) and decision[k].strip() for k in ("reviewer", "rationale")
        ),
        "Missing explicit decision/reviewer",
    )
    recorded = utc(decision["recorded_at_utc"])
    require(recorded <= datetime.now(UTC), "Decision timestamp is in the future")
    weights, context = _context(**inputs)
    require(
        utc(context["validation_evidence"]["evaluated_at_utc"]) <= recorded,
        "Validation evidence postdates the recorded decision",
    )
    for key, value in context.items():
        require(
            artifact[key] == value, f"Frozen {key} differs from actual requested inputs/evidence"
        )
    require(
        artifact["boundaries"]
        == {
            "test_evaluation_enabled": False,
            "prior_blindness_proven": False,
            "reviewer_identity_authenticated": False,
        },
        "Invalid stage1 boundary declaration",
    )
    return CheckedSelection(
        digest(snapshot),
        context["checkpoint"]["sha256"],
        weights,
        canonical(context["inference_protocol"]),
        canonical(context["canonical_data"]),
    )
