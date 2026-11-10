"""Post-hoc paired checkpoint preservation gate; no training/inference imports.

All weights are loaded safely from one hashed byte snapshot. Differences are
reported, never rounded into equality. Passing one pair does not prove universal
CUDA determinism, particularly with deterministic_algorithms(warn_only=True).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import torch

MISSING = object()
PROVENANCE_FIELDS = ("training_source_sha256", "derived_cache")


def snapshot(path):
    data = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a primitive state dictionary payload")
    return payload, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def safe_scalar(value):
    if value is MISSING:
        return {"missing": True}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(map(repr, value))}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    return value


def compare(a, b, path="root"):
    """Full recursive comparison, retaining every unequal leaf/missing key."""
    result = {
        "equal": True,
        "strict_tensor_equal": True,
        "tensors": 0,
        "differences": [],
        "nonfinite": [],
    }

    def difference(location, detail):
        result["equal"] = False
        result["differences"].append({"path": location, **detail})

    def visit(x, y, location):
        if x is MISSING or y is MISSING:
            difference(location, {"old": safe_scalar(x), "new": safe_scalar(y)})
            remaining = y if x is MISSING else x
            if isinstance(remaining, torch.Tensor):
                result["strict_tensor_equal"] = False
                if remaining.is_floating_point() or remaining.is_complex():
                    count = int((~torch.isfinite(remaining)).sum())
                    if count:
                        result["nonfinite"].append(
                            {
                                "path": location,
                                "side": "new" if x is MISSING else "old",
                                "count": count,
                            }
                        )
            elif isinstance(remaining, dict):
                for key, value in remaining.items():
                    visit(
                        MISSING if x is MISSING else value,
                        value if x is MISSING else MISSING,
                        f"{location}[{key!r}]",
                    )
            elif isinstance(remaining, (list, tuple)):
                for i, value in enumerate(remaining):
                    visit(
                        MISSING if x is MISSING else value,
                        value if x is MISSING else MISSING,
                        f"{location}[{i}]",
                    )
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            result["tensors"] += 1
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                result["strict_tensor_equal"] = False
                difference(location, {"tensor_type_mismatch": True})
                return
            for side, tensor in [("old", x), ("new", y)]:
                if tensor.is_floating_point() or tensor.is_complex():
                    count = int((~torch.isfinite(tensor)).sum())
                    if count:
                        result["equal"] = False
                        result["nonfinite"].append({"path": location, "side": side, "count": count})
            equal = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
            result["strict_tensor_equal"] &= equal
            if not equal:
                detail = {
                    "old_dtype": str(x.dtype),
                    "new_dtype": str(y.dtype),
                    "old_shape": list(x.shape),
                    "new_shape": list(y.shape),
                }
                if x.shape == y.shape:
                    detail["different_elements"] = int((x != y).sum())
                    dtype = torch.complex128 if x.is_complex() or y.is_complex() else torch.float64
                    delta = (x.to(dtype) - y.to(dtype)).abs()
                    finite = torch.isfinite(delta)
                    detail["finite_delta_elements"] = int(finite.sum())
                    detail["nonfinite_delta_elements"] = int((~finite).sum())
                    detail["max_abs_finite"] = float(delta[finite].max()) if finite.any() else None
                difference(location, detail)
        elif isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y), key=repr):
                visit(x.get(key, MISSING), y.get(key, MISSING), f"{location}[{key!r}]")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if type(x) is not type(y) or len(x) != len(y):
                difference(
                    location,
                    {
                        "old_sequence_type": type(x).__name__,
                        "new_sequence_type": type(y).__name__,
                        "old_length": len(x),
                        "new_length": len(y),
                    },
                )
            for i in range(max(len(x), len(y))):
                visit(
                    x[i] if i < len(x) else MISSING,
                    y[i] if i < len(y) else MISSING,
                    f"{location}[{i}]",
                )
        else:
            for side, value in [("old", x), ("new", y)]:
                if isinstance(value, float) and not math.isfinite(value):
                    result["equal"] = False
                    result["nonfinite"].append(
                        {"path": location, "side": side, "value": repr(value)}
                    )
            if type(x) is not type(y) or x != y:
                difference(location, {"old": safe_scalar(x), "new": safe_scalar(y)})

    visit(a, b, path)
    return result


def validation_view(metrics):
    if not isinstance(metrics, dict):
        return metrics
    header = {key: value for key, value in metrics.items() if key != "per_photo"}
    photos = {}
    for photo in metrics.get("per_photo", []):
        photo_id = photo["photo_id"]
        if photo_id in photos:
            raise ValueError(f"Duplicate validation photo ID: {photo_id}")
        channels = {}
        for channel, value in photo["channels"].items():
            channels[channel] = {
                key: value.get(key, MISSING) for key in ("tp", "fp", "fn", "valid_pixels")
            }
        photos[photo_id] = channels
    return {"header": header, "per_photo_channel_counts": photos}


def csv_epoch(path, epoch):
    data = Path(path).read_bytes()
    rows = [row for row in csv.DictReader(io.StringIO(data.decode())) if int(row["epoch"]) == epoch]
    if len(rows) != 1:
        raise ValueError(f"Require exactly one CSV row for epoch {epoch}: {path}")
    row = rows[0]
    values = {
        name: float(row[name]) for name in ("train_loss", "train_seconds", "validation_seconds")
    }
    values.update(epoch=int(row["epoch"]), steps=int(row["steps"]))
    if any(
        not math.isfinite(values[name])
        for name in ("train_loss", "train_seconds", "validation_seconds")
    ):
        raise ValueError(f"Non-finite CSV loss/time: {path}")
    if values["train_seconds"] <= 0 or values["validation_seconds"] < 0:
        raise ValueError(f"Invalid measured times: {path}")
    return values, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def build_report(old, new, old_csv, new_csv, epoch=1, old_wall_seconds=None, new_wall_seconds=None):
    left, left_file = snapshot(old)
    right, right_file = snapshot(new)
    required = (
        "state_dict",
        "training_state",
        "coverage",
        "config",
        "epoch",
        "steps",
        "manifest_sha256",
        "prepared_content_sha256",
        "validation_metrics",
    )
    issues = [
        f"{side} checkpoint missing {field}"
        for side, payload in [("old", left), ("new", right)]
        for field in required
        if field not in payload
    ]
    for side, payload in [("old", left), ("new", right)]:
        if payload.get("family") != "unified" or payload.get("epoch") != epoch:
            issues.append(f"{side} checkpoint is not unified requested epoch {epoch}")
        for field in (
            "manifest_sha256",
            "prepared_content_sha256",
            "training_source_sha256",
        ):
            if not payload.get(field):
                issues.append(f"{side} checkpoint lacks {field} identity")
        validation = payload.get("validation_metrics", {})
        if not isinstance(validation.get("per_photo"), list) or not isinstance(
            validation.get("channels"), dict
        ):
            issues.append(f"{side} validation lacks per_photo/channel records")
        state = payload.get("training_state", {})
        for field in ("optimizer", "scheduler", "rng", "coverage"):
            if field not in state:
                issues.append(f"{side} training_state missing {field}")
        cache = payload.get("derived_cache")
        if cache and (
            cache.get("canonical_fingerprint") != payload.get("prepared_content_sha256")
            or cache.get("allow_canonical_fallback") is not False
        ):
            issues.append(f"{side} cache identity/data or fallback policy inconsistent")
    if not isinstance(right.get("derived_cache"), dict) or not right["derived_cache"].get(
        "fingerprint"
    ):
        issues.append("new checkpoint must pin its derived cache fingerprint")
    sections = {}
    for field in ("state_dict", "training_state", "coverage"):
        sections[field] = compare(left.get(field, MISSING), right.get(field, MISSING), field)
    configs = [
        {key: value for key, value in payload.get("config", {}).items() if key != "device"}
        for payload in (left, right)
    ]
    sections["config_except_device"] = compare(*configs, path="config")
    excluded = {
        "state_dict",
        "training_state",
        "coverage",
        "config",
        "validation_metrics",
        *PROVENANCE_FIELDS,
    }
    sections["checkpoint_metadata"] = compare(
        {k: v for k, v in left.items() if k not in excluded},
        {k: v for k, v in right.items() if k not in excluded},
        path="checkpoint_metadata",
    )
    sections["validation"] = compare(
        validation_view(left.get("validation_metrics", MISSING)),
        validation_view(right.get("validation_metrics", MISSING)),
        path="validation",
    )
    old_row, old_csv_file = csv_epoch(old_csv, epoch)
    new_row, new_csv_file = csv_epoch(new_csv, epoch)
    sections["csv_loss_steps"] = compare(
        {k: old_row[k] for k in ("epoch", "steps", "train_loss")},
        {k: new_row[k] for k in ("epoch", "steps", "train_loss")},
        path="metrics_csv",
    )
    for side, row, payload in [("old", old_row, left), ("new", new_row, right)]:
        if row["steps"] != payload.get("steps"):
            issues.append(f"{side} CSV/checkpoint steps differ")
    timing = {
        "old_csv": old_row,
        "new_csv": new_row,
        "train_speedup": old_row["train_seconds"] / new_row["train_seconds"],
        "train_plus_validation_speedup": (old_row["train_seconds"] + old_row["validation_seconds"])
        / (new_row["train_seconds"] + new_row["validation_seconds"]),
        "scope": "CSV times exclude startup/support scan/cache verification; not end-to-end wall time",
    }
    if old_wall_seconds is not None or new_wall_seconds is not None:
        if not all(
            value is not None and math.isfinite(value) and value > 0
            for value in (old_wall_seconds, new_wall_seconds)
        ):
            raise ValueError("Provide both positive finite external wall times")
        timing.update(
            old_wall_seconds=old_wall_seconds,
            new_wall_seconds=new_wall_seconds,
            external_wall_speedup=old_wall_seconds / new_wall_seconds,
        )
    provenance = {}
    for field in (*PROVENANCE_FIELDS, "manifest_sha256", "prepared_content_sha256"):
        provenance[field] = {
            "old_present": field in left,
            "new_present": field in right,
            "old": left.get(field),
            "new": right.get(field),
        }
    provenance["config_device"] = {
        "old": left.get("config", {}).get("device"),
        "new": right.get("config", {}).get("device"),
    }
    return {
        "schema": 1,
        "epoch": epoch,
        "preservation_equal": not issues and all(s["equal"] for s in sections.values()),
        "strict_tensor_equal": all(s["strict_tensor_equal"] for s in sections.values()),
        "issues": issues,
        "sections": sections,
        "provenance": provenance,
        "inputs": {
            "old_checkpoint": left_file,
            "new_checkpoint": right_file,
            "old_metrics_csv": old_csv_file,
            "new_metrics_csv": new_csv_file,
        },
        "timing": timing,
        "documented_differences": [
            "Executable source fingerprint and optional derived-cache identity intentionally differ and are retained above; canonical data identity must remain equal",
            "Config device is provenance only; all other config keys compare strictly",
            "Per-photo validation compares photo IDs and channel confusion counts only; new av_conditional_counts and derived per-photo metric ratios do not enter this old/new gate",
        ],
        "limitation": "One paired snapshot comparison is not universal bitwise/determinism proof; CUDA warn_only may permit nondeterministic kernels. Every finite/missing/tensor difference remains reported.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "old_csv", "new_csv", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--old-wall-seconds", type=float)
    parser.add_argument("--new-wall-seconds", type=float)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Existing comparison report preserved: {args.output}")
    torch.set_num_threads(2)
    report = build_report(
        args.old,
        args.new,
        args.old_csv,
        args.new_csv,
        args.epoch,
        args.old_wall_seconds,
        args.new_wall_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "preservation_equal": report["preservation_equal"],
                "strict_tensor_equal": report["strict_tensor_equal"],
            }
        )
    )


if __name__ == "__main__":
    main()
