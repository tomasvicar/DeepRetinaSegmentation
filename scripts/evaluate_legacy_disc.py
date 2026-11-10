"""Validation-only, head-only comparison of the original optic-disc checkpoints.

Reuse the canonical GT choice/support and original tiling/normalization. Historical
training exposure/profile remain unresolved; this is no independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from deepretina.integrity import verify_prepared_content
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.preprocessing import local_contrast_and_clahe
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import UnifiedConfig, UnifiedDataset, _metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def disc_counts(logits, target, valid):
    logits = np.asarray(logits)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite original model output")
    prediction = logits > 0  # Original decoding: strict sign, no added hierarchy.
    target, valid = np.asarray(target, bool), np.asarray(valid, bool)
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Disc prediction/GT/support geometry differs")
    return np.array(
        [
            (prediction & target & valid).sum(),
            (prediction & ~target & valid).sum(),
            (~prediction & target & valid).sum(),
            valid.sum(),
        ],
        dtype=np.int64,
    )


def compare_reference(result, reference):
    if set(result["variants"]) != {"universal/raw"}:
        raise ValueError("Canary reference comparison requires only universal/raw")
    original = {r["photo_id"]: r for r in reference["per_photo"]}
    compared = []
    for row in result["variants"]["universal/raw"]["per_photo"]:
        expected = original[row["photo_id"]]
        if row["channel_dataset"] != expected["channel_datasets"]["disc"]:
            raise ValueError("Canary annotation source attribution differs")
        for key in ("tp", "fp", "fn", "valid_pixels"):
            if row["disc"][key] != expected["channels"]["disc"][key]:
                raise ValueError(f"Canary exact counts differ: {row['photo_id']} {key}")
        compared.append(row["photo_id"])
    if not compared:
        raise ValueError("Empty canary comparison")
    return {"exact_per_photo_counts_match": True, "photos": compared}


@torch.inference_mode()
def evaluate(manifest, checkpoints, profiles, *, limit=None, threads=4, progress=None):
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    manifest = Path(manifest)
    fingerprint = verify_prepared_content(manifest)
    dataset = UnifiedDataset(manifest, "val", UnifiedConfig())
    loaded, variants = {}, {}
    for name, path in checkpoints.items():
        model, payload = load_single_net(path, "cpu")
        channels = payload["channels"]
        if channels == ["disc"]:
            head = 0
        elif name == "universal" and channels == ["vessel", "disc", "cup"]:
            head = 1
        else:
            raise ValueError("Require original disc head or explicitly named universal canary")
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint parameters")
        loaded[name] = (model, payload, head)
        for profile in profiles:
            variants[f"{name}/{profile}"] = {
                "checkpoint_sha256": sha256(path),
                "checkpoint_path": str(path),
                "architecture": payload["architecture"],
                "tiling": payload["tiling"],
                "profile": profile,
                "head": head,
                "per_photo": [],
                "all_weights_and_logits_finite": True,
                "inference_seconds": 0.0,
            }
    skipped_no_task = skipped_invalid = processed = 0
    for index, rows in enumerate(dataset.groups):
        if not any(row["task"] == "disc_cup" for row in rows):
            skipped_no_task += 1
            continue  # No model forward or vascular-label decode for these photos.
        image, target, valid, trace = dataset.sample(
            index, random_annotator=False, return_trace=True
        )
        target, valid = target[3].numpy() >= 0.5, valid[3].numpy().copy()
        if not valid.any():
            skipped_invalid += 1
            continue
        if limit is not None and processed >= limit:
            break
        lookup = {row["id"]: row for row in rows}
        selected = [lookup[record_id] for record_id, task in trace if task == "disc_cup"]
        if len(selected) != 1:
            raise ValueError("Expected identical deterministic primary disc annotation")
        annotation_source = selected[0]["dataset"]
        for profile in profiles:
            prepared = image
            if profile == "normalized":
                # Deliberately identical to evaluate_original, including first-row FOV.
                with np.load(manifest.parent / rows[0]["mask"], allow_pickle=False) as label:
                    fov = label["fov"].astype(bool)
                rgb = (image.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                rgb = local_contrast_and_clahe(rgb, fov, pixels_per_degree=25)
                prepared = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
            for name, (model, payload, head) in loaded.items():
                before = time.perf_counter()
                logits = tiled_logits(
                    model,
                    prepared - 0.5,
                    device="cpu",
                    crop_size=payload["tiling"]["crop_size"],
                    border=payload["tiling"]["border"],
                )
                counts = disc_counts(logits[head].numpy(), target, valid)
                variant = variants[f"{name}/{profile}"]
                variant["inference_seconds"] += time.perf_counter() - before
                variant["per_photo"].append(
                    {
                        "photo_id": dataset.photo_ids[index],
                        "dataset": rows[0]["dataset"],
                        "channel_dataset": annotation_source,
                        "record_ids": [row["id"] for row in rows],
                        "selected_disc_record_id": selected[0]["id"],
                        "disc": _metrics(counts),
                    }
                )
        processed += 1
        if processed % 25 == 0 and progress is not None:
            snapshot = {
                "processed_disc_photos": processed,
                "elapsed_seconds": time.perf_counter() - started,
                "variants": {
                    k: {
                        "inference_seconds": v["inference_seconds"],
                        "last_photo": v["per_photo"][-1],
                    }
                    for k, v in variants.items()
                },
            }
            temporary = progress.with_suffix(progress.suffix + ".partial")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
            temporary.replace(progress)
            print(json.dumps({"stage": "legacy_disc_validation", "photos": processed}), flush=True)
    for variant in variants.values():
        total, grouped = np.zeros(4, np.int64), {}
        for row in variant["per_photo"]:
            counts = np.array(
                [row["disc"][k] for k in ("tp", "fp", "fn", "valid_pixels")], np.int64
            )
            total += counts
            grouped.setdefault(row["channel_dataset"], np.zeros(4, np.int64))
            grouped[row["channel_dataset"]] += counts
        variant["disc"] = _metrics(total)
        variant["by_dataset"] = {k: _metrics(v) for k, v in grouped.items()}
        scores = [v["dice"] for v in variant["by_dataset"].values() if v["dice"] is not None]
        variant["macro_dataset_disc_dice"] = float(np.mean(scores)) if scores else None
    return {
        "schema_version": 1,
        "split": "val",
        "device": "cpu",
        "threads": threads,
        "manifest_sha256": sha256(manifest),
        "prepared_content_sha256": fingerprint,
        "profiles": profiles,
        "validation_photos": len(dataset),
        "disc_photos": processed,
        "skipped_no_disc_task": skipped_no_task,
        "skipped_no_valid_disc": skipped_invalid,
        "preview_only": limit is not None,
        "variants": variants,
        "seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "script_sha256": sha256(__file__),
        "source_commit": "0d6d9087a92f0f07cf75fd1083c590e8af6bb18a",
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "policy": "Same UnifiedDataset val primary GT/FOV support and source attribution as evaluate_original; independent strictlogit>0 head only; no hierarchy, no vascular forward; no test evaluation",
        "independent_quality_claim": False,
        "limitations": [
            "Original photograph training exposure unresolved",
            "Original training raw/normalized profile unresolved; variants selected on val only",
            "Common prepared geometry, not publication-coordinate metrics",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, help="NAME=PATH; repeat for disc0/disc1"
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=["raw", "normalized"], default=["raw", "normalized"]
    )
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-report", type=Path, help="Exact universal/raw canary counts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.output.exists():
        parser.error("Preserve earlier evidence; choose a fresh output")
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        parser.error("Checkpoint names must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        args.manifest,
        checkpoints,
        args.profiles,
        limit=args.limit,
        threads=args.threads,
        progress=args.output.with_suffix(".progress.json"),
    )
    if args.reference_report:
        result["canary"] = compare_reference(result, json.loads(args.reference_report.read_text()))
        result["canary"]["reference_report_sha256"] = sha256(args.reference_report)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v["disc"] for k, v in result["variants"].items()}), flush=True)


if __name__ == "__main__":
    main()
