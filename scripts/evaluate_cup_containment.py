"""Optional VAL-only cup∩predicted-disc study; original routes remain untouched."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from compose_original_disc_reference import aggregate, metrics
from evaluate_legacy_disc import disc_counts

from deepretina.integrity import verify_prepared_content
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.single_net import tile_corners, tiled_logits
from deepretina.unified import UnifiedConfig, UnifiedDataset

KEYS = ("tp", "fp", "fn", "valid_pixels")
MODELS = ("universal", "disc0", "disc1")
REF_SHA = {
    "disc0": "2ce2a1891fac58999e903bb50b84a3378572d9ebea6ad0e430f55a535f8f0792",
    "disc1": "e8b0c48437b8b3c341228b57cf83e5567d5263b15bb86e4570b3d42fb5d99091",
}
CANONICAL = "61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counted(prediction, target, valid):
    values = disc_counts(prediction, target, valid)
    return metrics(dict(zip(KEYS, map(int, values), strict=True)))


def exact(actual, expected):
    if any(actual[key] != expected[key] for key in KEYS):
        raise ValueError(f"Native per-photo counts differ: {actual} vs {expected}")


def supported_forward(valid_cup, callback):
    return callback() if np.asarray(valid_cup, bool).any() else None


def cup_variants(cup_logits, disc0_logits, disc1_logits, target, valid):
    arrays = [np.asarray(value) for value in (cup_logits, disc0_logits, disc1_logits)]
    if any(value.shape != np.shape(target) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Invalid predicted cup/disc geometry or logits")
    cup, d0, d1 = [value > 0 for value in arrays]
    return {
        "native": counted(cup, target, valid),
        "disc0": counted(cup & d0, target, valid),
        "disc1": counted(cup & d1, target, valid),
    }


def compose_cup(base, replacements, variant):
    if base["split"] != "val" or base.get("preview_only", False):
        raise ValueError("Complete VAL reference required")
    expected = {
        row["photo_id"] for row in base["per_photo"] if row["channels"]["cup"]["valid_pixels"]
    }
    if set(replacements) != expected:
        raise ValueError("Require exactly all cup-supported photos")
    result = copy.deepcopy(base)
    for row in result["per_photo"]:
        if row["photo_id"] not in replacements:
            continue
        before, after = row["channels"]["cup"], replacements[row["photo_id"]]
        if (
            before["valid_pixels"] != after["valid_pixels"]
            or before["tp"] + before["fn"] != after["tp"] + after["fn"]
        ):
            raise ValueError("Cup GT support or positives changed")
        if after["tp"] > before["tp"] or after["fp"] > before["fp"] or after["fn"] < before["fn"]:
            raise ValueError("Intersection cannot invent positive predictions")
        row["channels"]["cup"] = metrics(after)
    for before, after in zip(base["per_photo"], result["per_photo"], strict=True):
        for channel in ("vessel", "artery", "vein", "disc"):
            if before["channels"][channel] != after["channels"][channel]:
                raise ValueError("Non-cup head changed")
        for key in ("photo_id", "dataset", "record_ids", "channel_datasets"):
            if before[key] != after[key]:
                raise ValueError("Photo/GT metadata changed")
    pooled, grouped, macro, score = aggregate(result["per_photo"], base["by_dataset"])
    result.update(
        channels=pooled,
        by_dataset=grouped,
        macro_dataset_channel_dice=macro,
        macro_dataset_mean_dice=score,
    )
    result["optional_cup_postprocessing_experiment"] = {
        "variant": variant,
        "policy": "native universal/raw cup>0 intersect native dedicated raw disc>0; predicted masks only",
        "original_defaults_or_weights_changed": False,
        "test_evaluation": False,
        "new_independent_quality_claim": False,
    }
    return result


def choose_canary(ids, lookup):
    selected = []
    for source in ("refuge", "g1020", "rimone", "drishti"):
        values = [pid for pid in ids if lookup[pid]["channel_datasets"]["cup"] == source]
        count = 2 if source == "refuge" else 1
        positions = [len(values) // 3, 2 * len(values) // 3] if count == 2 else [len(values) // 2]
        selected.extend(values[position] for position in positions)
    if len(set(selected)) != 5:
        raise ValueError("Need five representative cup VAL photos")
    return selected


def tiles(shape, payload):
    crop, border = payload["tiling"]["crop_size"], payload["tiling"]["border"]
    return len(tile_corners(max(shape[0], crop), max(shape[1], crop), crop, border))


@torch.inference_mode()
def evaluate(manifest, reference_dir, output, mode):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    manifest, reference_dir, output = Path(manifest), Path(reference_dir), Path(output)
    if output.exists():
        raise FileExistsError("Preserve previous experiment output")
    output.mkdir(parents=True)
    if verify_prepared_content(manifest, CANONICAL) != CANONICAL:
        raise ValueError("Canonical pin differs")
    refs = {}
    for name in ("disc0", "disc1"):
        path = reference_dir / f"original-calibrated-av-{name}-raw-val.json"
        if sha(path) != REF_SHA[name]:
            raise ValueError("Require exact strong reference from84f899b")
        refs[name] = json.loads(path.read_text())
    lookup = {
        name: {row["photo_id"]: row for row in ref["per_photo"]} for name, ref in refs.items()
    }
    for pid, row in lookup["disc0"].items():
        exact(row["channels"]["cup"], lookup["disc1"][pid]["channels"]["cup"])
    dataset = UnifiedDataset(manifest, "val", UnifiedConfig())
    if set(dataset.photo_ids) != set(lookup["disc0"]) or len(dataset) != 770:
        raise ValueError("VAL photo inventory differs")
    indices = {pid: i for i, pid in enumerate(dataset.photo_ids)}
    eligible = [
        pid for pid in dataset.photo_ids if lookup["disc0"][pid]["channels"]["cup"]["valid_pixels"]
    ]
    if len(eligible) != 612:
        raise ValueError("Cup-supported inventory differs")
    selected = choose_canary(eligible, lookup["disc0"]) if mode == "canary" else eligible
    model_root = Path(__file__).parents[1] / "src/deepretina/pretrained"
    paths = {
        "universal": model_root / "legacy-single-net-universal.pt",
        "disc0": model_root / "legacy-single-net-disc-0.pt",
        "disc1": model_root / "legacy-single-net-disc-1.pt",
    }
    models, payloads, model_info = {}, {}, {}
    for name, path in paths.items():
        model, payload = load_single_net(path, "cpu")
        expected = (
            refs["disc0"]["checkpoint_sha256"]
            if name == "universal"
            else refs[name]["disc_checkpoint_sha256"]
        )
        if sha(path) != expected:
            raise ValueError("Wrong original weights")
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Nonfinite weights")
        models[name], payloads[name] = model, payload
        model_info[name] = {
            "checkpoint_sha256": expected,
            "architecture": payload["architecture"],
            "tiling": payload["tiling"],
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    photos = {p["photo_id"]: p for p in json.loads((manifest.parent / "photos.json").read_text())}
    full_tiles = {
        name: sum(tiles(photos[pid]["prepared_shape"][:2], payloads[name]) for pid in eligible)
        for name in MODELS
    }
    observed_tiles = {name: 0 for name in MODELS}
    seconds = {name: 0.0 for name in MODELS}
    data_seconds = 0.0
    rows = []
    with (output / "per-photo.jsonl").open("x") as stream:
        for pid in selected:
            before = time.perf_counter()
            index = indices[pid]
            records = dataset.groups[index]
            image, target, valid, trace = dataset.sample(
                index, random_annotator=False, return_trace=True
            )
            cup, vc = target[4].numpy() >= 0.5, valid[4].numpy().copy()
            disc, vd = target[3].numpy() >= 0.5, valid[3].numpy().copy()
            data_seconds += time.perf_counter() - before
            if not vc.any():
                raise ValueError("Saved reference claims absent cup support")
            if [row["id"] for row in records] != lookup["disc0"][pid]["record_ids"]:
                raise ValueError("Primary annotation inventory differs")
            predicted = {}

            def forward_all(image=image, predicted=predicted):
                for name in MODELS:
                    before = time.perf_counter()
                    p = payloads[name]
                    logits = tiled_logits(
                        models[name],
                        image - 0.5,
                        "cpu",
                        p["tiling"]["crop_size"],
                        p["tiling"]["border"],
                    )
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite original logits")
                    predicted[name] = logits[2 if name == "universal" else 0].numpy().copy()
                    seconds[name] += time.perf_counter() - before
                    observed_tiles[name] += tiles(image.shape[-2:], p)

            supported_forward(vc, forward_all)
            native_cup = counted(predicted["universal"], cup, vc)
            exact(native_cup, lookup["disc0"][pid]["channels"]["cup"])
            exact(native_cup, lookup["disc1"][pid]["channels"]["cup"])
            dcounts = {name: counted(predicted[name], disc, vd) for name in ("disc0", "disc1")}
            for name in ("disc0", "disc1"):
                exact(dcounts[name], lookup[name][pid]["channels"]["disc"])
            variants = cup_variants(
                predicted["universal"], predicted["disc0"], predicted["disc1"], cup, vc
            )
            row = {
                "photo_id": pid,
                "source": lookup["disc0"][pid]["channel_datasets"]["cup"],
                "record_ids": [r["id"] for r in records],
                "selected_trace": trace,
                "shape": list(image.shape[-2:]),
                "cup": variants,
                "native_disc": dcounts,
                "native_counts_exact": True,
            }
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 10 == 0 or len(rows) == len(selected):
                print(
                    json.dumps(
                        {
                            "processed": len(rows),
                            "total": len(selected),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    projected = sum(
        seconds[name] / observed_tiles[name] * full_tiles[name] for name in MODELS
    ) + data_seconds / len(rows) * len(eligible)
    report = {
        "schema_version": 1,
        "split": "val",
        "mode": mode,
        "preview_only": mode != "full",
        "photos": len(dataset),
        "cup_supported_photos": len(eligible),
        "processed": len(rows),
        "skipped_without_cup_supervision": len(dataset) - len(eligible),
        "native_per_photo_counts_exact": True,
        "models": model_info,
        "inference_seconds": seconds,
        "data_seconds": data_seconds,
        "observed_tiles": observed_tiles,
        "full_tiles": full_tiles,
        "projected_full_seconds": projected,
        "projected_full_with_15pct_margin_seconds": projected * 1.15,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest_sha256": sha(manifest),
        "canonical_fingerprint": CANONICAL,
        "reference_sha256": REF_SHA,
        "script_sha256": sha(__file__),
        "threads": 2,
        "interop_threads": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "original_defaults_or_weights_changed": False,
        "postprocessing_gt_used": False,
        "no_gpu": True,
        "test_evaluation": False,
        "per_photo": rows,
    }
    if mode == "full":
        for name in ("disc0", "disc1"):
            composed = compose_cup(
                refs[name], {row["photo_id"]: row["cup"][name] for row in rows}, name + "/raw"
            )
            composed["cup_postprocess_runtime"] = {
                key: report[key] for key in report if key != "per_photo"
            }
            (output / f"cup-and-{name}-raw-val.json").write_text(
                json.dumps(composed, indent=2, allow_nan=False) + "\n"
            )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["canary", "full"], required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.reference_dir, args.output, args.mode)
    print(json.dumps({k: v for k, v in report.items() if k != "per_photo"}), flush=True)


if __name__ == "__main__":
    main()
