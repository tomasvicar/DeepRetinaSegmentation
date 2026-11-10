"""Refresh validation A/V with calibrated MATLAB-style Python preprocessing.

Non-A/V predictions are preserved from an existing full original-model report.
The dedicated vessel model is rerun only where A/V GT exists, and its confusion
counts must reproduce that report exactly before A/V preprocessing is changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from evaluate_original import confusion, geometry, raw_photo, sha256
from PIL import Image

from deepretina.integrity import verify_prepared_content
from deepretina.legacy.av import load_av, predict_av_input, restore_classes
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated
from deepretina.legacy.matlab_fov import calibrate_provided_fov
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import CHANNELS, UnifiedConfig, UnifiedDataset, _metrics


def summarize(reference, replacements):
    """Replace every eligible A/V photo, retaining all other heads exactly."""
    expected = {
        photo["photo_id"]
        for photo in reference["per_photo"]
        if photo["channels"]["artery"]["valid_pixels"] or photo["channels"]["vein"]["valid_pixels"]
    }
    lookup = {photo["photo_id"]: photo for photo in replacements}
    if len(lookup) != len(replacements) or set(lookup) != expected:
        raise ValueError("Refresh must contain each supported A/V photo exactly once")
    result = copy.deepcopy(reference)
    total = np.zeros((5, 4), np.int64)
    sources = {}
    correct = scored = eligible = end_to_end = 0
    for photo in result["per_photo"]:
        replacement = lookup.get(photo["photo_id"])
        if replacement is not None:
            for name in ("artery", "vein"):
                old, new = photo["channels"][name], replacement["channels"][name]
                if (
                    old["valid_pixels"] != new["valid_pixels"]
                    or old["tp"] + old["fn"] != new["tp"] + new["fn"]
                ):
                    raise ValueError("Refreshed A/V ground-truth support differs")
                photo["channels"][name] = copy.deepcopy(new)
            photo["av_calibration"] = replacement["calibration"]
            photo["av_conditional_counts"] = replacement["conditional_counts"]
            counts = replacement["conditional_counts"]
            correct += counts["correct"]
            scored += counts["scored_pixels"]
            eligible += counts["eligible_pixels"]
            end_to_end += counts["end_to_end_correct"]
        for channel, name in enumerate(CHANNELS):
            values = photo["channels"][name]
            row = [values[key] for key in ("tp", "fp", "fn", "valid_pixels")]
            total[channel] += row
            source = photo["channel_datasets"].get(name)
            if source is not None:
                sources.setdefault(source, np.zeros((5, 4), np.int64))[channel] += row
            elif any(row):
                raise ValueError("Missing source attribution for supported output")
    result["channels"] = {
        name: _metrics(value) for name, value in zip(CHANNELS, total, strict=True)
    }
    result["by_dataset"] = {
        source: {name: _metrics(value) for name, value in zip(CHANNELS, counts, strict=True)}
        for source, counts in sources.items()
    }
    macro = {}
    for name in CHANNELS:
        values = [
            metrics[name]["dice"]
            for metrics in result["by_dataset"].values()
            if metrics[name]["dice"] is not None
        ]
        macro[name] = float(np.mean(values)) if values else None
    result["macro_dataset_channel_dice"] = macro
    values = [value for value in macro.values() if value is not None]
    result["macro_dataset_mean_dice"] = float(np.mean(values)) if values else None
    result["av_conditional_accuracy"] = correct / scored if scored else None
    result["av_conditional_scored_pixels"] = scored
    result["av_end_to_end_accuracy"] = end_to_end / eligible if eligible else None
    result["av_labeled_pixels"] = eligible
    result["empty_vessel_photos"] = [
        photo["photo_id"]
        for photo in replacements
        if not photo["conditional_counts"]["scored_pixels"]
    ]
    return result


@torch.inference_mode()
def refresh(reference_path, manifest, data_root, vessel_path, av_path, *, limit=None, threads=8):
    started = time.monotonic()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    reference_bytes = Path(reference_path).read_bytes()
    baseline = json.loads(reference_bytes)
    if (
        baseline["split"] != "val"
        or baseline.get("preview_only")
        or baseline.get("single_net_normalized")
    ):
        raise ValueError("Requires a full raw-profile validation reference")
    if baseline["manifest_sha256"] != sha256(manifest):
        raise ValueError("Reference manifest identity differs")
    if baseline["vessel_checkpoint_sha256"] != sha256(vessel_path) or baseline[
        "av_checkpoint_sha256"
    ] != sha256(av_path):
        raise ValueError("Preserve exactly the same original vessel/A/V checkpoint weights")
    content = verify_prepared_content(Path(manifest))
    dataset = UnifiedDataset(Path(manifest), "val", UnifiedConfig(purpose="full_data"))
    previous = {photo["photo_id"]: photo for photo in baseline["per_photo"]}
    if len(previous) != len(baseline["per_photo"]) or set(previous) != set(dataset.photo_ids):
        raise ValueError("Full validation photograph identity differs")
    photos = {
        photo["photo_id"]: photo
        for photo in json.loads((Path(manifest).parent / "photos.json").read_text())
    }
    vessel_model, vessel_payload = load_single_net(vessel_path, "cpu")
    if vessel_payload["channels"] != ["vessel"]:
        raise ValueError("Expected the original dedicated vessel model")
    av_model = load_av(av_path, "cpu")[0]
    replacements = []
    sources = {}
    with ExitStack() as stack:
        for index, photo_id in enumerate(dataset.photo_ids):
            before = previous[photo_id]
            if not any(before["channels"][name]["valid_pixels"] for name in ("artery", "vein")):
                continue
            if limit is not None and len(replacements) >= limit:
                break
            image, target, valid = dataset.sample(index)
            target, valid = target.numpy() >= 0.5, valid.numpy()
            known = valid[1] & valid[2] & (target[1] ^ target[2])
            logits = tiled_logits(
                vessel_model,
                image - 0.5,
                device="cpu",
                crop_size=vessel_payload["tiling"]["crop_size"],
                border=vessel_payload["tiling"]["border"],
            )
            masks = np.zeros_like(target)
            masks[0] = logits.numpy()[0] > 0
            vessel_counts = confusion(masks, target, valid)[0]
            expected_counts = [
                before["channels"]["vessel"][key] for key in ("tp", "fp", "fn", "valid_pixels")
            ]
            if not np.array_equal(vessel_counts, expected_counts):
                raise ValueError(f"Original vessel counts do not reproduce exactly: {photo_id}")
            photo = photos[photo_id]
            transform = geometry(photo)
            rgb = raw_photo(photo, Path(data_root), sources, stack)
            vessel = transform.restore(masks[0], order=0)
            with Image.open(Path(manifest).parent / photo["fov"]) as handle:
                fov = transform.restore(np.asarray(handle) > 0, order=0)
            calibration = calibrate_provided_fov(fov, dataset=photo["canonical_dataset"])
            conditional_counts = {
                "correct": 0,
                "scored_pixels": 0,
                "eligible_pixels": int(known.sum()),
                "end_to_end_correct": 0,
            }
            if vessel.any() and not vessel.all():
                prepared = prepare_vesclass_matlab_calibrated(
                    rgb,
                    vessel,
                    fov=fov,
                    fov_diameter_pixels=calibration.diameter_pixels,
                    dataset=photo["canonical_dataset"],
                    field_of_view_degrees=photo["field_of_view_degrees"],
                )
                prediction = predict_av_input(
                    av_model, prepared.inputs(), device="cpu", original_size=rgb.shape[:2]
                )
                classes = transform.prepare_mask(prediction["classes"])
                masks[1], masks[2] = classes == 1, classes == 2
                z = prediction["prepared_logits"]
                conditional = np.zeros(z.shape[-2:], np.uint8)
                conditional[z[1] > z[2]] = 1
                conditional[z[2] > z[1]] = 2
                conditional = transform.prepare_mask(restore_classes(conditional, rgb.shape[:2]))
                correct = (conditional == 1) & target[1] | (conditional == 2) & target[2]
                conditional_counts.update(
                    correct=int((correct & known).sum()),
                    scored_pixels=int(known.sum()),
                    end_to_end_correct=int(
                        ((masks[1] & target[1] | masks[2] & target[2]) & known).sum()
                    ),
                )
            values = confusion(masks, target, valid)
            replacements.append(
                {
                    "photo_id": photo_id,
                    "channels": {
                        name: _metrics(values[channel])
                        for channel, name in ((1, "artery"), (2, "vein"))
                    },
                    "conditional_counts": conditional_counts,
                    "calibration": dict(
                        asdict(calibration),
                        diameter_pixels=calibration.diameter_pixels,
                        raw_fov_sha256=hashlib.sha256(fov.tobytes()).hexdigest(),
                        source="canonical legacy25 FOV restored to original frame; generic souradnice Python estimator",
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "calibrated_original_av",
                        "photos": len(replacements),
                        "photo_id": photo_id,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if limit is not None:
        result = {
            "schema_version": 1,
            "preview_only": True,
            "split": "val",
            "calibrated_photo_results": replacements,
        }
    else:
        result = summarize(baseline, replacements)
    result.update(
        baseline_report_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        manifest_sha256=sha256(manifest),
        prepared_content_sha256=content,
        av_preprocessing_profile="vesclass_matlab_provided_fov_calibrated",
        av_refresh_photos=len(replacements),
        refresh_seconds=time.monotonic() - started,
        refresh_threads=threads,
        refresh_device="cpu",
        refresh_script_sha256=sha256(__file__),
        independent_quality_improvement_claim=False,
        refresh_qualifications=[
            "Non-AV predictions preserved; original vessel confusion counts reproduced exactly on every refreshed photo",
            "Provided FOV is restored from canonical preparation, not recovered historical native FOV",
            "Generic calibration matches MATLAB oracle; raw photometry retains documented numerical residuals",
            "Historical training exposure and file preprocessing provenance remain unresolved",
            "Validation only; this does not establish independent test improvement",
        ],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "manifest", "data-root", "vessel", "av", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Choose a fresh output, positive thread count and positive optional limit")
    result = refresh(
        args.reference,
        args.manifest,
        args.data_root,
        args.vessel,
        args.av,
        limit=args.limit,
        threads=args.threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
