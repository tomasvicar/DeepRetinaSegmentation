"""Evaluate original models on the unified manifest's identical prepared GT support.

This is an exposure-qualified reference comparison. Historical split mapping and
raw/normalized training profile are unresolved. Optional raw A/V inference is an
explicitly uncertified Python proxy, not a reproduction of MATLAB quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deepretina.inference import resolve_device
from deepretina.legacy.av import load_av, predict_av_input, restore_classes
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.preprocessing import (
    ImageGeometry,
    local_contrast_and_clahe,
    prepare_vesclass_raw_proxy,
)
from deepretina.legacy.single_net import tiled_logits
from deepretina.prepare import ARCHIVES
from deepretina.sources import Source
from deepretina.unified import CHANNELS, SOURCE_CHANNELS, UnifiedConfig, UnifiedDataset, _metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confusion(masks, target, valid):
    """Unknown targets contribute neither positive nor negative evidence."""
    masks, target, valid = (np.asarray(a, dtype=bool) for a in (masks, target, valid))
    if masks.shape != target.shape or target.shape != valid.shape or masks.shape[0] != 5:
        raise ValueError("Expected matching5channel prediction/target/valid arrays")
    return np.asarray(
        [
            [(p & t & v).sum(), (p & ~t & v).sum(), (~p & t & v).sum(), v.sum()]
            for p, t, v in zip(masks, target, valid, strict=True)
        ],
        dtype=np.int64,
    )


def geometry(photo):
    value = photo["geometry"]
    return ImageGeometry(
        tuple(value["original_shape"]),
        tuple(value["crop_position"]),
        value["rescale_factor"],
        tuple(value["prepared_shape"]),
    )


def raw_photo(photo, data_root, sources, stack):
    dataset = photo["canonical_dataset"]
    if dataset not in sources:
        sources[dataset] = stack.enter_context(Source(data_root / ARCHIVES[dataset]))
    source = sources[dataset]
    resource = photo["sources"][0]["resource"]
    image = source.read(source.find(resource), "RGB")
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    if digest.hexdigest() != photo["raw_rgb_sha256"]:
        raise ValueError(f"Original raw photo identity differs: {photo['photo_id']}")
    return image


def segmentation_masks(image, universal, device, *, vessel=None, disc=None):
    """Keep original independent heads, with optional task-specific originals."""
    result = np.zeros((5, *image.shape[-2:]), dtype=bool)
    for item, destination in ((universal, [0, 3, 4]), (vessel, [0]), (disc, [3])):
        if item is None:
            continue
        model, payload = item
        logits = tiled_logits(
            model,
            image - 0.5,
            device=device,
            crop_size=payload["tiling"]["crop_size"],
            border=payload["tiling"]["border"],
        )
        result[destination] = logits.numpy() > 0
    return result


@torch.inference_mode()
def evaluate(
    manifest,
    universal_path,
    *,
    split="val",
    device="auto",
    limit=None,
    av_path=None,
    data_root=None,
    normalized=False,
    vessel_path=None,
    disc_path=None,
):
    manifest = Path(manifest)
    device = resolve_device(device)
    dataset = UnifiedDataset(manifest, split, UnifiedConfig())
    model, payload = load_single_net(universal_path, device)
    if payload["channels"] != ["vessel", "disc", "cup"]:
        raise ValueError("Expected original universal vessel/disc/cup model")
    vessel = load_single_net(vessel_path, device) if vessel_path else None
    disc = load_single_net(disc_path, device) if disc_path else None
    for override, expected in ((vessel, ["vessel"]), (disc, ["disc"])):
        if override is not None and override[1]["channels"] != expected:
            raise ValueError(f"Original override checkpoint must have channels {expected}")
    if av_path and data_root is None:
        raise ValueError("Raw A/V proxy requires original data root")
    av = load_av(av_path, device)[0] if av_path else None
    photos = {}
    if av is not None:
        photos = {
            p["photo_id"]: p for p in json.loads((manifest.parent / "photos.json").read_text())
        }
    total = np.zeros((5, 4), np.int64)
    grouped = {}
    results = []
    conditional_correct = conditional_pixels = end_to_end_correct = 0
    sources = {}
    empty_vessel_photos = []
    with ExitStack() as stack:
        for index, rows in enumerate(dataset.groups):
            if limit is not None and index >= limit:
                break
            image, target, valid, trace = dataset.sample(
                index, random_annotator=False, return_trace=True
            )
            target, valid = target.numpy() >= 0.5, valid.numpy().copy()
            if normalized:
                with np.load(manifest.parent / rows[0]["mask"], allow_pickle=False) as label:
                    fov = label["fov"].astype(bool)
                rgb = (image.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                rgb = local_contrast_and_clahe(rgb, fov, pixels_per_degree=25)
                image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
            masks = segmentation_masks(image, (model, payload), device, vessel=vessel, disc=disc)
            photo_id = dataset.photo_ids[index]
            known = valid[1] & valid[2] & (target[1] ^ target[2])
            if av is None:
                valid[1:3] = False
            elif known.any():
                photo = photos[photo_id]
                transform = geometry(photo)
                rgb = raw_photo(photo, Path(data_root), sources, stack)
                vessel_mask = transform.restore(masks[0], order=0)
                if vessel_mask.any() and not vessel_mask.all():
                    with Image.open(manifest.parent / photo["fov"]) as im:
                        fov = transform.restore(np.asarray(im) > 0, order=0)
                    proxy = prepare_vesclass_raw_proxy(
                        rgb,
                        vessel_mask,
                        dataset=photo["canonical_dataset"],
                        fov=fov,
                        field_of_view_degrees=photo["field_of_view_degrees"],
                    )
                    output = predict_av_input(
                        av, proxy.inputs(), device=device, original_size=rgb.shape[:2]
                    )
                    classes = transform.prepare_mask(output["classes"])
                    masks[1], masks[2] = classes == 1, classes == 2
                    z = output["prepared_logits"]
                    conditional = np.zeros(z.shape[-2:], dtype=np.uint8)
                    conditional[z[1] > z[2]] = 1
                    conditional[z[2] > z[1]] = 2
                    conditional = transform.prepare_mask(
                        restore_classes(conditional, rgb.shape[:2])
                    )
                    correct = (conditional == 1) & target[1] | (conditional == 2) & target[2]
                    conditional_correct += int((correct & known).sum())
                    conditional_pixels += int(known.sum())
                else:
                    # Original global-standard-deviation division is undefined.
                    # The usable pipeline emits no AV detections, counted as misses.
                    empty_vessel_photos.append(photo_id)
                end_to_end_correct += int(
                    ((masks[1] & target[1] | masks[2] & target[2]) & known).sum()
                )
            value = confusion(masks, target, valid)
            total += value
            name = rows[0]["dataset"]
            channel_datasets = {}
            lookup = {row["id"]: row for row in rows}
            for record_id, task in trace:
                annotation_source = lookup[record_id]["dataset"]
                grouped.setdefault(annotation_source, np.zeros_like(total))
                for channel in SOURCE_CHANNELS[task]:
                    grouped[annotation_source][channel] += value[channel]
                    channel_datasets[CHANNELS[channel]] = annotation_source
            results.append(
                {
                    "photo_id": photo_id,
                    "dataset": name,
                    "channel_datasets": channel_datasets,
                    "record_ids": [r["id"] for r in rows],
                    "channels": {c: _metrics(v) for c, v in zip(CHANNELS, value, strict=True)},
                }
            )
            if len(results) % 25 == 0:
                print(
                    json.dumps({"stage": "original_validation", "photos": len(results)}), flush=True
                )
    known_pixels = int(total[1, 3])
    dataset_metrics = {
        d: {c: _metrics(v) for c, v in zip(CHANNELS, a, strict=True)} for d, a in grouped.items()
    }
    macro_channels = {}
    for channel in CHANNELS:
        available = [
            m[channel]["dice"] for m in dataset_metrics.values() if m[channel]["dice"] is not None
        ]
        macro_channels[channel] = float(np.mean(available)) if available else None
    macro_scores = [v for v in macro_channels.values() if v is not None]
    return {
        "schema_version": 1,
        "family": "original_reference",
        "split": split,
        "independent_quality_improvement_claim": False,
        "comparison_scope": "exposure-qualified reference; identical canonical legacy25 prepared GT support",
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(universal_path),
        "av_checkpoint_sha256": sha256(av_path) if av_path else None,
        "vessel_checkpoint_sha256": sha256(vessel_path) if vessel_path else None,
        "disc_checkpoint_sha256": sha256(disc_path) if disc_path else None,
        "single_net_normalized": normalized,
        "photos": len(results),
        "preview_only": limit is not None
        or any(r.get("preparation_preview") for r in dataset.rows),
        "channels": {c: _metrics(v) for c, v in zip(CHANNELS, total, strict=True)},
        "by_dataset": dataset_metrics,
        "macro_dataset_channel_dice": macro_channels,
        "macro_dataset_mean_dice": float(np.mean(macro_scores)) if macro_scores else None,
        "av_conditional_accuracy": conditional_correct / conditional_pixels
        if conditional_pixels
        else None,
        "av_conditional_scored_pixels": conditional_pixels,
        "av_end_to_end_accuracy": end_to_end_correct / known_pixels if known_pixels else None,
        "av_labeled_pixels": known_pixels,
        "empty_vessel_photos": empty_vessel_photos,
        "per_photo": results,
        "policy": "one photo once; first sorted GT variant per task; AV vessel GT priority; unknown labels ignored; original strictlogit>0 and tie0 retained; no addedcup containment",
        "limitations": [
            "Original photograph-to-training-index crosswalk unavailable; historical exposure unresolved",
            "Original raw/normalized training profile uncertain; selected profile is recorded explicitly",
            "AV raw Python preprocessing is not MATLAB-parity certified",
            "AV proxy uses canonical FOV restored to raw frame; native historical FOV unavailable",
            "Scores are on common prepared geometry, not original-coordinate publication scores",
            "Conditional AV excludes undefined empty/full vessel inputs; end-to-end counts them as misses",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--universal", type=Path, required=True)
    parser.add_argument("--av-raw-proxy", type=Path)
    parser.add_argument("--vessel", type=Path, help="Optional original dedicated vessel checkpoint")
    parser.add_argument("--disc", type=Path, help="Optional original dedicated disc checkpoint")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Original Python local-contrast single_net profile",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--selection-record",
        type=Path,
        help="Required for test: frozen config/selection evidence; hash is recorded",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("Preserve earlier results; choose a new output")
    if args.split == "test" and args.selection_record is None:
        parser.error("Test requires --selection-record; select settings on validation first")
    result = evaluate(
        args.manifest,
        args.universal,
        split=args.split,
        device=args.device,
        limit=args.limit,
        av_path=args.av_raw_proxy,
        data_root=args.data_root,
        normalized=args.normalized,
        vessel_path=args.vessel,
        disc_path=args.disc,
    )
    result["selection_record_sha256"] = (
        sha256(args.selection_record) if args.selection_record else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["channels"]))


if __name__ == "__main__":
    main()
