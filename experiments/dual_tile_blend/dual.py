"""One isolated experimental tile-blend contrast. Not imported by release inference.

Caller must separately authorize and bind the checked original checkpoint/data.
This module never opens a checkpoint, dataset path, image, GT or probability cache.
"""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = "521b8881d1b2c69a7395be6687a8b86581838e3f5dbb004a150333d454132cea"
REFERENCE = "0a5c65e6ecd0ceef68bba554504f0d2e2222905d5ebf27813b07242cb3e7aa0f"
UNIFIED_SHA = "ff862e9e16727c0a2910c598f77d7db9f4d61c50e8dbabd6665660a8fd70fc59"
FIELDS = (
    "channels",
    "by_dataset",
    "per_photo",
    "macro_dataset_mean_dice",
    "av_conditional_accuracy",
    "av_accuracy",
)
DESCRIPTOR = {
    "kind": "experimental_dual_spatial_tile_blend_NOT_gate_evidence",
    "version": 2,
    "streams": [
        "probability_control",
        "spatial_logit_alternative",
        "segmentation_logit_control_AV",
    ],
    "hybrid": "final V/D/C bit-identical spatial_logit_alternative; final conditional A/V bit-identical probability_control AFTER rot4",
    "tile_size": 512,
    "overlap": 64,
    "threshold": 0.35,
    "tta": "rot4",
    "precision": "float32_no_autocast",
    "window": "existing clipped Hann FP32, min0.05",
    "control_tile_blend": "FP32 weighted joint probabilities then conditional recovery",
    "alternative_tile_blend": "FP32 weighted raw5 logits then SAME joint activation and conditional recovery",
    "view_blend": "both streams: inverse-aligned FP32 JOINT probabilities; never view logits or unweighted conditionals",
    "shared_model_calls": True,
    "full_probability_cache": False,
}


def require_source(u):
    if hashlib.sha256(Path(u.__file__).read_bytes()).hexdigest() != UNIFIED_SHA:
        raise ValueError("Exact frozen6d unified source required")


def dual_probabilities(model, image, device, config, *, u):
    """Return three five-channel conditional views from identical shared tile calls."""
    require_source(u)
    cfg = asdict(config) if is_dataclass(config) else dict(config)
    if cfg.get("train_view", "patch") != "patch":
        raise ValueError("Fixed tiled patch experiment only")
    if (
        image.ndim != 3
        or image.shape[0] != 3
        or image.dtype != torch.float32
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("Full prepared RGB01 FP32 required")
    totals = {"probability_control": None, "spatial_logit_alternative": None}
    calls = 0
    with (
        torch.inference_mode(),
        torch.autocast(device_type=torch.device(device).type, enabled=False),
    ):
        for rotation, flip in u.TTA_OPERATORS["rot4"]:
            view = u._tta_transform(image, rotation, flip)
            model.eval()
            h, w = view.shape[-2:]
            result = np.zeros((5, h, w), np.float32)
            logit_sum = np.zeros_like(result)
            weights = np.zeros((h, w), np.float32)
            for y in u.tile_starts(h, 512, 64):
                for x in u.tile_starts(w, 512, 64):
                    tile = view[:, y : y + 512, x : x + 512]
                    logits = model(tile[None].to(device) - 0.5)
                    calls += 1
                    if (
                        logits.dtype != torch.float32
                        or logits.shape[:2] != (1, 5)
                        or not torch.isfinite(logits).all()
                    ):
                        raise ValueError("Finite five-channel FP32 tile logits required")
                    # Keep the frozen control activation/window/addition arithmetic.
                    probability = u._activation(logits)[0].cpu().numpy()
                    th, tw = probability.shape[-2:]
                    if (th, tw) != tuple(tile.shape[-2:]):
                        raise ValueError("Tile output shape differs")
                    wy = np.maximum(np.hanning(th), 0.05) if th > 1 else np.ones(1)
                    wx = np.maximum(np.hanning(tw), 0.05) if tw > 1 else np.ones(1)
                    weight = (wy[:, None] * wx[None]).astype(np.float32)
                    result[:, y : y + th, x : x + tw] += probability * weight
                    weights[y : y + th, x : x + tw] += weight
                    raw = logits[0].cpu().numpy()
                    logit_sum[:, y : y + th, x : x + tw] += raw * weight
                    del logits, probability, raw
            control = u._conditional_probabilities(result / weights[None])
            aligned = u._tta_inverse(u._joint_probabilities(control), rotation, flip)
            if totals["probability_control"] is None:
                totals["probability_control"] = aligned.copy()
            else:
                totals["probability_control"] += aligned
            del result, control, aligned
            # Spatial raw-logit averaging ONLY; activation stays on original backend.
            logit_sum /= weights[None]
            mean_logits = torch.from_numpy(logit_sum[None]).to(device)
            alternative_joint = u._activation(mean_logits)[0].cpu().numpy()
            alternative = u._conditional_probabilities(alternative_joint)
            aligned = u._tta_inverse(u._joint_probabilities(alternative), rotation, flip)
            if totals["spatial_logit_alternative"] is None:
                totals["spatial_logit_alternative"] = aligned.copy()
            else:
                totals["spatial_logit_alternative"] += aligned
            del logit_sum, weights, mean_logits, alternative_joint, alternative, aligned
        for name, total in totals.items():
            total /= np.float32(len(u.TTA_OPERATORS["rot4"]))
            totals[name] = u._conditional_probabilities(total)
    hybrid = totals["spatial_logit_alternative"].copy()
    hybrid[1:3] = totals["probability_control"][1:3]
    totals["segmentation_logit_control_AV"] = hybrid
    return totals, {"model_calls": calls, "views": 4, "single_shared_model_call_per_tile": True}


def stream_counts(
    model, dataset, device, config, payload, reference_bytes, *, u, counters, progress=None
):
    """Under80 lines of scorer reuse; caller binds source/data/eligibility before entry.

    No disk output, probability cache, model loading or second dataset sample.
    Counts may be persisted by a separately reviewed bounded producer after return.
    """
    require_source(u)
    if hashlib.sha256(reference_bytes).hexdigest() != REFERENCE:
        raise ValueError("Exact original0a5c control reference bytes required")
    reference = json.loads(reference_bytes)
    if (
        hashlib.sha256(Path(counters.__file__).read_bytes()).hexdigest()
        != "7be1227ac7d41247f847c169aaf43a061b0c4ac60363a9346b506ca8495bf147"
        or counters.u is not u
    ):
        raise ValueError("Exact historical scorer and same frozen unified decoder required")
    if len(dataset.groups) != 770 or any(
        r["split"] != "val" for rows in dataset.groups for r in rows
    ):
        raise ValueError("Exactly full770 VAL metadata required before prediction")
    if payload.get("checkpoint_sha256") != CHECKPOINT or payload.get("epoch") != 50:
        raise ValueError("Checked original BEST50 snapshot required")
    if (
        reference.get("checkpoint", {}).get("sha256") != CHECKPOINT
        or reference.get("checkpoint", {}).get("epoch") != 50
    ):
        raise ValueError("Original control reference differs")
    expected = reference.get("metrics", reference)
    if expected.get("split") != "val" or len(expected.get("per_photo", [])) != 770:
        raise ValueError("Original full770 VAL reference required")
    if (
        reference.get("inference_protocol", {}).get("threshold") != 0.35
        or reference["inference_protocol"].get("tta", {}).get("mode") != "rot4"
    ):
        raise ValueError("Control rot4/.35 protocol differs")
    photos = {
        "probability_control": [],
        "spatial_logit_alternative": [],
        "segmentation_logit_control_AV": [],
    }
    calls = 0
    supported = getattr(model, "supervised_channels", [True] * 5)
    for index, rows in enumerate(dataset.groups):
        if any(r["split"] != "val" for r in rows):
            raise ValueError("VAL only; no TEST route")
        image, target, valid, trace = dataset.sample(
            index, random_annotator=False, return_trace=True
        )
        predictions, observed = dual_probabilities(model, image, device, config, u=u)
        calls += observed["model_calls"]
        for name, probability in predictions.items():
            photos[name].append(
                counters.photo_counts(
                    probability,
                    target,
                    valid,
                    rows,
                    trace,
                    dataset.photo_ids[index],
                    supported,
                    0.35,
                )
            )
        del predictions, image, target, valid
        if progress is not None:
            progress({"photos": index + 1, "model_calls": calls})
    reports = {}
    for name, counts in photos.items():
        result = counters.aggregate(counts, 0.35)
        # The reused historical aggregate has an unrelated constant checkpoint tag;
        # explicitly bind the checked actual artifact in this NEW experimental kind.
        result.update(
            kind=DESCRIPTOR["kind"],
            checkpoint_sha256=CHECKPOINT,
            checkpoint_epoch=50,
            ordinary_freeze_evidence=False,
            math_descriptor=DESCRIPTOR,
            branch=name,
        )
        reports[name] = result
    parity = {key: reports["probability_control"][key] == expected[key] for key in FIELDS}
    return {
        "kind": DESCRIPTOR["kind"],
        "reports": reports,
        "control_reference_sha256_required": REFERENCE,
        "control_fields_exact": parity,
        "control_parity_pass": all(parity.values()),
        "model_calls": calls,
        "dataset_samples": len(dataset.groups),
        "new_selection_or_frozen_protocol_change": False,
    }
