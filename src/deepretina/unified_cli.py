"""Public routes for experimental unified checkpoints; training stays in unified.py."""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from contextlib import redirect_stdout
from dataclasses import fields

import numpy as np
import torch
from PIL import Image

from .integrity import verify_prepared_content
from .unified import (
    UnifiedConfig,
    evaluate_unified_model,
    inference_config,
    load_unified,
    predict_unified,
    train_unified,
    tta_protocol,
)


def train_from_args(args):
    settings = {}
    content_pin = args.expected_content_sha256
    cache_pin = args.expected_cache_fingerprint
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("family") != "unified" or "training_state" not in saved:
            raise ValueError("--resume requires a unified last.pt with training state")
        settings.update(saved["config"])
        if content_pin is None:
            content_pin = saved.get("prepared_content_sha256")
        if cache_pin is None and saved.get("derived_cache"):
            cache_pin = saved["derived_cache"]["fingerprint"]
    if args.config:
        data = (
            json.loads(args.config.read_text())
            if args.config.suffix.lower() == ".json"
            else tomllib.loads(args.config.read_text())
        )
        if not isinstance(data, dict) or any(isinstance(value, dict) for value in data.values()):
            raise ValueError("Unified config must be a flat JSON object or TOML table")
        settings.update(data)
    allowed = {field.name for field in fields(UnifiedConfig)}
    if set(settings) - allowed:
        raise ValueError(f"Unknown unified settings: {sorted(set(settings) - allowed)}")
    for name in ("device", "purpose", "epochs", "epoch_samples", "max_steps", "mixed_precision"):
        if getattr(args, name, None) is not None:
            settings[name] = getattr(args, name)
    config = UnifiedConfig(**settings)
    config.validate()
    # Keep progress visible on stderr and the public command result valid JSON.
    with redirect_stdout(sys.stderr):
        path = train_unified(
            args.manifest,
            args.output,
            config,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
            expected_content_sha256=content_pin,
            derived_cache_root=args.derived_cache,
            expected_cache_fingerprint=cache_pin,
        )
    return {"checkpoint": str(path), "purpose": config.purpose, "experimental": True}


def apply_unified(args, checkpoint):
    if args.command == "evaluate":
        return _evaluate(args, checkpoint)
    forbidden = (
        "normalized",
        "prepared",
        "prepared_array",
        "dicoms",
        "av_raw_proxy",
        "av_raw_matlab",
        "av_raw_matlab_source",
        "vessel_mask",
        "vessel_checkpoint",
        "vessel_normalized",
        "fov_diameter_pixels",
        "original_size",
    )
    supplied = [
        "--" + name.replace("_", "-")
        for name in forbidden
        if getattr(args, name, None) is not None and getattr(args, name, None) is not False
    ]
    if supplied:
        raise ValueError(
            f"Unified checkpoints require raw RGB; unsupported legacy options: {', '.join(supplied)}"
        )
    if args.pixels_per_degree != 25:
        raise ValueError("Unified checkpoints require --pixels-per-degree 25")
    if args.input is None:
        raise ValueError("--input is required for unified prediction")
    fov = None
    if args.fov is not None:
        if args.input.is_dir():
            raise ValueError("A supplied per-image --fov requires a single image, not a folder")
        with Image.open(args.fov) as image:
            fov = np.asarray(image.convert("L")) > 0
    options = {
        "device": args.device,
        "dataset": args.dataset,
        "fov": fov,
        "field_of_view_degrees": args.field_of_view_degrees,
        "threshold": args.threshold,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
    }
    if getattr(args, "tta", "none") != "none":
        options["tta"] = args.tta
    if args.input.is_dir():
        images = sorted(
            path
            for path in args.input.iterdir()
            if path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".gif",
            }
        )
        if not images:
            raise ValueError(f"No supported images in {args.input}")
        return [
            predict_unified(checkpoint, image, args.output / image.name, **options)
            for image in images
        ]
    return predict_unified(checkpoint, args.input, args.output, **options)


def _evaluate(args, checkpoint):
    if args.output.exists():
        raise FileExistsError(f"Existing evaluation report is preserved: {args.output}")
    if args.split != "val":
        raise ValueError(
            "Public unified evaluation permits --split val only; test requires a reviewed frozen-selection record schema"
        )
    if not 0 < args.threshold < 1:
        raise ValueError("Unified threshold must be in (0,1)")
    model, payload, device = load_unified(checkpoint, args.device)
    config = inference_config(payload, tile_size=args.tile_size, overlap=args.overlap)
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_sha != payload.get("manifest_sha256"):
        raise ValueError("Unified validation manifest differs from checkpoint")
    content_pin = payload.get("prepared_content_sha256")
    if not content_pin:
        raise ValueError("Unified validation requires checkpoint-pinned audited prepared content")
    verified = verify_prepared_content(args.manifest, expected=content_pin)
    metrics = evaluate_unified_model(
        model,
        args.manifest,
        split="val",
        device=device,
        config=config,
        threshold=args.threshold,
        **({"tta": args.tta} if getattr(args, "tta", "none") != "none" else {}),
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "prepared_content_sha256": verified,
            "experimental": True,
            "accuracy_claim": False,
            "test_evaluation_performed": False,
            "inference": {
                "mode": "whole" if config.train_view == "whole" else "tiled",
                "tile_size": config.tile_size if config.train_view == "patch" else None,
                "overlap": config.overlap if config.train_view == "patch" else None,
                "view_size": config.view_size if config.train_view == "whole" else None,
                **tta_protocol(getattr(args, "tta", "none")),
                "overrides": {
                    key: value
                    for key, value in {"tile_size": args.tile_size, "overlap": args.overlap}.items()
                    if value is not None
                },
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as report:
        report.write(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics
