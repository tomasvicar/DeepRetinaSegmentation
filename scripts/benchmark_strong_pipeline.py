"""Time the calibrated original four-network reference and one unified RGB model.

Validation reports select the original components; timing reads no annotations.
Keep this separate from the earlier universal+AV proxy latency measurement.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import torch
from benchmark_latency import measure
from PIL import Image

from deepretina.inference import resolve_device
from deepretina.legacy.av import load_av, predict_av_input
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import (
    UnifiedConfig,
    decode_probabilities,
    load_unified,
    predict_probabilities,
    restore_probabilities,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_snapshots(image_path, fov_path):
    """Decode and identify exactly one byte snapshot of each input resource."""
    image_bytes = Path(image_path).read_bytes()
    fov_bytes = Path(fov_path).read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as source:
        rgb = np.asarray(source.convert("RGB")).copy()
    with Image.open(io.BytesIO(fov_bytes)) as source:
        if source.mode not in {"1", "L", "I", "I;16", "P"}:
            raise ValueError("Require a single-channel FOV")
        # Palette indices are not intensities; match the public prediction route.
        fov = np.asarray(source.convert("L")) > 0
    if fov.shape != rgb.shape[:2] or not fov.any():
        raise ValueError("Require a nonempty original-frame FOV matching RGB")
    identity = {
        "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "fov_sha256": hashlib.sha256(fov_bytes).hexdigest(),
    }
    return rgb, fov, identity


def validate_reference(reference, paths):
    """Require precisely the calibrated four-component validation reference."""
    if (
        reference.get("family") != "original_reference"
        or reference.get("split") != "val"
        or reference.get("preview_only") is not False
        or reference.get("av_preprocessing_profile") != "vesclass_matlab_provided_fov_calibrated"
        or reference.get("disc_reference_variant") not in {"disc0/raw", "disc1/raw"}
        or reference.get("single_net_normalized") is not False
    ):
        raise ValueError("Require a full calibrated AV + dedicated raw disc validation reference")
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    for name, key in keys.items():
        if sha256(paths[name]) != reference.get(key):
            raise ValueError(f"Original {name} weights differ from selected reference")


def original_operation(rgb, fov, models, av, device, kwargs):
    """Share single-net preparation, retaining each checkpoint's native tiling."""
    prepared = prepare_single_net(rgb, fov=fov, normalized=False, **kwargs)
    image = prepared.tensor()
    restored = {}
    for name, (model, payload) in models.items():
        logits = tiled_logits(model, image, device=device, **payload["tiling"])
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite original logits")
        restored[name] = prepared.geometry.restore(logits.numpy() > 0, order=0)
    vessel = restored["vessel"][0]
    calibrated = prepare_vesclass_matlab_calibrated(rgb, vessel, fov=fov, **kwargs)
    classified = predict_av_input(
        av, calibrated.inputs(), device=device, original_size=rgb.shape[:2]
    )
    classes = classified["classes"]
    return np.stack(
        (vessel, classes == 1, classes == 2, restored["disc"][0], restored["universal"][2])
    )


def unified_operation(rgb, fov, model, config, device, kwargs):
    prepared = prepare_single_net(rgb, fov=fov, normalized=False, **kwargs)
    image = torch.from_numpy(prepared.image.transpose(2, 0, 1).copy()).float() / 255
    probability = predict_probabilities(
        model, image, device, config, config.tile_size, config.overlap
    )
    probability[[0, 3, 4]] *= prepared.fov[None]
    restored = restore_probabilities(probability, prepared.geometry)
    domain = prepared.geometry.restore(prepared.fov.astype(np.uint8), order=0) > 0
    restored[[0, 3, 4]] *= domain[None]
    restored[1:3, ~domain] = 0.5
    return decode_probabilities(restored, supported=model.supervised_channels)


def benchmark(args):
    device = resolve_device(args.device)
    paths = {name: getattr(args, name) for name in ("vessel", "disc", "universal", "av")}
    all_paths = {**paths, "unified": args.unified, "reference": args.reference_report}
    before = {key: sha256(path) for key, path in all_paths.items()}
    reference = json.loads(args.reference_report.read_text())
    validate_reference(reference, paths)
    rgb, fov, input_identity = input_snapshots(args.input, args.fov)
    kwargs = {"dataset": args.dataset, "field_of_view_degrees": args.field_of_view_degrees}
    models = {
        name: load_single_net(paths[name], device) for name in ("vessel", "disc", "universal")
    }
    for name, expected in {
        "vessel": ["vessel"],
        "disc": ["disc"],
        "universal": ["vessel", "disc", "cup"],
    }.items():
        if models[name][1]["channels"] != expected:
            raise ValueError(f"Incorrect {name} checkpoint channels")
        # crop_size and border are the runtime tiling arguments; blend is provenance.
        models[name][1]["tiling"] = {
            key: models[name][1]["tiling"][key] for key in ("crop_size", "border")
        }
    av, _ = load_av(paths["av"], device)
    with torch.inference_mode():
        original_timing = measure(
            lambda models=models, av=av: original_operation(rgb, fov, models, av, device, kwargs),
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    original_parameters = sum(p.numel() for model, _ in models.values() for p in model.parameters())
    original_parameters += sum(p.numel() for p in av.parameters())
    del models, av
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, payload, _ = load_unified(args.unified, device)
    config = UnifiedConfig(**payload["config"])
    if config.purpose != "full_data" or not all(model.supervised_channels):
        raise ValueError("Require a full-data unified model supervised on all five heads")
    if config.preprocessing_profile != "legacy25_raw":
        raise ValueError("Benchmark requires the declared raw legacy25 unified profile")
    with torch.inference_mode():
        unified_timing = measure(
            lambda: unified_operation(rgb, fov, model, config, device, kwargs),
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    if before != {key: sha256(path) for key, path in all_paths.items()}:
        raise ValueError("Benchmark checkpoint/reference changed; use immutable snapshots")
    return {
        "schema_version": 1,
        "accuracy_claim": False,
        "scope": "one raw photo, CPU preprocessing + model inference + original-frame mask restoration; excludes file IO/model loading",
        "script_sha256": sha256(__file__),
        "input_sha256": input_identity["input_sha256"],
        "fov_sha256": input_identity["fov_sha256"],
        "input_shape": list(rgb.shape),
        "dataset": args.dataset,
        "field_of_view_degrees": args.field_of_view_degrees,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "original": {
            "checkpoint_sha256": {key: before[key] for key in paths},
            "reference_sha256": before["reference"],
            "disc_variant": reference["disc_reference_variant"],
            "parameters": original_parameters,
            "checkpoint_bytes": sum(path.stat().st_size for path in paths.values()),
            "av_profile": "vesclass_matlab_provided_fov_calibrated",
            "timing": original_timing,
        },
        "unified": {
            "checkpoint_sha256": before["unified"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": args.unified.stat().st_size,
            "config": payload["config"],
            "checkpoint_epoch": payload.get("epoch"),
            "timing": unified_timing,
        },
        "limitations": [
            "One photograph cannot establish general speed or quality; repeat on representative geometries",
            "Historical original exposure and preprocessing provenance remain unresolved",
            "Calibrated AV photometry retains documented numerical residuals",
            "Original single-net preparation is shared across three networks; individual native tiles are retained",
            "Original threshold is strict logit>0, AV retains original tie0; no added original cup containment",
            "Unified retains its configured inference geometry, vessel gating and cup containment",
            "Torch CUDA allocation excludes external/process memory; other CPU/GPU load affects timings",
            "Checkpoint epoch/purpose do not establish final model selection or sufficient convergence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "fov",
        "vessel",
        "disc",
        "universal",
        "av",
        "unified",
        "reference-report",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--field-of-view-degrees", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier measurements; choose a new output")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key]["timing"] for key in ("original", "unified")}))


if __name__ == "__main__":
    main()
