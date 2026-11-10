"""Measure complete original/unified prediction pipelines on one real photograph.

This is a latency/size measurement, never an accuracy evaluation. The original
A/V path uses the explicitly uncertified Python raw preprocessing proxy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deepretina.inference import resolve_device
from deepretina.legacy.av import load_av, predict_av_input
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.preprocessing import prepare_single_net, prepare_vesclass_raw_proxy
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import (
    decode_probabilities,
    load_unified,
    predict_probabilities,
    restore_probabilities,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(operation, device, *, warmup=1, repeats=3):
    """Synchronize CUDA around each complete call; retain all timing samples."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        operation()
    synchronize()
    baseline = peak = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append(time.perf_counter() - start)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    return {
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_working_bytes": None if peak is None else peak - baseline,
        "warmup_calls": warmup,
        "measured_calls": repeats,
    }


def benchmark(
    image_path,
    single_path,
    av_path,
    unified_path,
    *,
    dataset=None,
    degrees=None,
    device="auto",
    fov_path=None,
    repeats=3,
    warmup=1,
):
    device = resolve_device(device)
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB")).copy()
    fov = None
    if fov_path is not None:
        with Image.open(fov_path) as source:
            fov = np.asarray(source.convert("L")) > 0
    kwargs = {"dataset": dataset, "field_of_view_degrees": degrees, "fov": fov}
    single, single_info = load_single_net(single_path, device)
    av, _ = load_av(av_path, device)
    if single_info["channels"] != ["vessel", "disc", "cup"]:
        raise ValueError("Original complete pipeline requires the universal3head checkpoint")

    @torch.inference_mode()
    def original():
        prepared = prepare_single_net(rgb, **kwargs)
        logits = tiled_logits(
            single,
            prepared.tensor(),
            device=device,
            crop_size=single_info["tiling"]["crop_size"],
            border=single_info["tiling"]["border"],
        )
        masks = prepared.geometry.restore(logits.numpy() > 0, order=0)
        proxy = prepare_vesclass_raw_proxy(rgb, masks[0], **kwargs)
        classified = predict_av_input(
            av, proxy.inputs(), device=device, original_size=rgb.shape[:2]
        )
        return masks, classified["classes"]

    report = {
        "schema_version": 1,
        "accuracy_claim": False,
        "scope": "single real photograph; complete CPU preprocessing + GPU/CPU inference + original-frame restoration; excludes file IO and model loading",
        "input_sha256": file_sha256(image_path),
        "input_shape": list(rgb.shape),
        "dataset": dataset,
        "field_of_view_degrees": degrees,
        "fov_sha256": file_sha256(fov_path) if fov_path else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "original": {
            "checkpoint_sha256": [file_sha256(single_path), file_sha256(av_path)],
            "parameters": sum(p.numel() for m in (single, av) for p in m.parameters()),
            "checkpoint_bytes": sum(Path(p).stat().st_size for p in (single_path, av_path)),
            "single_net_profile": "source-inferred raw legacy25",
            "av_matlab_preprocessing_parity_certified": False,
            "timing": measure(original, device, warmup=warmup, repeats=repeats),
        },
    }
    del original
    single, av = None, None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, payload, _ = load_unified(unified_path, device)

    @torch.inference_mode()
    def unified():
        prepared = prepare_single_net(rgb, **kwargs)
        image = torch.from_numpy(prepared.image.transpose(2, 0, 1).copy()).float() / 255
        probability = predict_probabilities(model, image, device)
        restored = restore_probabilities(probability, prepared.geometry)
        return decode_probabilities(restored, supported=model.supervised_channels)

    report["unified"] = {
        "checkpoint_sha256": file_sha256(unified_path),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": Path(unified_path).stat().st_size,
        "purpose": payload["config"]["purpose"],
        "timing": measure(unified, device, warmup=warmup, repeats=repeats),
    }
    report["limitations"] = [
        "One photograph cannot establish general speed or model quality",
        "Original A/V raw preprocessing is an uncertified proxy",
        "Original raw/normalized training profile remains historically uncertain",
        "CPU preprocessing included in latency; other system load may affect results",
        "CUDA peak reports Torch allocation, not total process or device memory",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original-universal", type=Path, required=True)
    parser.add_argument("--original-av", type=Path, required=True)
    parser.add_argument("--unified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--field-of-view-degrees", type=float)
    parser.add_argument("--fov", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve earlier measurements")
    result = benchmark(
        args.input,
        args.original_universal,
        args.original_av,
        args.unified,
        dataset=args.dataset,
        degrees=args.field_of_view_degrees,
        fov_path=args.fov,
        device=args.device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key]["timing"] for key in ("original", "unified")}))


if __name__ == "__main__":
    main()
