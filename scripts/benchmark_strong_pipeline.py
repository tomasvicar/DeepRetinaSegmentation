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
import os
import platform
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from benchmark_latency import measure, synchronize_device
from PIL import Image

from deepretina import unified as unified_api
from deepretina.inference import resolve_device
from deepretina.inference_tta import MODES
from deepretina.inference_tta import protocol_extension as tta_protocol
from deepretina.legacy.av import load_av, predict_av_input
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.legacy.single_net import tiled_logits
from deepretina.photometry import apply as apply_photometry
from deepretina.unified import (
    UnifiedConfig,
    decode_probabilities,
    inference_config,
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


CUP_COMPONENT_POLICY = "Fresh native cup strictlogit>0 vs cup intersect SAMEforward fresh dedicated disc strictlogit>0; no GTused for clipping"


def reference_cup_policy(reference):
    """Identify the completed paired source policy, without reading GT/resources."""
    component = reference.get("fresh_paired_cup_component")
    if component is None:
        if reference.get("optional_cup_postprocessing_experiment") is not None:
            raise ValueError("Require the completed paired cup policy reference")
        return "native"
    disc = reference["disc_reference_variant"].split("/")[0]
    if (
        component.get("policy") != CUP_COMPONENT_POLICY
        or component.get("reference_context") != disc
    ):
        raise ValueError("Cup reference policy/source context differs")
    variant = component.get("cup_variant")
    if variant == "native":
        return "native"
    if variant == disc:
        return "inside-disc"
    raise ValueError("Cup clipping disc differs from selected disc reference")


def validate_reference(reference, paths, *, cup_policy="native", disc_variant=None):
    """Require precisely the calibrated four-component validation reference."""
    if (
        reference.get("family")
        not in {"original_reference", "optional_fresh_paired_cup_experiment"}
        or reference.get("family") == "optional_fresh_paired_cup_experiment"
        and not reference.get("fresh_paired_cup_component")
        or reference.get("split") != "val"
        or reference.get("preview_only") is not False
        or reference.get("av_preprocessing_profile") != "vesclass_matlab_provided_fov_calibrated"
        or reference.get("disc_reference_variant") not in {"disc0/raw", "disc1/raw"}
        or reference.get("single_net_normalized") is not False
    ):
        raise ValueError("Require a full calibrated AV + dedicated raw disc validation reference")
    if cup_policy not in {"native", "inside-disc"}:
        raise ValueError("Unknown original cup policy")
    if disc_variant is not None and reference["disc_reference_variant"] != disc_variant:
        raise ValueError("Selected disc variant differs from reference")
    if cup_policy == "inside-disc" and disc_variant is None:
        raise ValueError("Inside-disc requires an explicit selected disc variant")
    if reference_cup_policy(reference) != cup_policy:
        raise ValueError("Quality reference cup policy differs from timed decoding")
    for key in ("manifest_sha256", "prepared_content_sha256"):
        value = reference.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Require exact reference manifest/canonical SHA256 metadata")
    if not isinstance(reference.get("policy"), str) or not reference["policy"]:
        raise ValueError("Require declared quality reference GT/decoding policy")
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    for name, key in keys.items():
        if sha256(paths[name]) != reference.get(key):
            raise ValueError(f"Original {name} weights differ from selected reference")


def original_operation(rgb, fov, models, av, device, kwargs, *, cup_policy="native"):
    """Share single-net preparation, retaining each checkpoint's native tiling."""
    if cup_policy not in {"native", "inside-disc"}:
        raise ValueError("Unknown original cup policy")
    prepared = prepare_single_net(rgb, fov=fov, normalized=False, **kwargs)
    image = prepared.tensor()
    native = {}
    for name, (model, payload) in models.items():
        logits = tiled_logits(model, image, device=device, **payload["tiling"])
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite original logits")
        native[name] = logits.numpy() > 0
    if cup_policy == "inside-disc":
        # Exactly the completed paired experiment: same-forward raw masks, >0.
        native["universal"][2] &= native["disc"][0]
    restored = {name: prepared.geometry.restore(mask, order=0) for name, mask in native.items()}
    vessel = restored["vessel"][0]
    calibrated = prepare_vesclass_matlab_calibrated(rgb, vessel, fov=fov, **kwargs)
    classified = predict_av_input(
        av, calibrated.inputs(), device=device, original_size=rgb.shape[:2]
    )
    classes = classified["classes"]
    return np.stack(
        (vessel, classes == 1, classes == 2, restored["disc"][0], restored["universal"][2])
    )


def unified_operation(rgb, fov, model, config, device, kwargs, *, threshold=0.5, tta="none"):
    tta_protocol(tta)
    prepared = prepare_single_net(rgb, fov=fov, normalized=False, **kwargs)
    processed = apply_photometry(
        prepared.image,
        config.photometric_profile,
        fov=prepared.fov,
        pixels_per_degree=config.pixels_per_degree,
    )
    image = torch.from_numpy(processed.transpose(2, 0, 1).copy()).float() / 255
    probability = predict_probabilities(
        model, image, device, config, config.tile_size, config.overlap,
        **({"tta": tta} if tta != "none" else {}),
    )
    probability[[0, 3, 4]] *= prepared.fov[None]
    restored = restore_probabilities(probability, prepared.geometry)
    domain = prepared.geometry.restore(prepared.fov.astype(np.uint8), order=0) > 0
    restored[[0, 3, 4]] *= domain[None]
    restored[1:3, ~domain] = 0.5
    return decode_probabilities(restored, threshold=threshold, supported=model.supervised_channels)


def package_identity():
    root = Path(unified_api.__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return {"package_root": str(root), "python_package_tree_sha256": digest.hexdigest(),
            "python_files": len(files), "unified_module_sha256": sha256(unified_api.__file__),
            "timing_helper_sha256": sha256(Path(__file__).with_name("benchmark_latency.py"))}


def output_identity(masks, shape):
    """Inspect the retained last call outside clocks; never add a forward."""
    if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_ or masks.shape != (5, *shape):
        raise ValueError("Require five original-frame boolean masks")
    return {"shape": list(masks.shape), "dtype": str(masks.dtype),
            "sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
            "foreground_pixels": masks.sum(axis=(1, 2)).tolist()}


def timed_load(operation, device):
    synchronize_device(device)
    started = time.perf_counter()
    result = operation()
    synchronize_device(device)
    return result, time.perf_counter() - started


def benchmark(args):
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Benchmark requires default float32 inference")
    if not np.isfinite(args.unified_threshold) or not 0 < args.unified_threshold < 1:
        raise ValueError("Unified threshold must be finite and strictly between0 and1")
    tta = getattr(args, "unified_tta", "none")
    tta_protocol(tta)
    phase_timings = getattr(args, "phase_timings", False)
    include_input_io = getattr(args, "include_input_io", False)
    cup_policy = getattr(args, "original_cup_policy", "native")
    disc_variant = getattr(args, "original_disc_variant", None)
    if cup_policy not in {"native", "inside-disc"}:
        raise ValueError("Unknown original cup policy")
    device = resolve_device(args.device)
    paths = {name: getattr(args, name) for name in ("vessel", "disc", "universal", "av")}
    all_paths = {**paths, "unified": args.unified, "reference": args.reference_report}
    before = {key: sha256(path) for key, path in all_paths.items()}
    expected_sha = getattr(args, "expected_unified_sha256", None)
    if expected_sha is not None and before["unified"] != expected_sha:
        raise ValueError("Unified checkpoint differs from explicit selected byte SHA")
    reference = json.loads(args.reference_report.read_text())
    validate_reference(reference, paths, cup_policy=cup_policy, disc_variant=disc_variant)
    rgb, fov, input_identity = input_snapshots(args.input, args.fov)
    kwargs = {"dataset": args.dataset, "field_of_view_degrees": args.field_of_view_degrees}
    synchronize_device(device)
    load_started = time.perf_counter()
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
    synchronize_device(device)
    original_load_seconds = time.perf_counter() - load_started
    last_outputs = {}

    def inputs_for_call():
        if not include_input_io:
            return rgb, fov
        current_rgb, current_fov, identity = input_snapshots(args.input, args.fov)
        if identity != input_identity:
            raise ValueError("Input snapshot changed during input-IO timing")
        return current_rgb, current_fov

    def original_call(models=models, av=av):
        current_rgb, current_fov = inputs_for_call()
        last_outputs["original"] = original_operation(
            current_rgb, current_fov, models, av, device, kwargs, cup_policy=cup_policy
        )

    timing_options = {"record_warmup": True} if phase_timings else {}
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        original_timing = measure(
            original_call,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            **timing_options,
        )
    original_parameters = sum(p.numel() for model, _ in models.values() for p in model.parameters())
    original_parameters += sum(p.numel() for p in av.parameters())
    del original_call, models, av
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    (model, payload, _), unified_load_seconds = timed_load(
        lambda: load_unified(args.unified, device), device
    )
    tile = getattr(args, "unified_tile_size", None)
    overlap = getattr(args, "unified_overlap", None)
    config = inference_config(payload, tile_size=tile, overlap=overlap) if tile is not None or overlap is not None else UnifiedConfig(**payload["config"])
    if config.purpose != "full_data" or not all(model.supervised_channels):
        raise ValueError("Require a full-data unified model supervised on all five heads")
    if config.preprocessing_profile != "legacy25_raw":
        raise ValueError("Benchmark requires the declared raw legacy25 unified profile")
    def unified_call():
        current_rgb, current_fov = inputs_for_call()
        options = {"tta": tta} if tta != "none" else {}
        last_outputs["unified"] = unified_operation(
            current_rgb, current_fov, model, config, device, kwargs,
            threshold=args.unified_threshold, **options
        )

    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        unified_timing = measure(
            unified_call,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            **timing_options,
        )
    if before != {key: sha256(path) for key, path in all_paths.items()}:
        raise ValueError("Benchmark checkpoint/reference changed; use immutable snapshots")
    masks = {key: output_identity(value, rgb.shape[:2]) for key, value in last_outputs.items()}
    return {
        "schema_version": 1,
        "accuracy_claim": False,
        "scope": "one raw photo; five-mask pipeline preprocessing+model+restore/decode; excludes model loading/output serialization/CLI startup" + ("; includes input read/decode/SHA per call" if include_input_io else "; excludes input file IO"),
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
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "interop_threads": torch.get_num_interop_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "inference_precision": "float32_no_autocast",
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "machine": platform.machine(),
            "cuda_runtime": torch.version.cuda,
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid) if device.type == "cuda" else None,
            "torch_default_dtype": str(torch.get_default_dtype()),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "process_nice": os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "whole_process_peak_RSS_KiB_not_per_pipeline": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "execution_source": package_identity(),
        "last_mask_identity_no_extra_forward": masks,
        **({"model_load_seconds": {"original_four_networks": original_load_seconds, "unified": unified_load_seconds},
             "load_scope": "same-process deserialize+device transfer, synchronized; preceding byte hashes warm filesystem; not cold process startup"} if phase_timings else {}),
        "original": {
            "network_count": 4,
            "checkpoint_sha256": {key: before[key] for key in paths},
            "reference_sha256": before["reference"],
            "reference_manifest_sha256": reference["manifest_sha256"],
            "reference_prepared_content_sha256": reference["prepared_content_sha256"],
            "quality_reference_gt_policy": reference["policy"],
            "disc_variant": reference["disc_reference_variant"],
            "cup_policy": cup_policy,
            "cup_decoding": "native universal/raw strictlogit>0"
            if cup_policy == "native"
            else "native universal/raw strictlogit>0 intersect same-forward selected dedicated raw disc strictlogit>0 before nearest restoration",
            "quality_reference_cup_policy": reference_cup_policy(reference),
            "parameters": original_parameters,
            "checkpoint_bytes": sum(path.stat().st_size for path in paths.values()),
            "av_profile": "vesclass_matlab_provided_fov_calibrated",
            "timing": original_timing,
        },
        "unified": {
            "threshold": args.unified_threshold,
            **tta_protocol(tta),
            "resolved_tile_size": config.tile_size if config.train_view == "patch" else None,
            "resolved_overlap": config.overlap if config.train_view == "patch" else None,
            "checkpoint_sha256": before["unified"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": args.unified.stat().st_size,
            "config": payload["config"],
            "model_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            "floating_model_dtypes": sorted({str(t.dtype) for t in model.state_dict().values() if t.is_floating_point()}),
            "optimizer_or_training_state_present": "optimizer" in payload or "training_state" in payload,
            "weight_transformation_or_compression_applied": False,
            "checkpoint_epoch": payload.get("epoch"),
            "timing": unified_timing,
        },
        "limitations": [
            "One photograph cannot establish general speed or quality; repeat on predeclared TRAIN geometries",
            "Five-mask benchmark matches the declared composition, not full product API probability exports, output file writes or CLI process startup",
            "Original four-network parameter count includes each complete network even though universal vessel/disc outputs are discarded",
            "Historical original exposure and preprocessing provenance remain unresolved",
            "Calibrated AV photometry retains documented numerical residuals",
            "Original single-net preparation is shared across three networks; individual native tiles are retained",
            "Original threshold is strict logit>0, AV retains original tie0; declared optional cup policy is identical in selected quality reference and timed pipeline",
            "Inside-disc is an added postprocessing variant, not a modification of original weights/defaults; completed paired VAL evidence includes TP losses and backend numeric drift",
            "Unified retains its configured inference geometry, vessel gating and cup containment",
            "Torch CUDA allocation covers measured calls after warmup, excludes load/warmup peaks and external/process memory; other CPU/GPU load affects timings",
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
    parser.add_argument(
        "--original-cup-policy", choices=("native", "inside-disc"), default="native"
    )
    parser.add_argument(
        "--original-disc-variant",
        choices=("disc0/raw", "disc1/raw"),
        required=True,
        help="Explicit selected component; must match checkpoint and quality reference",
    )
    parser.add_argument(
        "--unified-threshold", type=float, default=0.5, help="Frozen selected VAL threshold"
    )
    parser.add_argument("--unified-tta", choices=MODES, default="none")
    parser.add_argument("--unified-tile-size", type=int)
    parser.add_argument("--unified-overlap", type=int)
    parser.add_argument("--expected-unified-sha256", help="Exact already-selected immutable checkpoint SHA; does not select a model")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--interop-threads", type=int)
    parser.add_argument("--opencv-threads", type=int)
    parser.add_argument("--phase-timings", action="store_true", help="Record synchronized model-load and each warmup call separately")
    parser.add_argument("--include-input-io", action="store_true", help="Read/decode/hash same pinned RGB+FOV inside every mask-pipeline call; output IO still excluded")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier measurements; choose a new output")
    for name in ("cpu_threads", "interop_threads", "opencv_threads"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error("Thread counts must be positive")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    if args.interop_threads is not None:
        torch.set_num_interop_threads(args.interop_threads)
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key]["timing"] for key in ("original", "unified")}))


if __name__ == "__main__":
    main()
