"""The fixed four-component original benchmark pipeline for one raw photograph."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .bundled import bundled_checkpoint
from .inference import resolve_device
from .legacy.av import load_av, predict_av_input, restore_classes
from .legacy.checkpoints import load_single_net
from .legacy.inference import _resize_continuous_chw, _save
from .legacy.matlab_av import prepare_vesclass_matlab_calibrated
from .legacy.matlab_clahe import AV_KERNEL_PIXELS
from .legacy.matlab_fov import calibrate_provided_fov
from .legacy.preprocessing import DATASET_FOV_DEGREES, prepare_single_net
from .legacy.single_net import tiled_logits
from .prediction_output import check_prediction_output

CHANNELS = ("vessel", "artery", "vein", "disc", "cup")
POLICY = "original-raw-vessel-calibrated-provided-fov-av-disc0-universal-cup-in-disc0-v1"
COMPONENTS = {
    "vessel": (
        "legacy-vessels",
        "dae07480ec048aa8bb065c40d1dc07a3b9eb62b8e2f5ef9e66fe5acc70130d01",
    ),
    "disc": ("legacy-disc-0", "7d6ddc3bf144e970cc7f5e011e834e241b8d3762464aeacf08d6a87287cdc51f"),
    "universal": (
        "legacy-universal",
        "9fa4f9ffddb2fa39828f32bde85299ffaff74b7b8817a70c70243fa11362569f",
    ),
    "av": ("legacy-av", "824289039bad6c4c4bdc05ca289e157813c5307aaf3d31b49775499e227f270b"),
}


@dataclass
class OriginalPrediction:
    """Original-frame arrays; binary masks are authoritative (see probability semantics)."""

    masks: np.ndarray
    probabilities: np.ndarray
    av_conditional_probabilities: np.ndarray
    av_vessel_support: np.ndarray
    metadata: dict
    outputs: dict
    channels: tuple[str, ...] = CHANNELS


def _input(value, *, rgb):
    if isinstance(value, (str, Path)):
        data = Path(value).read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if rgb:
                array = np.array(image.convert("RGB"))
            else:
                array = np.array(image.convert("L") if image.mode == "P" else image)
        provenance = {"encoded_sha256": hashlib.sha256(data).hexdigest()}
    else:
        array = np.array(value, copy=True)
        provenance = {"array_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    provenance.update(shape=list(array.shape), dtype=str(array.dtype))
    return array, provenance


def _load_components(device):
    models, av, provenance = {}, None, {}
    for role, (name, expected) in COMPONENTS.items():
        with bundled_checkpoint(name) as path:
            data = Path(path).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Original component SHA mismatch: {name}")
        model, payload = (load_av if role == "av" else load_single_net)(
            io.BytesIO(data), device=device
        )
        provenance[role] = {"model": name, "sha256": actual}
        if role == "av":
            av = model
        else:
            models[role] = (model, payload)
    return models, av, provenance


def _compose(rgb, fov, prepared, models, av, device, kwargs, diameter):
    native, scores = {}, {}
    image = prepared.tensor()
    for name, (model, payload) in models.items():
        logits = tiled_logits(model, image, device=device, **payload["tiling"])
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite original {name} logits")
        native[name] = logits.numpy() > 0
        scores[name] = prepared.geometry.restore(logits.sigmoid().numpy(), order=1)
    # This is the SAME dedicated-disc forward, intersected before nearest restoration.
    native["universal"][2] &= native["disc"][0]
    restored = {k: prepared.geometry.restore(v, order=0) for k, v in native.items()}
    vessel = restored["vessel"][0]
    calibrated = prepare_vesclass_matlab_calibrated(
        rgb, vessel, fov=fov, fov_diameter_pixels=diameter, **kwargs
    )
    inputs = calibrated.inputs()
    classified = predict_av_input(av, inputs, device=device, original_size=rgb.shape[:2])
    classes = classified["classes"]
    conditional = _resize_continuous_chw(
        classified["prepared_conditional_probabilities"], rgb.shape[:2]
    )
    support = restore_classes((inputs[3] > 0).astype(np.uint8), rgb.shape[:2]) > 0
    masks = np.stack(
        (vessel, classes == 1, classes == 2, restored["disc"][0], restored["universal"][2])
    )
    probability = np.stack(
        (
            scores["vessel"][0],
            conditional[0] * support,
            conditional[1] * support,
            scores["disc"][0],
            scores["universal"][2],
        )
    )
    return masks, probability, conditional, support, classes, calibrated.metadata


def predict_original(
    image,
    *,
    fov,
    output=None,
    dataset=None,
    field_of_view_degrees=None,
    fov_diameter_pixels=None,
    device="auto",
):
    """Predict five masks with the fixed original benchmark composition.

    ``image`` is uint8 RGB or a path; ``fov`` is a provided binary original-frame
    mask or path. Supply a known dataset camera or explicit field angle. This is
    the calibrated provided-FOV route, not certification for arbitrary cameras.
    No checkpoint, threshold, preprocessing or tiling overrides are accepted.
    """
    if output is not None:
        check_prediction_output(output)
    rgb, image_info = _input(image, rgb=True)
    mask, fov_info = _input(fov, rgb=False)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB HWC image")
    if mask.dtype.kind not in "buif" or mask.shape != rgb.shape[:2] or not np.isfinite(mask).all():
        raise ValueError("Provide a finite single-channel original-frame FOV")
    levels = np.unique(mask)
    if len(levels) != 2 or levels[0] != 0 or levels[1] <= 0:
        raise ValueError("Provide a nonempty binary FOV with zero background")
    mask = mask > 0
    angle = field_of_view_degrees
    if angle is None:
        angle = DATASET_FOV_DEGREES.get((dataset or "").lower())
    if isinstance(angle, bool) or angle is None or not math.isfinite(angle) or angle <= 0:
        raise ValueError("Supply a known dataset or positive finite field_of_view_degrees")
    diameter = fov_diameter_pixels
    if diameter is None:
        diameter = calibrate_provided_fov(mask, dataset=dataset).diameter_pixels
    if isinstance(diameter, bool) or not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Supply a positive finite FOV diameter")
    shape = np.ceil(np.asarray(rgb.shape[:2]) * (25 * angle / diameter))
    if np.any(np.floor(shape / AV_KERNEL_PIXELS + 0.5) < 2):
        raise ValueError("Calibrated image is too small for source CLAHE tile grid")
    kwargs = {"dataset": dataset, "field_of_view_degrees": angle, "pixels_per_degree": 25}
    prepared = prepare_single_net(rgb, fov=mask, normalized=False, **kwargs)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Original prediction requires the default torch dtype float32")
    device = resolve_device(device)
    # Neither caller autocast nor BF16 training settings change this FP32 route.
    with (
        torch.inference_mode(),
        torch.autocast(device_type=torch.device(device).type, enabled=False),
    ):
        models, av, components = _load_components(device)
        masks, probability, conditional, support, classes, av_info = _compose(
            rgb, mask, prepared, models, av, device, kwargs, fov_diameter_pixels
        )
    metadata = {
        "kind": "original_composed_prediction",
        "policy": POLICY,
        "channels": list(CHANNELS),
        "output_shape": list(masks.shape),
        "components": components,
        "image": image_info,
        "provided_fov": fov_info,
        "single_net_geometry": asdict(prepared.geometry),
        "single_net_preprocessing": "raw RGB/255-.5 at 25 pixels/degree",
        "native_tiling": {k: p["tiling"] for k, (_, p) in models.items()},
        "av_preprocessing": av_info,
        "device": str(device),
        "precision": "float32",
        "binary_policy": "strict raw logits>0; cup intersects same-forward disc0 before nearest restore; AV ties unclassified",
        "probability_semantics": {
            "vessel_disc_cup": "independent sigmoid scores, linearly restored; cup score is not clipped",
            "artery_vein": "conditional softmax, linearly restored and gated by nearest-restored calibrated binary vessel; not soft-vessel joint probabilities",
            "masks": "authoritative nearest-restored decisions; not thresholds of exported interpolated scores",
        },
        "scope": "Provided-FOV calibrated benchmark policy; AV vessel roundtrip can differ at boundaries. No extra FOV clipping. Historical raw photometry/training exposure and arbitrary-camera MATLAB parity are not certified.",
        "benchmark_script_sha256": "3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae",
        "quality_policy_script_sha256": "44f1faf4ff41b3dd60b8d6dea747a577bcfd2bb77b1f2366e153bc688fb27a61",
    }
    outputs = {}
    if output is not None:
        outputs = _save(
            output,
            rgb,
            probability,
            CHANNELS,
            masks,
            metadata,
            arrays={"av_conditional_probability": conditional, "av_vessel_support": support},
            classes=classes,
        )
    return OriginalPrediction(
        masks, probability.astype(np.float32), conditional, support, metadata, outputs
    )
