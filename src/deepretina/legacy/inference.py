"""Original-model inference with explicit input contracts and geometry provenance."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ..geometry_policy import prediction_geometry, validate_geometry_policy
from ..inference import resolve_device
from .av import load_av, predict_av_input, restore_classes
from .checkpoints import load_single_net
from .preprocessing import load_vesclass_dicoms, prepare_single_net, stack_vesclass_inputs
from .single_net import tiled_logits

_PALETTE = {
    "vessel": (50, 220, 80),
    "disc": (255, 190, 40),
    "cup": (50, 220, 255),
    "artery": (255, 50, 50),
    "vein": (50, 100, 255),
}


def _rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _mask(path):
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim != 2:
        raise ValueError("Supply a single-channel mask, not a color visualization")
    return mask


def _resize_continuous_chw(array, shape, mode="bilinear"):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None]
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(tensor, size=tuple(shape), mode=mode, **kwargs)[0].numpy()


def _save(output, image, probability, channels, masks, metadata, *, arrays=None, classes=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"probabilities": str(output / "probabilities.npz")}
    np.savez_compressed(
        paths["probabilities"],
        probability=probability.astype(np.float32),
        channels=np.asarray(channels),
        **(arrays or {}),
    )
    color = np.zeros_like(image)
    for name, mask in zip(channels, masks, strict=True):
        path = output / f"{name}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        paths[name] = str(path)
        color[mask] = _PALETTE[name]
    if classes is not None:
        paths["classes"] = str(output / "classes.png")
        Image.fromarray(classes.astype(np.uint8)).save(paths["classes"])
    active = masks.any(axis=0)
    overlay = image.copy()
    overlay[active] = np.round(0.55 * image[active] + 0.45 * color[active]).astype(np.uint8)
    paths["overlay"] = str(output / "overlay.png")
    Image.fromarray(overlay).save(paths["overlay"])
    paths["metadata"] = str(output / "prediction.json")
    metadata["outputs"] = paths
    Path(paths["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return paths


def predict_legacy(
    checkpoint: Path,
    image_path: Path | None,
    output: Path,
    device: str = "auto",
    dataset: str | None = None,
    fov_path: Path | None = None,
    field_of_view_degrees: float | None = None,
    normalized: bool | None = None,
    threshold: float = 0.5,
    vessel_mask_path: Path | None = None,
    dicom_paths=None,
    prepared: bool = False,
    original_size: tuple[int, int] | None = None,
    pixels_per_degree: float = 25.0,
    prepared_inputs: np.ndarray | None = None,
) -> dict:
    """Export original-size masks, probabilities, overlay and audit metadata.

    ``single_net`` takes a raw photograph and applies its original loader profile.
    ``legacy_av`` requires explicitly prepared RGB plus a binary vessel mask, or
    four stored DICOM planes in R,G,B,vessel order. Its optional ``original_size``
    means full-frame (height,width) resize only; cropped inputs need external
    geometry restoration. No raw MATLAB A/V preprocessing parity is claimed.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    family = payload.get("family")
    if payload.get("geometry_policy") is not None:
        validate_geometry_policy(payload["geometry_policy"])
    device = resolve_device(device)
    metadata = {
        "family": family,
        "image": str(image_path) if image_path is not None else None,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "provenance": payload.get("provenance", {}),
        "checkpoint_preprocessing": payload.get("preprocessing", {}),
    }
    if family == "legacy_single_net":
        if image_path is None:
            raise ValueError("single_net requires an RGB image_path")
        if (
            prepared
            or dicom_paths is not None
            or vessel_mask_path is not None
            or prepared_inputs is not None
        ):
            raise ValueError("Prepared RGB/vessel and DICOM inputs are specific to legacy_av")
        if original_size is not None:
            raise ValueError("single_net restores its original photograph geometry automatically")
        image = _rgb(image_path)
        saved_normalized = payload.get("preprocessing", {}).get("normalized")
        if saved_normalized is not None and type(saved_normalized) is not bool:
            raise ValueError("Saved single_net normalized choice must be boolean or unknown")
        effective_normalized = bool(saved_normalized) if normalized is None else normalized
        angle, geometry_provenance = prediction_geometry(
            payload.get("geometry_policy"), dataset, image.shape[:2], field_of_view_degrees
        )
        prepared_image = prepare_single_net(
            image,
            dataset=dataset,
            fov=_mask(fov_path) if fov_path is not None else None,
            field_of_view_degrees=angle,
            pixels_per_degree=pixels_per_degree,
            normalized=effective_normalized,
        )
        model, payload = load_single_net(checkpoint, device)
        tiling = payload["tiling"]
        logits = tiled_logits(
            model,
            prepared_image.tensor(),
            device,
            crop_size=tiling["crop_size"],
            border=tiling["border"],
        )
        prepared_probability = logits.sigmoid().numpy()
        geometry = prepared_image.geometry
        # Classify before nearest restoration, retaining the original decisions.
        # Restore probabilities AFTER sigmoid: empty canvas pixels must stay zero.
        probability = np.stack([geometry.restore(p, order=1) for p in prepared_probability])
        # Preserve strict original logit decisions at the default threshold:
        # float32 sigmoid(+1e-8) rounds to exactly0.5 and would erase positives.
        prepared_masks = (
            logits.numpy() > 0 if threshold == 0.5 else prepared_probability > threshold
        )
        masks = np.stack([geometry.restore(p, order=0) for p in prepared_masks])
        channels = payload["channels"]
        metadata.update(
            channels=channels,
            threshold=threshold,
            shape=list(image.shape[:2]),
            inference_shape=list(prepared_image.image.shape[:2]),
            geometry=asdict(geometry),
            dataset=dataset,
            preprocessing={
                "profile": "single_net_normalized" if effective_normalized else "single_net_raw",
                "normalized": bool(effective_normalized),
                "normalization_verified": bool(
                    payload.get("preprocessing", {}).get("normalization_verified", False)
                ),
                "normalization_selection": "explicit caller override"
                if normalized is not None
                else (
                    "saved checkpoint preprocessing metadata"
                    if saved_normalized is not None
                    else "historical source-inferred raw default"
                ),
                "historical_profile_status": payload.get("preprocessing", {}).get(
                    "normalization_status", "historical raw/normalized choice not verified"
                ),
                "field_of_view_degrees": prepared_image.field_of_view_degrees,
                "pixels_per_degree": prepared_image.pixels_per_degree,
                "fov_source": prepared_image.fov_source,
                "fov_path": str(fov_path) if fov_path is not None else None,
                **(
                    {"geometry_policy": geometry_provenance}
                    if geometry_provenance is not None
                    else {}
                ),
            },
            tiling=tiling,
            probability_semantics="independent sigmoid after weighted logit blending",
            mask_decision="strict_logit_gt_zero"
            if threshold == 0.5
            else "strict_probability_gt_threshold",
            mask_policy="independent original heads; no cup/disc containment postprocessing",
            resampling="nearest prepared binary masks; linear sigmoid probabilities; zero outside crop",
        )
        return _save(output, image, probability, channels, masks, metadata)

    if family != "legacy_av":
        raise ValueError(f"Unsupported legacy checkpoint family: {family!r}")
    from .canonical_av import PROFILE

    if payload.get("preprocessing", {}).get("profile") == PROFILE:
        if (
            prepared
            or prepared_inputs is not None
            or dicom_paths is not None
            or original_size is not None
        ):
            raise ValueError(
                "Canonical raw AV requires raw RGB and an original-frame vessel mask; historical prepared/DICOM modes do not apply"
            )
        if normalized:
            raise ValueError("Canonical raw AV was trained without contrast normalization")
        if image_path is None or vessel_mask_path is None:
            raise ValueError(
                "Canonical raw AV requires --input and --vessel-mask; no historical vessel model is selected implicitly"
            )
        from .canonical_av import prepare_raw

        image = _rgb(image_path)
        prepared_image, inputs, geometry_provenance = prepare_raw(
            image,
            _mask(vessel_mask_path),
            payload,
            dataset=dataset,
            fov=_mask(fov_path) if fov_path is not None else None,
            field_of_view_degrees=field_of_view_degrees,
            pixels_per_degree=pixels_per_degree,
        )
        model, payload = load_av(checkpoint, device)
        result = predict_av_input(model, inputs, device=device)
        geometry = prepared_image.geometry
        classes = geometry.restore(result["prepared_classes"], order=0).astype(np.uint8)
        conditional = np.stack(
            [
                geometry.restore(p, order=1, fill=0.5)
                for p in result["prepared_conditional_probabilities"]
            ]
        )
        support = geometry.restore(inputs[3] != 0, order=0).astype(bool)
        probability = conditional * support[None]
        metadata.update(
            channels=["artery", "vein"],
            shape=list(image.shape[:2]),
            inference_shape=result["prepared_shape"],
            geometry=asdict(geometry),
            dataset=dataset,
            preprocessing={
                "profile": PROFILE,
                "normalized": False,
                "pixels_per_degree": 25,
                "field_of_view_degrees": prepared_image.field_of_view_degrees,
                "fov_source": prepared_image.fov_source,
                "vessel_source": "explicit caller-supplied original-frame mask",
                "geometry_policy": geometry_provenance,
            },
            probability_semantics="conditional softmax of original logits1:3, gated by supplied vessel support",
            mask_decision="strict artery/vein logit comparison; ties0; supplied vessel support",
            mask_policy="independent canonical AV recipe; no historical MATLAB input claim",
            accuracy_claim="conditional classification requires independent evaluation; supplied vessel quality is separate",
        )
        return _save(
            output,
            image,
            probability,
            ["artery", "vein"],
            np.stack([classes == 1, classes == 2]),
            metadata,
            arrays={"vessel_support": support, "conditional_probability": conditional},
            classes=classes,
        )
    if dicom_paths is None and not prepared and prepared_inputs is None:
        raise ValueError(
            "Legacy A/V requires already-preprocessed RGB with prepared=True and vessel_mask_path, "
            "or dicom_paths=(red,green,blue,vessel). Raw MATLAB CLAHE preprocessing has no "
            "independent golden validation; a raw photograph is not accepted as prepared input."
        )
    if normalized or fov_path is not None or field_of_view_degrees is not None:
        raise ValueError(
            "A/V prepared inputs must not receive single_net normalization or FOV crop"
        )
    if prepared_inputs is not None:
        if dicom_paths is not None or vessel_mask_path is not None:
            raise ValueError("Prepared tensor already supplies the four A/V planes")
        inputs = np.asarray(prepared_inputs, dtype=np.float32)
        if inputs.ndim != 3 or inputs.shape[0] != 4:
            raise ValueError("Prepared A/V tensor must have shape 4,H,W")
        input_profile = "caller_supplied_prepared_four_channel_tensor"
    elif dicom_paths is not None:
        if vessel_mask_path is not None:
            raise ValueError("DICOM input already supplies its vessel plane")
        dicom_paths = tuple(dicom_paths)
        inputs = load_vesclass_dicoms(dicom_paths)
        input_profile = "stored_dicom_planes"
    else:
        if image_path is None or vessel_mask_path is None:
            raise ValueError("Prepared A/V inference requires image_path and vessel_mask_path")
        image = _rgb(image_path)
        vessel = _mask(vessel_mask_path)
        values = np.unique(vessel)
        if len(values) != 2 or values[0] != 0 or values[1] <= 0:
            raise ValueError("A/V vessel mask must contain zero and one positive foreground value")
        inputs = stack_vesclass_inputs(image, vessel)
        input_profile = "caller_asserted_preprocessed_rgb"
    if (inputs[3] < 0).any():
        raise ValueError("A/V vessel foreground values must be positive")
    if original_size is not None:
        if len(original_size) != 2 or any(int(x) != x or x < 1 for x in original_size):
            raise ValueError("original_size must contain positive integer height,width")
        original_size = tuple(int(x) for x in original_size)
    model, payload = load_av(checkpoint, device)
    result = predict_av_input(model, inputs, device=device, original_size=original_size)
    classes = result["classes"]
    shape = classes.shape
    conditional = _resize_continuous_chw(
        result["prepared_conditional_probabilities"], shape, mode="nearest"
    )
    vessel = restore_classes((inputs[3] != 0).astype(np.uint8), shape).astype(bool)
    probability = conditional * vessel[None]
    if image_path is not None:
        image = _rgb(image_path)
        overlay_source = "supplied RGB image"
    else:
        rgb = inputs[:3].transpose(1, 2, 0)
        high = max(float(rgb.max()), 1.0)
        image = np.clip(rgb / high * 255, 0, 255).astype(np.uint8)
        overlay_source = "DICOM RGB scaled by common maximum for display only"
    if image.shape[:2] != shape:
        image = np.array(Image.fromarray(image).resize(shape[::-1], Image.Resampling.BILINEAR))
    channels = ["artery", "vein"]
    masks = np.stack([classes == 1, classes == 2])
    metadata.update(
        channels=channels,
        shape=list(shape),
        inference_shape=result["prepared_shape"],
        dataset=dataset,
        preprocessing={
            "profile": input_profile,
            "raw_matlab_pipeline_certified": False,
            "input_preprocessing_verified": False,
            "normalization": "original internal per-image/channel sample standard deviation; no epsilon",
            "dicom_paths": [str(p) for p in dicom_paths] if dicom_paths is not None else None,
            "vessel_mask_path": str(vessel_mask_path) if vessel_mask_path is not None else None,
            "pixels_per_degree": "caller-supplied preparation; expected historical25",
        },
        probability_semantics="two-way artery/vein conditional softmax, multiplied by supplied vessel support",
        class_encoding={"0": "background or artery/vein tie", "1": "artery", "2": "vein"},
        mask_policy="compare artery/vein logits strictly; ignore background logit; gate by supplied vessel",
        threshold_used=False,
        tiling={"mode": "full_image"},
        resampling="nearest classes, vessel support and conditional probabilities; full-frame geometry only",
        overlay_source=overlay_source,
    )
    return _save(
        output,
        image,
        probability,
        channels,
        masks,
        metadata,
        arrays={"conditional_probability": conditional, "vessel_support": vessel},
        classes=classes,
    )
