"""Versioned full prepared-image photometry; canonical RGB and labels stay raw."""

import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

PROFILES = ("raw", "clahe_hsv_v1", "legacy_lc_clahe_v1")


def contract(profile="raw", pixels_per_degree=25.0):
    if profile not in PROFILES:
        raise ValueError("Unknown unified photometric_profile")
    if profile == "raw":
        return {"profile": "raw"}
    if not math.isfinite(pixels_per_degree) or pixels_per_degree <= 0:
        raise ValueError("Photometry requires positive finite pixels/degree")
    kernel = 250 / (55 / pixels_per_degree)
    if int(kernel) < 1:
        raise ValueError("Photometry CLAHE kernel must be >=1 pixel after truncation")
    return {
        "profile": profile,
        "contract_version": 1,
        "stage": "full legacy25_raw prepared RGB before crop/augmentation/tiling",
        "input": "uint8 RGB HWC",
        "output": "uint8 RGB HWC",
        "outside_fov": "restore original prepared RGB exactly after operator",
        "FOV_provenance": "prepared image-derived/official FOV; canonical photos.json for corpus or prepare_single_net for raw inference; never target validity",
        "corpus_FOV_binding": "schema2 prepared_content_fingerprint includes photos.json and every encoded FOV asset",
        "RGB_scaling": "float32/255; RGB_to_HSV; CLAHE_on_V; HSV_to_RGB",
        "clip_limit": 0.005,
        "nbins": 256,
        "kernel_size_requested": kernel,
        "kernel_size_effective": [int(kernel), int(kernel)],
        "kernel_rounding": "skimage int truncation toward zero; reject below1",
        "uint8_export": "clip to[0,255], truncate toward zero",
        "background": {
            "operator": "legacy Python local_contrast_and_clahe",
            "sigma": 80 / (55 / pixels_per_degree),
            "formula": "clip((RGB-background)/background+0.5,0,1)",
            "boundary": "original eroded-FOV EDT/Sobel/OpenCV remap",
            "Gaussian": "skimage gaussian mode nearest, truncate4, channel_axis2",
        }
        if profile == "legacy_lc_clahe_v1"
        else None,
        "backend_versions": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-image", "opencv-python-headless")
        },
        "matlab_parity_claim": False,
    }


def apply(image, profile="raw", *, fov=None, pixels_per_degree=25.0):
    """Pure deterministic full-image operation. Raw returns its input without work."""
    if profile == "raw":
        return image
    parameters = contract(profile, pixels_per_degree)
    raw = np.asarray(image)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("Photometry requires prepared uint8 RGB HWC")
    domain = np.asarray(fov)
    if domain.dtype != np.bool_ or domain.shape != raw.shape[:2] or not domain.any():
        raise ValueError("Photometry requires a nonempty matching image-derived boolean FOV")
    if profile == "legacy_lc_clahe_v1":
        from .legacy.preprocessing import local_contrast_and_clahe

        processed = local_contrast_and_clahe(raw, domain, pixels_per_degree)
    else:
        from skimage.color import hsv2rgb, rgb2hsv
        from skimage.exposure import equalize_adapthist

        hsv = rgb2hsv(raw.astype(np.float32) / 255)
        hsv[..., 2] = equalize_adapthist(
            hsv[..., 2],
            kernel_size=parameters["kernel_size_requested"],
            clip_limit=parameters["clip_limit"],
            nbins=parameters["nbins"],
        )
        processed = np.clip(hsv2rgb(hsv) * 255, 0, 255).astype(np.uint8)
    if processed.shape != raw.shape or not np.isfinite(processed).all():
        raise ValueError("Invalid photometric operator output")
    return np.ascontiguousarray(np.where(domain[..., None], processed, raw))


def fov_index(root):
    """Only image geometry metadata; never infer FOV from segmentation targets."""
    photos = json.loads((Path(root) / "photos.json").read_text())
    if not isinstance(photos, list):
        raise TypeError("Opt-in photometry requires canonical photos.json")
    result = {}
    for photo in photos:
        pid = photo["photo_id"]
        if pid in result:
            raise ValueError("Duplicate photo_id in canonical FOV metadata")
        result[pid] = photo
    return result


def image_fov(root, photo_id, shape, index=None):
    photo = (fov_index(root) if index is None else index)[photo_id]
    if photo["preprocessing_profile"] != "legacy25_raw" or photo["normalized"] is not False:
        raise ValueError("Photometry requires genuine raw canonical geometry")
    if tuple(photo["prepared_shape"]) != tuple(shape) or photo["pixels_per_degree"] != 25:
        raise ValueError("Canonical RGB/FOV geometry or physical scale differs")
    relative = Path(photo["fov"])
    path = Path(root) / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError("Canonical FOV path escapes corpus")
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L")) > 0
    if mask.shape != shape[:2] or not mask.any():
        raise ValueError("Canonical image-derived FOV is empty or mismatched")
    return mask


def validate_saved(payload, profile, pixels_per_degree=25.0):
    saved = payload.get("preprocessing", {}).get("photometry")
    if (profile != "raw" or saved is not None) and saved != contract(profile, pixels_per_degree):
        raise ValueError(
            "Checkpoint photometry parameters/backend differ; use the locked uv runtime"
        )
