"""Explicit provided-FOV/calibrated MATLAB-style raw AV preprocessing.

This separate profile retains full-frame geometry and leaves the original
single_net normalized Python preprocessing unchanged. Reference agreement and
remaining precision differences are recorded in docs/MATLAB_AV_PROFILE.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, map_coordinates
from skimage.color import hsv2rgb, rgb2hsv

from .matlab_clahe import av_matlab_clahe_v
from .matlab_fov import calibrate_provided_fov
from .matlab_resize import imresize_bicubic, imresize_nearest_mask
from .preprocessing import (
    DATASET_FOV_DEGREES,
    ImageGeometry,
    PreparedVesClassProxy,
    _mask,
    _rgb,
    quantize_vesclass_rgb,
)


def _sobel_single(distance):
    """Unnormalized Sobel stencils with explicit float32 accumulation order."""
    values = np.pad(distance, 1, mode="edge")
    gx = -values[2:, :-2]
    gx -= 2 * values[1:-1, :-2]
    gx -= values[:-2, :-2]
    gx += values[2:, 2:]
    gx += 2 * values[1:-1, 2:]
    gx += values[:-2, 2:]
    gy = values[2:, :-2].copy()
    gy -= values[:-2, :-2]
    gy += 2 * values[2:, 1:-1]
    gy -= 2 * values[:-2, 1:-1]
    gy += values[2:, 2:]
    gy -= values[:-2, 2:]
    return gx, gy


def matlab_av_photometric(image, fov, *, return_stages=False):
    """Normalize already-resized double RGB with fixed25px source parameters.

    Gaussian support is2*ceil(2*sigma)+1 with replicate borders. Inputs may retain
    bicubic overshoot; clipping happens only after illumination correction. The
    provided FOV must correspond exactly to this resized full frame.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite double RGB HWC")
    fov = _mask(fov, image.shape[:2])
    eroded = binary_erosion(
        fov, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), border_value=1
    )
    distance = distance_transform_edt(~eroded).astype(np.float32)
    gx, gy = _sobel_single(distance)
    x, y = np.meshgrid(
        np.arange(1, image.shape[1] + 1, dtype=np.float32),
        np.arange(1, image.shape[0] + 1, dtype=np.float32),
    )
    x -= gx / 4 * distance
    y -= gy / 4 * distance
    extrapolated = np.stack(
        [
            map_coordinates(
                image[:, :, c], [y - 1, x - 1], order=1, mode="constant", cval=0, prefilter=False
            )
            for c in range(3)
        ],
        axis=-1,
    )
    sigma = 80 / (55 / 25)
    radius = math.ceil(2 * sigma)
    background = gaussian_filter(
        extrapolated, sigma=(sigma, sigma, 0), radius=(radius, radius, 0), mode="nearest"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.clip((extrapolated - background) / background + 0.5, 0, 1)
    if not np.isfinite(corrected).all():
        raise ValueError("Illumination correction produced zero/invalid background")
    hsv = rgb2hsv(corrected)
    value_before = hsv[:, :, 2].copy() if return_stages else None
    hsv[:, :, 2] = av_matlab_clahe_v(hsv[:, :, 2])
    normalized = np.clip(hsv2rgb(hsv), 0, 1)
    normalized[~fov] = 0
    if return_stages:
        return normalized, {
            "distance": distance,
            "gx": gx,
            "gy": gy,
            "x": x,
            "y": y,
            "extrapolated": extrapolated,
            "background": background,
            "corrected": corrected,
            "v_before": value_before,
            "v_after": hsv[:, :, 2],
        }
    return normalized


def prepare_vesclass_matlab_calibrated(
    image,
    vessel_mask,
    *,
    fov,
    fov_diameter_pixels=None,
    dataset=None,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
    storage="dicom12",
):
    """Prepare raw RGB with provided FOV and explicit or generic source calibration.

    An explicit diameter takes priority; otherwise the verified generic provided-
    mask souradnice estimator is used. Special source branches fail closed. No
    automatic-threshold FOV or bbox-radius substitute is introduced. Historical
    training-file provenance and arbitrary raw images are not certified here.
    """
    image = _rgb(image)
    fov = _mask(fov, image.shape[:2])
    vessel = np.asarray(vessel_mask)
    if vessel.shape != fov.shape or not np.isfinite(vessel).all():
        raise ValueError("Provide a finite original-frame vessel mask matching RGB")
    if np.issubdtype(vessel.dtype, np.floating) and not np.isin(vessel, [0.0, 1.0]).all():
        raise ValueError("Threshold vessel probabilities explicitly before AV preparation")
    vessel = vessel > 0
    if field_of_view_degrees is None:
        if dataset is None or dataset.lower() not in DATASET_FOV_DEGREES:
            raise ValueError("Supply a known dataset or explicit field_of_view_degrees")
        field_of_view_degrees = DATASET_FOV_DEGREES[dataset.lower()]
    if fov_diameter_pixels is None:
        calibration = calibrate_provided_fov(fov, dataset=dataset)
        fov_diameter_pixels = calibration.diameter_pixels
        calibration_info = {
            "mode": "estimated_from_provided_fov",
            "method": "souradnice_generic_binary_sobel",
            **asdict(calibration),
        }
        diameter_source = "provided FOV generic souradnice estimate"
    else:
        calibration_info = {
            "mode": "explicit_diameter",
            "method": "caller_supplied_diameter",
            "center_row_one_based": None,
            "center_col_one_based": None,
            "radius_pixels": float(fov_diameter_pixels) / 2,
        }
        diameter_source = "explicit supplied calibration; no heuristic substitute"
    if not all(
        math.isfinite(v) and v > 0
        for v in (pixels_per_degree, field_of_view_degrees, fov_diameter_pixels)
    ):
        raise ValueError("Pixels/degree, field angle and diameter must be positive/finite")
    scale = pixels_per_degree * field_of_view_degrees / fov_diameter_pixels
    resized = imresize_bicubic(image.astype(np.float64) / 255, scale)
    prepared_fov = imresize_nearest_mask(fov, resized.shape[:2])
    prepared_vessel = imresize_nearest_mask(vessel, resized.shape[:2])
    normalized = matlab_av_photometric(resized, prepared_fov)
    stored = quantize_vesclass_rgb(normalized, storage)
    h, w = image.shape[:2]
    geometry = ImageGeometry((h, w), (0, h - 1, 0, w - 1), float(scale), tuple(resized.shape[:2]))
    profile = {
        "name": "vesclass_matlab_provided_fov_calibrated",
        "matlab_parity_certified": False,
        "reference_runtime": "R2026b Update1",
        "scope": "provided FOV with explicit or verified generic calibration; numerical residuals remain",
        "geometry": "full frame; no bbox crop; scalar-scale bicubic with antialiasing; ceil shape",
        "pixels_per_degree": float(pixels_per_degree),
        "photometry_pixels_per_degree": 25.0,
        "field_of_view_degrees": float(field_of_view_degrees),
        "fov_diameter_pixels": float(fov_diameter_pixels),
        "fov_source": "provided original-frame FOV",
        "diameter_source": diameter_source,
        "fov_calibration": calibration_info,
        "mask_resampling": "nearest to explicit resized image dimensions",
        "storage": storage,
        "historical_training_file_provenance_verified": False,
        "original_single_net_python_profile_changed": False,
    }
    return PreparedVesClassProxy(stored, prepared_vessel, prepared_fov, geometry, profile)
