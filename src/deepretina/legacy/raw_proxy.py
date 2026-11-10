"""Explicit raw-photograph convenience path for the original A/V network.

The default Python surrogate and explicit calibrated MATLAB-style profile have
separate numerical qualifications. Neither is selected silently or certified as
universal bitwise parity with historical MATLAB/DICOM exports.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..bundled import bundled_checkpoint
from .inference import predict_legacy
from .preprocessing import DATASET_FOV_DEGREES, prepare_vesclass_raw_proxy


def predict_raw_av(
    checkpoint,
    image_path,
    output,
    *,
    device="auto",
    dataset=None,
    fov_path=None,
    field_of_view_degrees=None,
    vessel_mask_path=None,
    vessel_checkpoint=None,
    vessel_normalized=False,
    fov_diameter_pixels=None,
    pixels_per_degree=25.0,
    threshold=0.5,
    preprocessing_profile="proxy",
):
    if preprocessing_profile not in {"proxy", "matlab"}:
        raise ValueError("Choose raw A/V preprocessing_profile=proxy or matlab")
    if preprocessing_profile == "matlab":
        if fov_path is None:
            raise ValueError("--av-raw-matlab requires provided --fov")
        if fov_diameter_pixels is not None and (
            not math.isfinite(fov_diameter_pixels) or fov_diameter_pixels <= 0
        ):
            raise ValueError("--av-raw-matlab requires positive finite --fov-diameter-pixels")
        angle = field_of_view_degrees
        if angle is None:
            angle = DATASET_FOV_DEGREES.get((dataset or "").lower())
        if angle is None or not math.isfinite(angle) or angle <= 0:
            raise ValueError(
                "--av-raw-matlab requires a known --dataset or positive finite --field-of-view-degrees"
            )
        if not math.isfinite(pixels_per_degree) or pixels_per_degree <= 0:
            raise ValueError("--pixels-per-degree must be positive and finite")
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    output = Path(output)
    fov = None
    if fov_path:
        with Image.open(fov_path) as image:
            fov = np.array(image.convert("L")) > 0
    if fov is not None and fov.shape != rgb.shape[:2]:
        raise ValueError("Original-frame FOV and RGB image dimensions differ")
    if preprocessing_profile == "matlab":
        if not fov.any():
            raise ValueError("Provided --fov must contain foreground pixels")
        diameter = fov_diameter_pixels
        if diameter is None:
            from .matlab_fov import calibrate_provided_fov

            diameter = calibrate_provided_fov(fov, dataset=dataset).diameter_pixels
        scale = pixels_per_degree * angle / diameter
        shape = np.ceil(np.asarray(rgb.shape[:2]) * scale)
        if np.any(np.floor(shape / (250 / (55 / 25)) + 0.5) < 2):
            raise ValueError(
                "Calibrated MATLAB AV image is too small for the source CLAHE tile grid"
            )
    vessel_metadata = None
    if vessel_mask_path:
        with Image.open(vessel_mask_path) as image:
            vessel = np.array(image.convert("L")) > 0
    else:
        with tempfile.TemporaryDirectory(prefix="deepretina-vessel-") as temporary:

            def infer(source):
                return predict_legacy(
                    source,
                    image_path,
                    Path(temporary),
                    device=device,
                    dataset=dataset,
                    fov_path=fov_path,
                    field_of_view_degrees=field_of_view_degrees,
                    pixels_per_degree=pixels_per_degree,
                    normalized=vessel_normalized,
                    threshold=threshold,
                )

            if vessel_checkpoint:
                paths = infer(vessel_checkpoint)
            else:
                with bundled_checkpoint("legacy-vessels") as source:
                    paths = infer(source)
            if "vessel" not in paths:
                raise ValueError("The supplied segmentation model has no vessel output")
            with Image.open(paths["vessel"]) as image:
                vessel = np.array(image) > 0
            vessel_metadata = json.loads(Path(paths["metadata"]).read_text())
    if vessel.shape != rgb.shape[:2]:
        raise ValueError("Original-frame vessel mask and RGB image dimensions differ")
    prepare = prepare_vesclass_raw_proxy
    if preprocessing_profile == "matlab":
        from .matlab_av import prepare_vesclass_matlab_calibrated

        prepare = prepare_vesclass_matlab_calibrated
    prepared = prepare(
        rgb,
        vessel,
        dataset=dataset,
        fov=fov,
        field_of_view_degrees=field_of_view_degrees,
        fov_diameter_pixels=fov_diameter_pixels,
        pixels_per_degree=pixels_per_degree,
    )
    paths = predict_legacy(
        checkpoint,
        image_path,
        output,
        device=device,
        prepared_inputs=prepared.inputs(),
        original_size=rgb.shape[:2],
    )
    vessel_path = output / "vessel.png"
    Image.fromarray(vessel.astype(np.uint8) * 255).save(vessel_path)
    paths["vessel"] = str(vessel_path)
    metadata_path = Path(paths["metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["raw_av_preprocessing"] = prepared.metadata
    metadata["raw_av_geometry"] = asdict(prepared.geometry)
    metadata["preprocessing"]["raw_profile"] = prepared.metadata["name"]
    if preprocessing_profile == "proxy":
        metadata["raw_av_proxy"] = prepared.metadata
    metadata["raw_av_preprocessing"]["matlab_parity_certified"] = False
    metadata["vessel_source"] = (
        str(vessel_mask_path) if vessel_mask_path else "original single_net vessel checkpoint"
    )
    metadata["vessel_prediction"] = vessel_metadata
    metadata["outputs"] = paths
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return paths
