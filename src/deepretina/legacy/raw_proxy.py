"""Explicit raw-photograph convenience path for the original A/V network.

The network port is exact. This Python preprocessing surrogate is not certified
against the historical MATLAB/DICOM exports, and is never selected silently.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..bundled import bundled_checkpoint
from .inference import predict_legacy
from .preprocessing import prepare_vesclass_raw_proxy


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
):
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fov = None
    if fov_path:
        with Image.open(fov_path) as image:
            fov = np.array(image.convert("L")) > 0
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
    prepared = prepare_vesclass_raw_proxy(
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
    metadata["raw_av_proxy"] = prepared.metadata
    metadata["raw_av_proxy"]["matlab_parity_certified"] = False
    metadata["vessel_source"] = (
        str(vessel_mask_path) if vessel_mask_path else "original single_net vessel checkpoint"
    )
    metadata["vessel_prediction"] = vessel_metadata
    metadata["outputs"] = paths
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return paths
