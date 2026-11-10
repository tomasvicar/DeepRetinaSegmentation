"""Input contract for freshly trained native AV models on canonical raw crops.

This profile is separate from the historical VesClass photometry/full-frame
contract. It uses the original network arithmetic with new explicit inputs.
"""

from __future__ import annotations

import numpy as np

from ..geometry_policy import prediction_geometry, validate_geometry_policy
from .preprocessing import prepare_single_net

PROFILE = "legacy25_raw_av4"


def validate_inputs(inputs):
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 4 or min(values.shape[1:]) < 16:
        raise ValueError("Canonical AV requires full-frame 4,H,W input, H/W at least16")
    if not np.isfinite(values).all() or np.any(np.abs(values[:3]) > 0.5):
        raise ValueError("Canonical AV RGB must be finite RGB/255-0.5 in [-0.5,0.5]")
    if not np.array_equal(np.unique(values[3]), np.array([0, 1], np.float32)):
        raise ValueError("Canonical AV requires a binary vessel mask containing both0 and1")
    # A reduction of float32 constants such as0.1 can report a tiny nonzero
    # standard deviation because its computed mean rounds differently.
    if np.any(np.ptp(values, axis=(1, 2)) == 0):
        raise ValueError("Original AV standardization is undefined for a constant channel")
    return values


def prepare_raw(
    rgb,
    vessel,
    payload,
    *,
    dataset=None,
    fov=None,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
):
    """Use exactly canonical RGB geometry and retain unknown-class vessels."""
    preprocessing = payload.get("preprocessing", {})
    if preprocessing.get("profile") != PROFILE:
        raise ValueError("Checkpoint does not declare canonical raw AV preparation")
    saved_ppd = preprocessing.get("pixels_per_degree", 25)
    if saved_ppd != 25 or pixels_per_degree != saved_ppd:
        raise ValueError("Canonical AV requires its saved25 pixels/degree contract")
    if payload.get("geometry_policy") is not None:
        validate_geometry_policy(payload["geometry_policy"])
    rgb, vessel = np.asarray(rgb), np.asarray(vessel)
    if vessel.ndim != 2 or vessel.shape != rgb.shape[:2]:
        raise ValueError("Supply an original-frame single-channel vessel mask matching RGB")
    if not np.isfinite(vessel).all() or np.any(vessel < 0):
        raise ValueError("Vessel values must be finite and nonnegative")
    levels = np.unique(vessel)
    if len(levels) != 2 or levels[0] != 0 or levels[1] <= 0:
        raise ValueError("Vessel mask requires zero background and one positive foreground value")
    angle, provenance = prediction_geometry(
        payload.get("geometry_policy"), dataset, rgb.shape[:2], field_of_view_degrees
    )
    prepared = prepare_single_net(
        rgb,
        dataset=dataset,
        fov=fov,
        field_of_view_degrees=angle,
        pixels_per_degree=saved_ppd,
        normalized=False,
    )
    support = prepared.geometry.prepare_mask(vessel > 0) & prepared.fov
    inputs = np.concatenate((prepared.tensor().numpy(), support[None].astype(np.float32)))
    return prepared, validate_inputs(inputs), provenance
