"""Lazy, predict-only defaults for explicitly named bundled unified models."""

import math
from pathlib import Path

from .bundled import bundled_checkpoint, model_catalog

OPTIONS = ("tta", "threshold", "tile_size", "overlap")
MODES = ("none", "flips2", "rot4", "dihedral8")


def resolve_prediction_options(model=None, *, registry=None, **overrides):
    """Resolve unset options without loading weights; None means inherit/default.

    Explicit values (including tta='none' and threshold=0.5) always win. Defaults
    apply only to a bundled unified model, never evaluation or custom weights.
    """
    if set(overrides) - set(OPTIONS):
        raise ValueError("Unknown inference option")
    registry = model_catalog() if registry is None else registry
    entry = None
    if model is not None:
        name = registry.get("aliases", {}).get(model, model)
        if name not in registry["models"]:
            raise ValueError(f"Unknown bundled model {model!r}")
        entry = registry["models"][name]
    values = {"tta": "none", "threshold": 0.5, "tile_size": None, "overlap": None}
    settings = (entry or {}).get("inference_defaults", {})
    if "inference_defaults" in (entry or {}):
        if entry.get("family") != "unified":
            raise ValueError("Inference defaults require a bundled unified model")
        if not isinstance(settings, dict) or set(settings) - set(OPTIONS):
            raise ValueError("Invalid inference default keys")
        # Malformed stored values must not be hidden by explicit overrides.
        _validate({**values, **settings})
        values.update(settings)
    values.update({key: value for key, value in overrides.items() if value is not None})
    if (entry or {}).get("family") == "unified":
        _validate(values)
    return values


def _validate(values):
    threshold = values["threshold"]
    if values["tta"] not in MODES:
        raise ValueError("Invalid unified TTA mode")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("Unified threshold must be finite and in (0,1)")
    tile, overlap = values["tile_size"], values["overlap"]
    if tile is not None and (type(tile) is not int or tile < 8):
        raise ValueError("Tile size must be an integer >=8")
    if overlap is not None and (type(overlap) is not int or overlap < 0):
        raise ValueError("Overlap must be a nonnegative integer")
    if tile is not None and overlap is not None and overlap >= tile:
        raise ValueError("Overlap must be smaller than tile size")


def predict_bundled_unified(
    model,
    image_path,
    output,
    *,
    tta=None,
    threshold=None,
    tile_size=None,
    overlap=None,
    **kwargs,
):
    """Predict with a bundled unified model and the same options as the CLI.

    Input is an unprocessed RGB photograph path; fov, if provided, is an array.
    Saved camera/photometry/tiling and ordinary prediction metadata stay intact.
    """
    registry = model_catalog()
    name = registry.get("aliases", {}).get(model, model)
    if name not in registry["models"] or registry["models"][name].get("family") != "unified":
        raise ValueError("This API requires a bundled unified model")
    options = resolve_prediction_options(
        model,
        registry=registry,
        tta=tta,
        threshold=threshold,
        tile_size=tile_size,
        overlap=overlap,
    )
    from .unified import predict_unified

    with bundled_checkpoint(model) as checkpoint:
        return predict_unified(checkpoint, Path(image_path), Path(output), **options, **kwargs)
