"""Pure reviewed TTA protocol descriptors; no model/data loading or permission."""

import json


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


MODES = ("none", "flips2", "rot4", "dihedral8")


def protocol_extension(mode):
    """Default NONE has exactly the historical schema, without extra fields."""
    if type(mode) is not str or mode not in MODES:
        raise ValueError(f"Unsupported reviewed TTA mode: {mode!r}")
    if mode == "none":
        return {}
    operators = (
        [(0, False), (0, True)]
        if mode == "flips2"
        else [(r, f) for f in (False, True) for r in range(4)]
        if mode == "dihedral8"
        else [(r, False) for r in range(4)]
    )
    return {
        "tta": {
            "version": 1,
            "mode": mode,
            "operators": [
                {"rot90_ccw": r, "horizontal_flip_after_rotation": f} for r, f in operators
            ],
            "domain": "full_prepared_RGB01_before_tiling",
            "inverse": "undo_horizontal_flip_then_inverse_rot90_before_combination",
            "combination": "inverse_align_then_float32_mean_of_vessel_jointAV_disc_cup_then_conditionalAV",
            "precision": "float32_no_autocast",
            "geometry_restoration": "once_after_all_views",
        }
    }


def mode_from_protocol(protocol):
    """Reject unknown/partial/experimental descriptors; never fall back to NONE."""
    if not isinstance(protocol, dict):
        raise TypeError("Missing reviewed inference protocol")
    if "tta" not in protocol:
        return "none"
    descriptor = protocol["tta"]
    if not isinstance(descriptor, dict):
        raise TypeError("Invalid reviewed TTA descriptor")
    mode = descriptor.get("mode")
    expected = protocol_extension(mode)
    if mode == "none" or _encoded(descriptor) != _encoded(expected.get("tta")):
        raise ValueError("Reviewed TTA descriptor does not match exact implementation")
    return mode


def evidence_kind(mode):
    protocol_extension(mode)
    return "unified_validation_evidence" if mode == "none" else "unified_tta_validation_evidence"


def require_metric_extension(metrics, protocol):
    """Require the typed descriptor in actual metrics, including JSON value types."""
    mode = mode_from_protocol(protocol)
    actual = {"tta": metrics["tta"]} if "tta" in metrics else {}
    if _encoded(actual) != _encoded(protocol_extension(mode)):
        raise ValueError("Metric TTA protocol differs from frozen inference")
