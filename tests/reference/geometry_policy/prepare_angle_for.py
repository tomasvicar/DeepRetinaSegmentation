"""Unmodified project-owned builder selection, fa1cf0336410676020bd4f05eb8e6d545ad3ce88."""
from copy import deepcopy
import math

def angle_for(policy, dataset, shape):
    """Select exact native camera shape before the explicitly nominal fallback."""
    if dataset not in policy["policies"]:
        raise ValueError("No explicit reviewed angle for dataset: " + dataset)
    entry = deepcopy(policy["policies"][dataset])
    shape_key = f"{shape[0]}x{shape[1]}"
    if "per_shape" in entry:
        matched = shape_key in entry["per_shape"]
        entry = deepcopy(entry["per_shape"][shape_key] if matched else entry["default"])
        attribution = "native_shape_rule" if matched else "explicit_unseen_shape_nominal_fallback"
    else:
        attribution = "dataset_acquisition_description_or_explicit_nominal_research_policy"
    angle = entry["field_of_view_degrees"]
    if isinstance(angle, bool) or not math.isfinite(float(angle)) or not 0 < float(angle) <= 180:
        raise ValueError("Reviewed field angle must be finite, positive and explicit")
    entry.update(
        policy_id=policy["policy_id"],
        native_shape_hw=list(shape[:2]),
        camera_attribution_method=attribution,
    )
    return entry
