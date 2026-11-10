"""Explicit optional MATLAB-source automatic FOV plus calibrated RGB preparation.

No public default is changed. The FOV processing mask and pre-hole-fill
calibration boundary have separate roles; the measured diameter is transferred
explicitly to the existing preparer, without another generic calibration.
"""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from .matlab_av import prepare_vesclass_matlab_calibrated
from .matlab_fov_source import MatlabSourceFOV, automatic_fov_source, provided_fov_source
from .preprocessing import PreparedVesClassProxy, _rgb

# Literal unambiguous degree parameters from user-owned7bb source load_*.m.
# REVIEW uses60/50 and RIGA45/35; require explicit caller angle for those.
_SOURCE_DEGREES = {
    "aria": 50.0,
    "avrdb": 30.0,
    "chasedb1": 30.0,
    "diaretdb": 50.0,
    "drhagis": 45.0,
    "drishtigs": 30.0,
    "drive": 45.0,
    "rite": 45.0,
    "eophtha": 40.0,
    "fire": 45.0,
    "g1020": 45.0,
    "heimed": 45.0,
    "hrf": 60.0,
    "idrid": 50.0,
    "inspireavr": 30.0,
    "iostar": 45.0,
    "messidor": 45.0,
    "onhsd": 45.0,
    "rcrgbma": 45.0,
    "refuge": 45.0,
    "ridb": 45.0,
    "roc": 45.0,
    "stare": 35.0,
    "ubmi": 45.0,
    "uoadr": 45.0,
    "vicavr": 45.0,
}


@dataclass(frozen=True)
class PreparedMatlabSourceAuto(PreparedVesClassProxy):
    """Prepared input plus original-frame source masks for inspection."""

    source_fov: MatlabSourceFOV


def prepare_vesclass_matlab_source(
    image,
    vessel_mask,
    *,
    dataset,
    fov=None,
    fov_diameter_pixels=None,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
    storage="dicom12",
):
    """Prepare raw RGB using explicit source automatic or provided FOV policies.

    Requires a binary original-frame vessel mask; no network/GT reader is called.
    Histogram edges/threshold overrides are deliberately absent. This profile
    fixes the source photometry at25px/degree and leaves historical export
    provenance and universal numerical equivalence uncertified.
    """
    image = _rgb(image)
    vessel = np.asarray(vessel_mask)
    if vessel.shape != image.shape[:2] or not np.isfinite(vessel).all():
        raise ValueError("Provide finite original-frame vessel mask matching RGB")
    if np.issubdtype(vessel.dtype, np.floating) and not np.isin(vessel, [0.0, 1.0]).all():
        raise ValueError("Threshold vessel probabilities explicitly before source-auto preparation")
    if isinstance(pixels_per_degree, (bool, np.bool_)) or float(pixels_per_degree) != 25.0:
        raise ValueError("This source photometric profile requires exactly25pixels/degree")
    if storage not in {"dicom12", "uint8"}:
        raise ValueError("Choose historical storage dicom12 or uint8")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Supply an explicit source dataset name")
    name = dataset.strip().lower().replace("-", "").replace("_", "")
    name = {"drishti": "drishtigs", "diaretdb0": "diaretdb", "diaretdb1": "diaretdb"}.get(
        name, name
    )
    if fov is None and name in {"rim", "rimone", "erlangen", "erlangenvideodata"}:
        raise ValueError(
            "This source loader uses a provided special FOV; automatic composition does not replay that policy"
        )
    if field_of_view_degrees is None:
        if name not in _SOURCE_DEGREES:
            raise ValueError(
                "Supply explicit field_of_view_degrees for unknown or conditional source angle (REVIEW/RIGA)"
            )
        angle = _SOURCE_DEGREES[name]
        angle_source = "literal unambiguous source loader degree parameter"
    else:
        if isinstance(field_of_view_degrees, (bool, np.bool_)):
            raise ValueError("Field angle must be positive and finite")
        angle = float(field_of_view_degrees)
        angle_source = "explicit supplied field angle"
    if not math.isfinite(angle) or angle <= 0:
        raise ValueError("Field angle must be positive and finite")
    if fov_diameter_pixels is not None and (
        isinstance(fov_diameter_pixels, (bool, np.bool_))
        or not math.isfinite(float(fov_diameter_pixels))
        or float(fov_diameter_pixels) <= 0
    ):
        raise ValueError("FOV diameter must be positive and finite")
    if fov is None:
        source = automatic_fov_source(image, dataset=name, threshold_backend="source_first_cutoff")
    else:
        mask = np.asarray(fov)
        if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or not (mask > 0).any():
            raise ValueError("Provide a finite nonempty original-frame FOV matching RGB")
        source = provided_fov_source(mask, dataset=name)
    diameter = (
        source.calibration.diameter_pixels
        if fov_diameter_pixels is None
        else float(fov_diameter_pixels)
    )
    prepared = prepare_vesclass_matlab_calibrated(
        image,
        vessel,
        fov=source.processing_fov,
        fov_diameter_pixels=diameter,
        dataset=name,
        field_of_view_degrees=angle,
        pixels_per_degree=25.0,
        storage=storage,
    )
    profile = {
        **prepared.profile,
        "name": "vesclass_matlab_source_auto" if fov is None else "vesclass_matlab_source_provided",
        "scope": "explicit source FOV branch; resize/photometric residuals and historical provenance qualifications remain",
        "source_ref": "7bb0539b90dfce2ed6eaeef1de812072be1155d0",
        "dataset": name,
        "field_angle_source": angle_source,
        "fov_source": "automatic source souradnice with scalar-zero FOV argument"
        if fov is None
        else "caller-provided mask with explicit source branch",
        "diameter_source": "explicit supplied override"
        if fov_diameter_pixels is not None
        else "source boundary calibration; explicitly transferred without second calibration",
        "fov_calibration": {
            "mode": "estimated_from_source_automatic_fov"
            if fov is None
            else "estimated_from_source_provided_fov",
            "method": source.source_branch,
            **asdict(source.calibration),
        },
        "calibration_boundary_policy": "final_closed before hole filling; source-special rules remain separate"
        if fov is None
        else source.source_branch,
        "used_diameter_pixels": diameter,
        "processing_fov_policy": "source returned processing FOV, distinct from calibration boundary",
        "source_fov_diagnostics": source.diagnostics,
        "generic_recalibration_performed": False,
        "histogram_oracle_override_used": False,
        "historical_training_file_provenance_verified": False,
        "matlab_parity_certified": False,
    }
    profile["source_calibration_boundary_sha256"] = hashlib.sha256(
        source.calibration_boundary.tobytes()
    ).hexdigest()
    profile["source_processing_fov_sha256"] = hashlib.sha256(
        source.processing_fov.tobytes()
    ).hexdigest()
    return PreparedMatlabSourceAuto(
        prepared.image, prepared.vessel, prepared.fov, prepared.geometry, profile, source
    )


def prepare_vesclass_matlab_source_auto(
    image,
    vessel_mask,
    *,
    dataset,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
    storage="dicom12",
):
    """Preserved explicit automatic-only API; never supplies an oracle override."""
    return prepare_vesclass_matlab_source(
        image,
        vessel_mask,
        dataset=dataset,
        field_of_view_degrees=field_of_view_degrees,
        pixels_per_degree=pixels_per_degree,
        storage=storage,
    )
