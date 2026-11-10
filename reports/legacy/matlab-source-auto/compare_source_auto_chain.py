"""Reproduce measured source stages; no model quality or held-out evaluation."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_av import matlab_av_photometric
from deepretina.legacy.matlab_resize import imresize_bicubic
from deepretina.legacy.matlab_source_auto import prepare_vesclass_matlab_source_auto
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
rows = []
for c in loadmat(root / "source_auto_chain.mat", simplify_cells=True)["cases"]:
    kwargs = {"dataset": c["dataset"], "field_of_view_degrees": float(c["degree"])}
    prepared = prepare_vesclass_matlab_source_auto(c["image"], c["vessel"], **kwargs)
    source = prepared.source_fov
    scale = 25 * c["degree"] / (2 * source.calibration.radius_pixels)
    resized = imresize_bicubic(c["image"].astype(np.float64) / 255, scale)
    normalized = matlab_av_photometric(resized, prepared.fov)

    def error(a, b):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return {
            "shape": list(a.shape),
            "nonfinite_python": int((~np.isfinite(a)).sum()),
            "nonfinite_matlab": int((~np.isfinite(b)).sum()),
            "unequal": int(np.count_nonzero(a != b)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }

    row = {
        "name": c["name"],
        "dataset": c["dataset"],
        "degree": float(c["degree"]),
        "original_shape": list(c["image"].shape),
        "prepared_shape": list(prepared.image.shape),
        "scale": float(scale),
        "calibration": asdict(source.calibration),
        "processing_fov_disagreements": int(
            np.count_nonzero(source.processing_fov != c["processing_fov"])
        ),
        "resized": error(resized, c["resized"]),
        "normalized": error(normalized, c["normalized"]),
        "stored_dicom12": error(prepared.image, c["stored_dicom12"]),
        "prepared_fov_disagreements": int(np.count_nonzero(prepared.fov != c["resized_fov"])),
        "vessel_disagreements": int(np.count_nonzero(prepared.vessel != c["resized_vessel"])),
        "geometry": asdict(prepared.geometry),
        "profile": prepared.profile,
    }
    assert [
        source.calibration.center_row_one_based,
        source.calibration.center_col_one_based,
        source.calibration.radius_pixels,
    ] == [c["row"], c["col"], c["radius"]]
    rows.append(row)
    print(
        c["name"],
        row["resized"]["max_abs"],
        row["normalized"]["max_abs"],
        row["stored_dicom12"]["unequal"],
        row["stored_dicom12"]["max_abs"],
        flush=True,
    )
report = {
    "scope": "Source auto FOV -> exact7bb image_adjustment -> separately invokedLC -> independent round(normalized*4096) storage contract; not historical exporter replay or model quality",
    "matlab_parity_certified": False,
    "historical_training_file_provenance_verified": False,
    "cases": rows,
    "source_helpers_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ["souradnice.m", "image_adjustment.m", "local_contrast_and_clahe.m"]
    },
}
(root / "source-auto-chain-comparison.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
