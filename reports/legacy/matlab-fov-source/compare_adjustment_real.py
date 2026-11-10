"""Compare real source geometry/photometry without storing photos in Git."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_av import matlab_av_photometric
from deepretina.legacy.matlab_fov_source import provided_fov_source, rimone_loader_mask
from deepretina.legacy.matlab_resize import imresize_bicubic, imresize_nearest_mask
from deepretina.legacy.preprocessing import quantize_vesclass_rgb
from PIL import Image
from scipy.io import loadmat


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite] - b[finite])
    return {
        "shape": list(a.shape),
        "unequal_values": int(np.count_nonzero(a != b)),
        "max_absolute_error": float(delta.max(initial=0)),
        "mean_absolute_error": float(delta.mean()) if delta.size else None,
        "python_nonfinite": int((~np.isfinite(a)).sum()),
        "matlab_nonfinite": int((~np.isfinite(b)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for c in loadmat(
        args.oracle_dir / "adjustment_real_goldens.mat", simplify_cells=True
    )["cases"]:
        raw = (args.oracle_dir / c["name"]).read_bytes()
        with Image.open(args.oracle_dir / c["name"]) as opened:
            image = np.asarray(opened.convert("RGB"))
        h, w = image.shape[:2]
        dataset = c["dataset"]
        mask = (
            rimone_loader_mask((h, w)) if dataset == "rimone" else np.ones((h, w), bool)
        )
        result = provided_fov_source(mask, dataset=dataset)
        scale = float(
            c["pixels_per_degree"] * c["degree"] / result.calibration.diameter_pixels
        )
        resized = imresize_bicubic(image.astype(np.float64) / 255, scale)
        fov = imresize_nearest_mask(result.processing_fov, resized.shape[:2])
        x, y = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
        pattern = ((x // 37 + y // 29) % 3).astype(np.uint8)
        labels = imresize_nearest_mask(pattern, resized.shape[:2])
        row = {
            "dataset": dataset,
            "input_file_sha256": hashlib.sha256(raw).hexdigest(),
            "input_shape": list(image.shape),
            "scale": scale,
            "center_row_one_based": result.calibration.center_row_one_based,
            "center_col_one_based": result.calibration.center_col_one_based,
            "radius_pixels": result.calibration.radius_pixels,
            "resized_rgb": error(resized, c["resized"]),
            "processing_fov_disagreements": int(
                np.count_nonzero(fov != c["resized_fov"])
            ),
            "categorical_pattern_disagreements": int(
                np.count_nonzero(labels != c["resized_pattern"])
            ),
            "photometry_source_error": c["photometry_error"]
            if isinstance(c["photometry_error"], str)
            else "",
        }
        assert [
            result.calibration.center_row_one_based,
            result.calibration.center_col_one_based,
            result.calibration.radius_pixels,
        ] == [c["row"], c["col"], c["radius"]]
        normalized = matlab_av_photometric(resized, fov)
        row["normalized_rgb"] = error(normalized, c["normalized"])
        row["storage_quantization"] = {}
        for storage in ["dicom12", "uint8"]:
            python_stored = quantize_vesclass_rgb(normalized, storage)
            source_stored = quantize_vesclass_rgb(c["normalized"], storage)
            delta = python_stored.astype(np.int32) - source_stored.astype(np.int32)
            changed = delta != 0
            scale_quant = 4096 if storage == "dicom12" else 255
            source_scaled = np.asarray(c["normalized"])[changed] * scale_quant
            row["storage_quantization"][storage] = {
                "disagreements": int(np.count_nonzero(delta)),
                "max_absolute_difference": int(np.abs(delta).max(initial=0)),
                "scale": scale_quant,
                "changed_source_distance_to_half_integer_max": float(
                    np.max(
                        np.abs(source_scaled - (np.floor(source_scaled) + 0.5)),
                        initial=0,
                    )
                ),
                "first_changed_values": [
                    {"python": float(a), "matlab": float(b)}
                    for a, b in list(
                        zip(
                            normalized[changed],
                            np.asarray(c["normalized"])[changed],
                            strict=True,
                        )
                    )[:3]
                ],
                "rounding": "positive half up, source normalized array independently rounded",
            }
        assert (
            row["processing_fov_disagreements"]
            == row["categorical_pattern_disagreements"]
            == 0
        )
        rows.append(row)
    report = {
        "scope": "Two real source image_adjustment frames plus separately invoked source photometry; no public activation",
        "reference": "7bb0539b90dfce2ed6eaeef1de812072be1155d0, MATLAB R2026b singleCompThread CPU nice10",
        "matlab_parity_certified": False,
        "historical_export_provenance_verified": False,
        "source_helpers_sha256": {
            name: hashlib.sha256((args.oracle_dir / name).read_bytes()).hexdigest()
            for name in [
                "souradnice.m",
                "image_adjustment.m",
                "local_contrast_and_clahe.m",
            ]
        },
        "notes": [
            "7bb source image_adjustment has no active crop; full stereo width retained for RIM",
            "Pattern is synthetic categorical test, not source GT",
            "Loader image_adjustment output and optional LC/CLAHE are separate stages",
        ],
        "cases": rows,
    }
    with args.report.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
