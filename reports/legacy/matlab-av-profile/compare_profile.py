"""Compare the explicit Python profile to external MATLAB goldens, CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skimage.color import hsv2rgb, rgb2hsv

from deepretina.legacy.matlab_av import matlab_av_photometric, prepare_vesclass_matlab_calibrated
from deepretina.legacy.matlab_clahe import av_matlab_clahe_v
from deepretina.legacy.matlab_resize import imresize_bicubic
from deepretina.legacy.preprocessing import quantize_vesclass_rgb


def metrics(a, b, fov):
    difference = np.abs(np.asarray(a, float) - np.asarray(b, float))[fov]
    return {
        "mae": float(difference.mean()),
        "p95": float(np.percentile(difference, 95)),
        "max": float(difference.max()),
        "equal_fraction": float((difference == 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name in ["drive", "hrf", "drive_35"]:
        filename = args.golden_dir / f"{name}_golden.mat"
        g = loadmat(filename)
        fov = g["prepared_fov"].astype(bool)
        ppd = 35 if name == "drive_35" else 25
        dataset = "hrf" if name == "hrf" else "drive"
        raw = g["raw"]
        vessel = np.indices(raw.shape[:2]).sum(axis=0) % 2 == 0
        prepared = prepare_vesclass_matlab_calibrated(
            raw,
            vessel,
            fov=g["fov"].astype(bool),
            fov_diameter_pixels=2 * float(g["radius"].item()),
            dataset=dataset,
            pixels_per_degree=ppd,
        )
        # Save only to external scratch for the next independent model-forward gate.
        cache = args.golden_dir / f"{name}_python_profile.npz"
        np.savez_compressed(cache, stored_rgb=prepared.image, fov=prepared.fov)
        resized = imresize_bicubic(raw.astype(np.float64) / 255, prepared.geometry.rescale_factor)
        normalized, stages = matlab_av_photometric(g["resized"], fov, return_stages=True)
        row = {
            "name": name,
            "golden_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
            "profile_rgb_sha256_c_order_uint16_le": hashlib.sha256(
                prepared.image.astype("<u2").tobytes()
            ).hexdigest(),
            "shape": list(prepared.image.shape),
            "fov_disagreements": int((prepared.fov != fov).sum()),
            "resized_rgb_fov": metrics(resized, g["resized"], fov),
            "photometric_from_reference_resized_stored12": metrics(
                quantize_vesclass_rgb(normalized), g["stored"], fov
            ),
            "full_raw_stored12": metrics(prepared.image, g["stored"], fov),
            "profile": prepared.metadata,
        }
        if name != "drive_35":
            coords = loadmat(args.golden_dir / f"{name}_coords.mat")
            reference = loadmat(
                args.golden_dir / ("drive25_stages.mat" if name == "drive" else "hrf25_stages.mat")
            )
            all_pixels = np.ones_like(fov)
            row["stages"] = {
                field: metrics(stages[field], coords[ref], all_pixels)
                for field, ref in [
                    ("distance", "dt"),
                    ("gx", "sx"),
                    ("gy", "sy"),
                    ("x", "xxx"),
                    ("y", "yyy"),
                ]
            }
            row["stages"]["extrapolated_allpixels"] = metrics(
                stages["extrapolated"], reference["extrapolated"], all_pixels
            )
            row["stages"]["background_fov"] = metrics(stages["background"], reference["G"], fov)
            row["stages"]["corrected_fov"] = metrics(
                stages["corrected"], reference["corrected"], fov
            )
            changed = (
                np.floor(stages["v_before"] * 255 + 0.5)
                != np.floor(reference["v_before"] * 255 + 0.5)
            ) & fov
            row["v_bin_disagreements_fov"] = int(changed.sum())
            # Controlled injection localizes downstream amplification to V binning.
            hsv = rgb2hsv(stages["corrected"])
            hsv[:, :, 2] = av_matlab_clahe_v(reference["v_before"])
            control = np.clip(hsv2rgb(hsv), 0, 1)
            control[~fov] = 0
            row["reference_v_injection_stored12"] = metrics(
                quantize_vesclass_rgb(control), g["stored"], fov
            )
        results.append(row)
        print(json.dumps(row), flush=True)
    synthetic = []
    for case in loadmat(args.golden_dir / "synthetic_photo.mat", simplify_cells=True)["cases"]:
        fov = case["fov"].astype(bool)
        actual = matlab_av_photometric(case["input"], fov)
        synthetic.append(
            {
                "name": case["name"],
                "normalized_rgb": metrics(actual, case["normalized"], fov),
                "stored12": metrics(quantize_vesclass_rgb(actual), case["stored"], fov),
            }
        )
    report = {
        "scope": "Explicit providedFOV/diameter calibrated MATLAB-style AV profile; numerical agreement is not universal bitwise parity",
        "runtime_reference": "MATLAB R2026b Update1 singleCompThread CUDA hidden",
        "production_route_activated": False,
        "full_pipeline_certified": False,
        "historical_training_file_provenance_verified": False,
        "real_goldens": results,
        "synthetic_photometry": synthetic,
        "no_ad_hoc_epsilon_or_bin_snapping": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
