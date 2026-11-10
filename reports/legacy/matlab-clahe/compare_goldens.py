"""CPU-only stage comparison; real arrays are external and never written to Git.

The temporary photometric candidate below is diagnostic, not a production route.
Existing single_net preprocessing is never edited or persistently patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    sobel,
)
from skimage.color import hsv2rgb, rgb2hsv

from deepretina.legacy import preprocessing as old
from deepretina.legacy.matlab_clahe import av_matlab_clahe_v


def stats(actual, expected, fov):
    difference = np.abs(np.asarray(actual, float) - np.asarray(expected, float))[fov]
    return {
        "mae": float(difference.mean()),
        "p95": float(np.percentile(difference, 95)),
        "max": float(difference.max()),
        "equal_fraction": float((difference == 0).mean()),
        "within_1e_12_fraction": float((difference <= 1e-12).mean()),
    }


def corrected_candidate(image, fov):
    """Provided-FOV, source-parameter photometry with documented Gaussian support."""
    eroded = binary_erosion(
        fov, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), border_value=1
    )
    distance = distance_transform_edt(~eroded).astype(np.float32)
    gx, gy = sobel(distance, axis=1, mode="nearest"), sobel(distance, axis=0, mode="nearest")
    xx, yy = np.meshgrid(
        np.arange(1, distance.shape[1] + 1, dtype=np.float32),
        np.arange(1, distance.shape[0] + 1, dtype=np.float32),
    )
    xx -= gx / 4 * distance
    yy -= gy / 4 * distance
    ext = np.stack(
        [
            map_coordinates(
                image[:, :, channel],
                [yy - 1, xx - 1],
                order=1,
                mode="constant",
                cval=0,
                prefilter=False,
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    sigma = 80 / (55 / 25)
    radius = int(np.ceil(2 * sigma))
    background = gaussian_filter(
        ext, sigma=(sigma, sigma, 0), radius=(radius, radius, 0), mode="nearest"
    )
    return np.clip((ext - background) / background + 0.5, 0, 1), ext, background


def finish(corrected, fov):
    hsv = rgb2hsv(corrected)
    hsv[:, :, 2] = av_matlab_clahe_v(hsv[:, :, 2])
    out = np.clip(hsv2rgb(hsv), 0, 1)
    out[~fov] = 0
    return out, old.quantize_vesclass_rgb(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name, stage_file in [("drive", "drive25_stages.mat"), ("hrf", "hrf25_stages.mat")]:
        golden = loadmat(args.golden_dir / f"{name}_golden.mat")
        stages = loadmat(args.golden_dir / stage_file)
        fov = golden["prepared_fov"].astype(bool)
        all_pixels = np.ones(fov.shape, bool)
        identical_v = av_matlab_clahe_v(stages["v_before"])
        exact_rgb, exact_stored = finish(stages["corrected"], fov)
        with patch.object(
            old, "equalize_adapthist", lambda value, **kwargs: av_matlab_clahe_v(value)
        ):
            existing_photo = old.quantize_vesclass_rgb(
                old._local_contrast_float(golden["resized"], fov, 25)
            )
            proxy = old.prepare_vesclass_raw_proxy(
                golden["raw"],
                np.zeros(golden["fov"].shape, bool),
                dataset=name,
                fov=golden["fov"].astype(bool),
                fov_diameter_pixels=2 * float(golden["radius"].item()),
            )
        corrected, ext, _background = corrected_candidate(golden["resized"], fov)
        _candidate_rgb, candidate_stored = finish(corrected, fov)
        sigma = 80 / (55 / 25)
        radius = int(np.ceil(2 * sigma))
        same_ext_background = gaussian_filter(
            stages["extrapolated"],
            sigma=(sigma, sigma, 0),
            radius=(radius, radius, 0),
            mode="nearest",
        )
        computed_v = rgb2hsv(corrected)[:, :, 2]
        reference_v = stages["v_before"]
        actual_bins = np.floor(computed_v * 255 + 0.5)
        reference_bins = np.floor(reference_v * 255 + 0.5)
        changed = (actual_bins != reference_bins) & fov
        result = {
            "dataset": name,
            "golden_sha256": hashlib.sha256(
                (args.golden_dir / f"{name}_golden.mat").read_bytes()
            ).hexdigest(),
            "stage_sha256": hashlib.sha256((args.golden_dir / stage_file).read_bytes()).hexdigest(),
            "shape": list(fov.shape),
            "v_identical_input_allpixels": stats(identical_v, stages["v_after"], all_pixels),
            "v_identical_input_fov": stats(identical_v, stages["v_after"], fov),
            "identical_corrected_rgb": stats(exact_rgb, golden["normalized"], fov),
            "identical_corrected_stored12": stats(exact_stored, golden["stored"], fov),
            "existing_python_photo_with_new_clahe_stored12": stats(
                existing_photo, golden["stored"], fov
            ),
            "existing_raw_proxy_with_new_clahe_stored12": stats(proxy.image, golden["stored"], fov),
            "provided_resized_candidate_stored12": stats(candidate_stored, golden["stored"], fov),
            "candidate_extrapolated_allpixels": stats(ext, stages["extrapolated"], all_pixels),
            "candidate_corrected_fov": stats(corrected, stages["corrected"], fov),
            "gaussian_same_extrapolated_documented_support": stats(
                same_ext_background, stages["G"], fov
            ),
            "candidate_v_bin_disagreements_fov": int(changed.sum()),
            "candidate_v_bin_disagreement_max_input_delta": float(
                np.max(np.abs(computed_v - reference_v)[changed])
            )
            if changed.any()
            else 0,
            "raw_proxy_fov_disagreements": int((proxy.fov != fov).sum()),
        }
        changed_rgb = (exact_stored != golden["stored"]) & fov[:, :, None]
        scaled = golden["normalized"] * 4096
        half_distance = np.abs(scaled - np.floor(scaled) - 0.5)
        result["identical_corrected_quantization_residual"] = {
            "changed_rgb_values": int(changed_rgb.sum()),
            "maximum_reference_distance_from_half_integer": float(half_distance[changed_rgb].max())
            if changed_rgb.any()
            else 0,
            "all_changes_at_rounding_boundary": bool((half_distance[changed_rgb] < 1e-10).all()),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "matlab_reference": "R2026b Update1, singleCompThread, CUDA hidden",
        "scope": "uniform full-range double V-plane CLAHE; full-chain residuals remain explicitly scoped",
        "no_proprietary_source_read": True,
        "single_net_python_profile_changed": False,
        "production_route_activated": False,
        "full_raw_pipeline_certified": False,
        "goldens": results,
    }
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
