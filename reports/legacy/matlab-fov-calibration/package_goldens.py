"""Package synthetic-only oracle arrays and report real-mask calibration errors."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from deepretina.legacy.matlab_fov import binary_sobel_edges, calibrate_provided_fov


def compare(case):
    edges, stages = binary_sobel_edges(case["mask"], return_stages=True)
    result = {
        "name": case["name"],
        "shape": list(edges.shape),
        "mask_sha256": hashlib.sha256(np.asarray(case["mask"], bool).tobytes()).hexdigest(),
        "edge_disagreements": int(np.count_nonzero(edges != case["edges"])),
        "unthinned_disagreements": int(np.count_nonzero(stages["unthinned"] != case["unthinned"])),
        "threshold_error": float(abs(stages["threshold"] - case["threshold"])),
        "gx_max_error": float(np.max(abs(stages["gx"] - case["gx"]))),
        "gy_max_error": float(np.max(abs(stages["gy"] - case["gy"]))),
        "reference_center_row": float(case["center_row"]),
        "reference_center_col": float(case["center_col"]),
        "reference_radius": float(case["radius"]),
        "reference_error": case["error"] if isinstance(case["error"], str) else "",
    }
    for key in ["reference_center_row", "reference_center_col", "reference_radius"]:
        if not np.isfinite(result[key]):
            result[key] = None
    try:
        result["python"] = asdict(calibrate_provided_fov(case["mask"]))
    except ValueError as exc:
        result["python_error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    cases = loadmat(args.golden_dir / "fov_synthetic_goldens.mat", simplify_cells=True)["cases"]
    arrays, metadata = {}, []
    for i, case in enumerate(cases):
        row = {
            k: case[k] for k in ["name", "threshold", "center_row", "center_col", "radius", "error"]
        }
        if not isinstance(row["error"], str):
            row["error"] = ""
        for key in ["mask", "edges", "unthinned", "gx", "gy"]:
            name = f"{i}_{key}"
            arrays[name] = case[key]
            row[key] = name
        for key in ["center_row", "center_col", "radius"]:
            if not np.isfinite(row[key]):
                row[key] = None
        metadata.append(row)
    np.savez_compressed(args.fixture_dir / "synthetic.npz", **arrays)
    fixture_sha = hashlib.sha256((args.fixture_dir / "synthetic.npz").read_bytes()).hexdigest()
    (args.fixture_dir / "metadata.json").write_text(
        json.dumps(
            {
                "runtime": "MATLAB R2026b Update1 singleCompThread",
                "fixture_sha256": fixture_sha,
                "cases": metadata,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    real = []
    for name in ["drive", "hrf"]:
        g = loadmat(args.golden_dir / f"{name}_fov_goldens.mat")
        case = {
            "name": name,
            "mask": g["fov"],
            "edges": g["e"],
            "unthinned": g["u"],
            "gx": g["gx"],
            "gy": g["gy"],
            "threshold": g["t"].item(),
            "center_row": g["row"].item(),
            "center_col": g["col"].item(),
            "radius": g["radius"].item(),
            "error": "",
        }
        real.append(compare(case))
    report = {
        "scope": "Provided binary FOV generic souradnice branch, no automatic FOV extraction or production activation",
        "runtime": "MATLAB R2026b Update1, CPU nice5 singleCompThread",
        "original_helper_ref": "7bb0539b90dfce2ed6eaeef1de812072be1155d0:matlab_preprocessing/souradnice.m",
        "original_helper_sha256": hashlib.sha256(
            (args.golden_dir / "souradnice.m").read_bytes()
        ).hexdigest(),
        "synthetic": [compare(c) for c in cases],
        "real": real,
        "production_activated": False,
        "historical_training_file_provenance_verified": False,
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
