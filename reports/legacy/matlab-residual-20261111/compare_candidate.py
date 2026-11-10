"""One existing TRAIN/image-only photometry case; no GT/model/quality score."""

import argparse
import json
from pathlib import Path

import numpy as np
from candidate_sobel import candidate
from scipy.io import loadmat

import deepretina.legacy.matlab_av as av
from deepretina.legacy.matlab_resize import imresize_bicubic


def error(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "unequal": int(np.count_nonzero(a != b)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    cases = list(loadmat(a.root / "sobel_operator_goldens.mat", simplify_cells=True)["cases"])
    stage = loadmat(a.root / "diaret_photometry_stages.mat", simplify_cells=True)
    cases.append(dict(name="diaret", **{k: stage[k] for k in ["distance", "gx", "gy"]}))
    operators = []
    for c in cases:
        x, y = candidate(c["distance"])
        operators.append({"name": c["name"], "gx": error(x, c["gx"]), "gy": error(y, c["gy"])})
    case = loadmat(a.root / "source_auto_chain.mat", simplify_cells=True)["cases"][2]
    reference = av._sobel_single
    results = {}
    python_resized = imresize_bicubic(
        case["image"].astype(np.float64) / 255,
        25 * float(case["degree"]) / (2 * float(case["radius"])),
    )
    variants = [
        ("old", reference, case["resized"]),
        ("candidate", candidate, case["resized"]),
        ("old_python_resized", reference, python_resized),
        ("candidate_python_resized", candidate, python_resized),
    ]
    for name, op, resized in variants:
        av._sobel_single = op
        normalized, python = av.matlab_av_photometric(
            resized, case["resized_fov"], return_stages=True
        )
        rows = {
            key: error(python[key], stage[key])
            for key in ["gx", "gy", "x", "y", "extrapolated", "background", "v_before", "v_after"]
        }
        rows["normalized"] = error(normalized, case["normalized"])
        rows["stored12"] = error(
            np.floor(normalized * 4096 + 0.5).astype(np.uint16), case["stored_dicom12"]
        )
        results[name] = rows
    av._sobel_single = reference
    result = {
        "scope": "isolated research operator model on same reference-resized TRAIN/image-only input; no runtime/source edit, no model forward, no quality claim",
        "operators": operators,
        "python_resized_input_error": error(python_resized, case["resized"]),
        "diaret": results,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
