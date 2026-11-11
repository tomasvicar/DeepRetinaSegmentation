"""One retained TRAIN/image-only raw-source chain, no GT/model forward."""

import argparse
import json
from pathlib import Path

import numpy as np
from candidate_followup import candidate as followup
from candidate_sobel import candidate as first
from nearest_research import resize as nearest
from scipy.io import loadmat

import deepretina.legacy.matlab_av as av
from deepretina.legacy.matlab_source_auto import prepare_vesclass_matlab_source_auto


def error(a, b):
    d = np.abs(a.astype(float) - b.astype(float))
    return {
        "unequal": int(np.count_nonzero(a != b)),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    case = loadmat(a.root / "source_auto_chain.mat", simplify_cells=True)["cases"][2]
    baseline = av._sobel_single
    rows = {}
    for name, operator in [
        ("production", baseline),
        ("research_v1", first),
        ("research_v2", followup),
        ("nearest_only", baseline),
        ("nearest_and_sobel", followup),
    ]:
        av._sobel_single = operator
        if name.startswith("nearest"):
            av.imresize_nearest_mask = nearest
        result = prepare_vesclass_matlab_source_auto(
            case["image"],
            case["vessel"],
            dataset=case["dataset"],
            field_of_view_degrees=float(case["degree"]),
        )
        rows[name] = {
            "stored12": error(result.image, case["stored_dicom12"]),
            "fov_unequal": int(np.count_nonzero(result.fov != case["resized_fov"])),
            "processing_fov_unequal": int(
                np.count_nonzero(result.source_fov.processing_fov != case["processing_fov"])
            ),
            "calibration_radius": float(result.source_fov.calibration.radius_pixels),
            "calibration_radius_reference": float(case["radius"]),
        }
    av._sobel_single = baseline
    a.output.write_text(
        json.dumps(
            {
                "scope": "one retained TRAIN/image-only raw automatic preparation; no labels/model/quality score",
                "case": str(case["name"]),
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
