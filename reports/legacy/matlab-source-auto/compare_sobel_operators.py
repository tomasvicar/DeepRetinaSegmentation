"""Reproduce measured source stages; no model quality or held-out evaluation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from deepretina.legacy.matlab_av import _sobel_single
from scipy.io import loadmat
from scipy.ndimage import correlate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
cases = list(loadmat(root / "sobel_operator_goldens.mat", simplify_cells=True)["cases"])
z = loadmat(
    root / "diaret_photometry_stages.mat",
    simplify_cells=True,
    variable_names=["distance", "gx", "gy"],
)
cases.append({"name": "diaret", **z})
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
ky = kx.T
rows = []
for c in cases:
    d = c["distance"]
    p = np.pad(d, 1, mode="edge")
    double = (
        correlate(d.astype(np.float64), kx, mode="nearest").astype(np.float32),
        correlate(d.astype(np.float64), ky, mode="nearest").astype(np.float32),
    )
    dx = p[:, 2:] - p[:, :-2]
    dy = p[2:, :] - p[:-2, :]
    diff_smooth = (dx[:-2] + 2 * dx[1:-1] + dx[2:], dy[:, :-2] + 2 * dy[:, 1:-1] + dy[:, 2:])
    vertical = p[:-2] + 2 * p[1:-1] + p[2:]
    horizontal = p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]
    smooth_diff = (vertical[:, 2:] - vertical[:, :-2], horizontal[2:] - horizontal[:-2])
    candidates = {
        "legacy": _sobel_single(d),
        "double_once": double,
        "difference_then_smooth": diff_smooth,
        "smooth_then_difference": smooth_diff,
        "opencv": (
            cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
            cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE),
        ),
        "scipy_float32": (
            correlate(d, kx, output=np.float32, mode="nearest"),
            correlate(d, ky, output=np.float32, mode="nearest"),
        ),
    }
    row = {"name": c["name"], "shape": list(d.shape), "methods": {}}
    for name, (gx, gy) in candidates.items():
        row["methods"][name] = {
            "gx_unequal": int(np.count_nonzero(gx != c["gx"])),
            "gy_unequal": int(np.count_nonzero(gy != c["gy"])),
            "gx_max_abs": float(np.abs(gx - c["gx"]).max()),
            "gy_max_abs": float(np.abs(gy - c["gy"]).max()),
        }
    rows.append(row)
    print(c["name"], {k: (v["gx_unequal"], v["gy_unequal"]) for k, v in row["methods"].items()})
(root / "sobel-operator-comparison.json").write_text(
    json.dumps({"cases": rows}, indent=2, allow_nan=False) + "\n"
)
