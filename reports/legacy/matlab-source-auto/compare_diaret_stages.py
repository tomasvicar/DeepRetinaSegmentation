"""Reproduce measured source stages; no model quality or held-out evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_av import matlab_av_photometric
from deepretina.legacy.matlab_clahe import av_matlab_clahe_v
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
case = loadmat(root / "source_auto_chain.mat", simplify_cells=True)["cases"][2]
source = loadmat(root / "diaret_photometry_stages.mat", simplify_cells=True)
_, python = matlab_av_photometric(case["resized"], case["resized_fov"], return_stages=True)
rows = {}
for key in [
    "distance",
    "gx",
    "gy",
    "x",
    "y",
    "extrapolated",
    "background",
    "corrected",
    "v_before",
    "v_after",
]:
    a, b = python[key], source[key]
    d = np.abs(a.astype(float) - b.astype(float))
    rows[key] = {
        "unequal": int(np.count_nonzero(a != b)),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
    }
same_v = av_matlab_clahe_v(source["v_before"])
d = np.abs(same_v - source["v_after"])
rows["clahe_same_matlab_v"] = {
    "unequal": int(np.count_nonzero(same_v != source["v_after"])),
    "max_abs": float(d.max()),
    "mean_abs": float(d.mean()),
}
a, b = python["v_before"], source["v_before"]
bins_a = np.minimum((a * 256).astype(int), 255)
bins_b = np.minimum((b * 256).astype(int), 255)
changed = bins_a != bins_b
coords = np.argwhere(changed)
rows["python_histogram_bin_changes"] = {
    "count": int(changed.sum()),
    "at_half": int(np.count_nonzero((a >= 0.5) != (b >= 0.5))),
    "examples": [
        {"row": int(y), "col": int(x), "python_v": float(a[y, x]), "matlab_v": float(b[y, x])}
        for y, x in coords[:4]
    ],
}
(root / "diaret-stage-comparison.json").write_text(
    json.dumps(rows, indent=2, allow_nan=False) + "\n"
)
print(json.dumps(rows, indent=2))
