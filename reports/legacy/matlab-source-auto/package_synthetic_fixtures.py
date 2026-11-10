"""Reproduce measured source stages; no model quality or held-out evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
arrays = {}
meta = {"half": [], "chains": []}
for i, c in enumerate(loadmat(root / "half_factor_goldens.mat", simplify_cells=True)["cases"]):
    error = c["error"] if isinstance(c["error"], str) else ""
    row = {
        "name": c["name"],
        "dataset": c["dataset"],
        "initial_low": float(c["initial_low"]),
        "final_low": float(c["final_low"]),
        "source_error": error,
    }
    for field in ["image", "initial_closed", "closed", "filled", "boundary", "processing_fov"]:
        key = f"half{i}_{field}"
        arrays[key] = c[field]
        row[field] = key
    if not error:
        row["calibration"] = [int(c["row"]), int(c["col"]), int(c["radius"])]
    meta["half"].append(row)
for i, c in enumerate(loadmat(root / "source_auto_chain.mat", simplify_cells=True)["cases"]):
    if i < 3:
        continue
    row = {
        "name": c["name"],
        "dataset": c["dataset"],
        "degree": float(c["degree"]),
        "prepared_shape": list(c["resized"].shape),
        "calibration": [int(c["row"]), int(c["col"]), int(c["radius"])],
    }
    for field in [
        "image",
        "vessel",
        "processing_fov",
        "resized_fov",
        "resized_vessel",
        "stored_dicom12",
    ]:
        key = f"chain{i}_{field}"
        arrays[key] = c[field]
        row[field] = key
    ys = np.linspace(0, c["resized"].shape[0] - 1, 9, dtype=int)
    xs = np.linspace(0, c["resized"].shape[1] - 1, 11, dtype=int)
    row["probe_rows"] = ys.tolist()
    row["probe_cols"] = xs.tolist()
    for field in ["resized", "normalized"]:
        key = f"chain{i}_{field}_probes"
        arrays[key] = c[field][ys[:, None], xs[None, :], :]
        row[field + "_probes"] = key
    meta["chains"].append(row)
np.savez_compressed(root / "source-auto-synthetic.npz", **arrays)
meta["fixture_sha256"] = hashlib.sha256(
    (root / "source-auto-synthetic.npz").read_bytes()
).hexdigest()
(root / "source-auto-synthetic-metadata.json").write_text(
    json.dumps(meta, indent=2, allow_nan=False) + "\n"
)
print((root / "source-auto-synthetic.npz").stat().st_size)
