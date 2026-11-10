import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_fov_source import (
    automatic_fov_source,
    automatic_fov_stages,
    source_rgb2gray,
)
from nelder_mead_candidate import candidate_multithresh
from scipy.io import loadmat

parser = argparse.ArgumentParser(
    description="Compare raw-source FOV stages only, no model inference"
)
parser.add_argument("--oracle-dir", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
rows = []
if args.report.exists():
    raise FileExistsError("Preserve existing comparison")
for c in loadmat(root / "firstcut_real_goldens.mat", simplify_cells=True)["cases"]:
    gray = source_rgb2gray(c["image"])
    candidate = candidate_multithresh(gray, c["n"])
    kwargs = {"dataset": c["dataset"], "threshold_backend": "source_first_cutoff"}
    stages = automatic_fov_stages(c["image"], **kwargs)
    result = automatic_fov_source(c["image"], **kwargs)
    row = {
        "dataset": c["dataset"],
        "shape": list(c["image"].shape),
        "source_first_threshold": float(np.min(c["thresholds"])),
        "candidate_first_threshold": float(candidate["thresholds"].min()),
        "first_threshold_abs_error": float(
            abs(candidate["thresholds"].min() - np.min(c["thresholds"]))
        ),
        "initial_low_abs_error": float(
            abs(candidate["thresholds"].min() * c["factor"] - c["initial_low"])
        ),
        "final_low_abs_error": float(abs(stages["low_threshold"] - c["final_low"])),
        "fallback_matches": bool(stages["fallback_used"] == c["fallback_used"]),
        "stage_disagreements": {
            key: int(np.count_nonzero(stages[key] != c[key]))
            for key in ["initial_closed", "closed", "filled"]
        },
        "boundary_disagreements": int(
            np.count_nonzero(result.calibration_boundary != c["boundary"])
        ),
        "processing_fov_disagreements": int(
            np.count_nonzero(result.processing_fov != c["processing_fov"])
        ),
        "calibration": asdict(result.calibration),
        "source_calibration": [int(c["row"]), int(c["col"]), int(c["radius"])],
        "source_full_threshold_vector": np.asarray(c["thresholds"]).tolist(),
        "candidate_full_threshold_vector": candidate["thresholds"].tolist(),
    }
    rows.append(row)
print(json.dumps(rows, indent=2))
args.report.write_text(
    json.dumps(
        {"cases": rows, "sources": json.loads((root / "firstcut_real_sources.json").read_text())},
        indent=2,
        allow_nan=False,
    )
    + "\n"
)
