"""Package synthetic-only threshold references; real images remain external."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
parser.add_argument("--fixture-dir", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
output = args.fixture_dir
output.mkdir(parents=True, exist_ok=True)
for name in ("multithresh-synthetic.npz", "multithresh-metadata.json"):
    if (output / name).exists():
        raise FileExistsError("Preserve existing fixtures; choose a fresh output directory")
a = loadmat(root / "hist_multi_probe.mat", simplify_cells=True)["multi_cases"]
b = loadmat(root / "threshold_holdout.mat", simplify_cells=True)["cases"]
arrays = {}
cases = []
for c in list(a) + list(b):
    if c["name"].startswith("real_"):
        continue
    name = c["name"]
    arrays[name] = c["image"]
    em = float(c["em"])
    cases.append(
        {
            "name": name,
            "n": int(c["n"]),
            "thresholds": np.asarray(c["thresholds"]).tolist(),
            "effectiveness": em if np.isfinite(em) else None,
            "effectiveness_status": "finite" if np.isfinite(em) else "negative_infinity",
        }
    )
with (output / "multithresh-synthetic.npz").open("xb") as stream:
    np.savez_compressed(stream, **arrays)
meta = {
    "reference": "MATLABR2026b CPU1",
    "fixture_sha256": hashlib.sha256(
        (output / "multithresh-synthetic.npz").read_bytes()
    ).hexdigest(),
    "cases": cases,
}
(output / "multithresh-metadata.json").write_text(
    json.dumps(meta, indent=2, allow_nan=False) + "\n"
)
print(len(arrays), len(cases), (output / "multithresh-synthetic.npz").stat().st_size)
