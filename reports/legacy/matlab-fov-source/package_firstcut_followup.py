import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_firstcut import first_cutoff_source
from deepretina.legacy.matlab_fov_source import automatic_fov_source, automatic_fov_stages
from scipy.io import loadmat

parser = argparse.ArgumentParser(
    description="Package measured first-cutoff contracts, synthetic-only fixtures"
)
parser.add_argument("--oracle-dir", type=Path, required=True)
parser.add_argument("--fixture-dir", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
output = args.fixture_dir
output.mkdir(parents=True, exist_ok=True)
for name in [
    "firstcut-histograms.npz",
    "firstcut-fov-synthetic.npz",
    "firstcut-fixture-metadata.json",
]:
    if (output / name).exists():
        raise FileExistsError("Choose fresh fixture outputs")
if args.report.exists():
    raise FileExistsError("Preserve existing reports")
counts = []
minimums = []
maximums = []
fixture = []
batches = []
for label, matname, seed in [
    ("a", "firstcut_holdout.mat", 11827),
    ("b", "firstcut_holdout2.mat", 90173),
]:
    cases = loadmat(root / matname, simplify_cells=True)["cases"]
    first_exact = decisions = 0
    maxerr = 0.0
    mapping = {}
    for c in cases:
        image = np.asarray(c["image"], np.float64)
        n = int(c["n"])
        result = first_cutoff_source(image, n)
        source = float(np.min(c["thresholds"]))
        error = abs(result.value - source)
        first_exact += result.value == source
        maxerr = max(maxerr, error)
        factors = [0.1] if n == 4 else [1.0, 0.5] if n == 5 else [1.0]
        decisions += sum(
            int(np.count_nonzero((image > source * f) != (image > result.value * f)))
            for f in factors
        )
        name = label + "_" + c["name"]
        lo, hi = float(image.min()), float(image.max())
        idx = np.floor((image - lo) / (hi - lo) * 255 + 0.5).astype(int)
        histogram = np.bincount(idx.ravel(), minlength=256)
        if (histogram > 0).sum() > n:
            if name not in mapping:
                mapping[name] = len(counts)
                counts.append(histogram)
                minimums.append(lo)
                maximums.append(hi)
            fixture.append(
                {
                    "name": name,
                    "histogram_row": mapping[name],
                    "n": n,
                    "source_first_cutoff": source,
                    "effectiveness_status": "finite"
                    if np.isfinite(c["em"])
                    else "negative_infinity",
                }
            )
    batches.append(
        {
            "name": label,
            "seed": seed,
            "cases": len(cases),
            "first_float_bitwise_exact": int(first_exact),
            "max_first_absolute_error": maxerr,
            "threshold_decision_disagreements": decisions,
            "oracle_sha256": hashlib.sha256((root / matname).read_bytes()).hexdigest(),
        }
    )
synthetic = []
arrays = {}
for i, c in enumerate(loadmat(root / "firstcut_synthetic_fov.mat", simplify_cells=True)["cases"]):
    kwargs = {"dataset": c["dataset"], "threshold_backend": "source_first_cutoff"}
    s = automatic_fov_stages(c["image"], **kwargs)
    row = {
        "name": c["name"],
        "dataset": c["dataset"],
        "initial_low": float(c["initial_low"]),
        "final_low": float(c["final_low"]),
        "stage_disagreements": {
            k: int(np.count_nonzero(s[k] != c[k])) for k in ["initial_closed", "closed", "filled"]
        },
    }
    assert s["low_threshold"] == c["final_low"]
    error = c["error"] if isinstance(c["error"], str) else ""
    try:
        r = automatic_fov_source(c["image"], **kwargs)
    except ValueError as exc:
        assert error
        row.update(source_error=error, python_error=str(exc))
    else:
        assert not error
        row.update(
            calibration=asdict(r.calibration),
            boundary_disagreements=int(np.count_nonzero(r.calibration_boundary != c["boundary"])),
            processing_disagreements=int(np.count_nonzero(r.processing_fov != c["processing_fov"])),
        )
        assert [
            r.calibration.center_row_one_based,
            r.calibration.center_col_one_based,
            r.calibration.radius_pixels,
        ] == [c["row"], c["col"], c["radius"]]
    synthetic.append(row)
    item = {
        "name": c["name"],
        "dataset": c["dataset"],
        "source_error": error,
        "initial_low": float(c["initial_low"]),
        "final_low": float(c["final_low"]),
    }
    for field in ["image", "initial_closed", "closed", "filled", "boundary", "processing_fov"]:
        key = f"{i}_{field}"
        arrays[key] = c[field]
        item[field] = key
    if not error:
        item["calibration"] = [int(c["row"]), int(c["col"]), int(c["radius"])]
    synthetic[-1]["initial_low_absolute_error"] = float(
        abs(
            s["first_cutoff"]["value"] * (0.1 if c["dataset"] == "diaretdb" else 1)
            - c["initial_low"]
        )
    )
    row["fixture"] = item
np.savez_compressed(
    output / "firstcut-histograms.npz",
    counts=np.asarray(counts, np.int32),
    minimums=np.asarray(minimums),
    maximums=np.asarray(maximums),
)
np.savez_compressed(output / "firstcut-fov-synthetic.npz", **arrays)
(output / "firstcut-fixture-metadata.json").write_text(
    json.dumps(
        {
            "histogram_fixture_sha256": hashlib.sha256(
                (output / "firstcut-histograms.npz").read_bytes()
            ).hexdigest(),
            "cases": fixture,
            "fov_fixture_sha256": hashlib.sha256(
                (output / "firstcut-fov-synthetic.npz").read_bytes()
            ).hexdigest(),
            "fov_cases": [row.pop("fixture") for row in synthetic],
        },
        indent=2,
        allow_nan=False,
    )
    + "\n"
)
report = {
    "source_ref": "7bb0539b90dfce2ed6eaeef1de812072be1155d0",
    "caller_path": "matlab_preprocessing/souradnice.m",
    "caller_lines": {
        "N4": 11,
        "N5": 13,
        "N3": 15,
        "N5_half": 17,
        "consume_low_mask": 29,
        "histogram_fallback_replacement": 74,
    },
    "input_double_caller": {"path": "matlab_preprocessing/image_adjustment.m", "lines": [2, 3]},
    "automatic_loader_lines": {
        "load_stare.m": 30,
        "load_diaretdb0.m": 10,
        "load_drishtigs.m_training": 12,
    },
    "source_helper_sha256": hashlib.sha256((root / "souradnice.m").read_bytes()).hexdigest(),
    "batches": batches,
    "synthetic_fov_cases": synthetic,
    "real_fov": json.loads((root / "firstcut_real_native_comparison.json").read_text()),
    "scope": "First cutoff only; opt-in optional FOV backend, no full multithresh or universal equivalence certification",
    "public_defaults_changed": False,
    "frozen_sources_changed": False,
    "prior_measurement_correction": {
        "problem": "Rounded bin coordinates hid fractional failed-search cutoffs",
        "first_batch_float_mismatches_before_fix": 22,
        "threshold_decision_differences_before_fix": 14,
        "fix": "Return continuous uniform initial coordinates on failed search, no integer rounding",
    },
    "degenerate_arbitrary_threshold_policy": "unsupported, fails closed",
}
args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(batches, "fov", len(synthetic), "fixture histogram rows", len(counts), "cases", len(fixture))
