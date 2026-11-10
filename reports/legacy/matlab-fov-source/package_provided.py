"""Compare optional provided-source FOV port and package synthetic-only oracles."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from deepretina.legacy.matlab_fov_source import provided_fov_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = loadmat(args.oracle_dir / "provided_source_goldens.mat", simplify_cells=True)["cases"]
    arrays, fixture_cases, reports = {}, [], []
    for i, case in enumerate(cases):
        error = case["error"] if isinstance(case["error"], str) else ""
        row = {
            "name": case["name"],
            "dataset": case["dataset"],
            "shape": list(case["mask"].shape),
            "mask_sha256": hashlib.sha256(np.asarray(case["mask"], bool).tobytes()).hexdigest(),
            "source_error": error,
        }
        try:
            result = provided_fov_source(case["mask"], dataset=case["dataset"])
        except ValueError as exc:
            assert error, (case["name"], str(exc))
            row["python_error"] = str(exc)
        else:
            assert not error, (case["name"], "source error not reproduced")
            row.update(
                calibration=asdict(result.calibration),
                diameter_pixels=result.calibration.diameter_pixels,
                boundary_disagreements=int(
                    np.count_nonzero(result.calibration_boundary != case["boundary"])
                ),
                processing_fov_disagreements=int(
                    np.count_nonzero(result.processing_fov != case["processing_fov"])
                ),
                source_branch=result.source_branch,
            )
            assert row["boundary_disagreements"] == row["processing_fov_disagreements"] == 0
            assert [
                result.calibration.center_row_one_based,
                result.calibration.center_col_one_based,
                result.calibration.radius_pixels,
            ] == [case["row"], case["col"], case["radius"]]
            row["reference_calibration"] = {
                "row": int(case["row"]),
                "col": int(case["col"]),
                "radius": int(case["radius"]),
            }
        reports.append(row)
        if not case["name"].startswith("real_"):
            item = {
                "name": case["name"],
                "dataset": case["dataset"],
                "source_error": error,
                "row": None if error else int(case["row"]),
                "col": None if error else int(case["col"]),
                "radius": None if error else int(case["radius"]),
            }
            for field in ["mask", "boundary", "processing_fov"]:
                key = f"{i}_{field}"
                arrays[key] = case[field]
                item[field] = key
            fixture_cases.append(item)
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture_dir / "synthetic.npz", **arrays)
    metadata = {
        "reference": "MATLAB R2026b singleCompThread, source souradnice7bb0539b",
        "fixture_sha256": hashlib.sha256(
            (args.fixture_dir / "synthetic.npz").read_bytes()
        ).hexdigest(),
        "cases": fixture_cases,
    }
    (args.fixture_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    real_sources = {
        name: json.loads((args.oracle_dir / file).read_text())
        for name, file in [("erlangen", "real_source.json"), ("rimone", "real_rimone_source.json")]
    }
    report = {
        "scope": "Optional provided-special source FOV stages only; no automatic/photometric certification or public activation",
        "runtime": "MATLAB R2026b singleCompThread CPU nice10 CUDA hidden",
        "source_ref": "7bb0539b90dfce2ed6eaeef1de812072be1155d0",
        "helper_sha256": hashlib.sha256(
            (args.oracle_dir / "souradnice.m").read_bytes()
        ).hexdigest(),
        "real_sources": real_sources,
        "cases": reports,
        "public_route_changed": False,
        "automatic_port_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "measured_cases": len(cases),
                "synthetic_cases": len(fixture_cases),
                "all_defined_calibrations_exact": True,
            }
        )
    )


if __name__ == "__main__":
    main()
