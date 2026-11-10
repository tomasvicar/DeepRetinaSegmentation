"""Package only synthetic automatic-FOV MATLAB oracles and quantify stage errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_fov_source import (
    automatic_fov_source,
    automatic_fov_stages,
    source_close,
    source_disk10,
    source_histcounts250,
    source_rgb2gray,
    source_seed_components,
)
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    primitive = loadmat(
        args.oracle_dir / "automatic_primitives.mat", simplify_cells=True
    )
    automatic = loadmat(
        args.oracle_dir / "automatic_zero_goldens.mat", simplify_cells=True
    )["cases"]
    arrays = {}
    metadata = {
        "rgb": [],
        "morphology": [],
        "components": [],
        "histograms": [],
        "automatic": [],
    }

    def store(group, index, case, fields):
        item = {}
        for field in fields:
            key = f"{group}_{index}_{field}"
            arrays[key] = case[field]
            item[field] = key
        return item

    assert np.array_equal(source_disk10(), primitive["footprint"])
    arrays["footprint"] = primitive["footprint"]
    for i, c in enumerate(primitive["rgb_cases"]):
        assert np.array_equal(source_rgb2gray(c["rgb"]), c["gray"])
        metadata["rgb"].append(store("rgb", i, c, ["rgb", "gray"]))
    for i, c in enumerate(primitive["morphology"]):
        assert np.array_equal(source_close(c["mask"]), c["closed"])
        metadata["morphology"].append(
            store("morphology", i, c, ["mask", "closed", "filled"])
        )
    for i, key in enumerate(["components", "components_with_background"]):
        c = primitive[key]
        assert np.array_equal(
            source_seed_components(c["mask"], c["seeds"]), c["selected"]
        )
        metadata["components"].append(
            store("components", i, c, ["mask", "seeds", "selected"])
        )
    for i, c in enumerate(primitive["histograms"]):
        counts, edges = source_histcounts250(c["values"], oracle_edges=c["edges"])
        assert np.array_equal(counts, c["counts"]) and np.array_equal(edges, c["edges"])
        metadata["histograms"].append(
            store("histograms", i, c, ["values", "counts", "edges"])
        )
    rows = []
    for i, c in enumerate(automatic):
        kwargs = {"dataset": c["dataset"], "histogram_edges": c["histogram_edges"]}
        stages = automatic_fov_stages(c["rgb"], **kwargs)
        differences = {
            k: int(np.count_nonzero(stages[k] != c[k]))
            for k in ["gray", "selected", "initial_closed", "closed", "filled"]
        }
        assert not any(differences.values()), (c["name"], differences)
        assert (
            stages["low_threshold"] == c["low"]
            and stages["high_threshold"] == c["high"]
        )
        error = c["error"] if isinstance(c["error"], str) else ""
        row = {
            "name": c["name"],
            "dataset": c["dataset"],
            "stage_disagreements": differences,
            "fallback_used": stages["fallback_used"],
            "low_threshold": stages["low_threshold"],
            "source_error": error,
            "histogram_backend": stages["histogram_backend"],
        }
        try:
            r = automatic_fov_source(c["rgb"], **kwargs)
        except ValueError as exc:
            assert error, (c["name"], str(exc))
            row["python_error"] = str(exc)
        else:
            assert not error, c["name"]
            calib = [
                r.calibration.center_row_one_based,
                r.calibration.center_col_one_based,
                r.calibration.radius_pixels,
            ]
            assert calib == [c["row"], c["col"], c["radius"]]
            assert np.array_equal(r.processing_fov, c["processing_fov"])
            row["calibration"] = calib
            row["processing_fov_disagreements"] = 0
        rows.append(row)
        item = store(
            "automatic",
            i,
            c,
            [
                "rgb",
                "gray",
                "selected",
                "initial_closed",
                "closed",
                "filled",
                "histogram_edges",
            ],
        )
        item.update(
            name=c["name"],
            dataset=c["dataset"],
            source_error=error,
            low=float(c["low"]),
            high=float(c["high"]),
            calibration=None if error else row["calibration"],
        )
        if not error:
            item.update(store("automatic", i, c, ["processing_fov"]))
        metadata["automatic"].append(item)
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixture_dir / "automatic-synthetic.npz", **arrays)
    metadata.update(
        reference="MATLAB R2026b CPU singleCompThread, souradnice7bb0539b",
        fixture_sha256=hashlib.sha256(
            (args.fixture_dir / "automatic-synthetic.npz").read_bytes()
        ).hexdigest(),
    )
    (args.fixture_dir / "automatic-metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    report = {
        "scope": "Optional source-stage study, no public activation/full automatic certification",
        "automatic_port_complete": False,
        "public_route_changed": False,
        "unresolved": [
            "multithresh N3/4/5 local optimizer",
            "histcounts250 implicit edge placement",
        ],
        "oracle_thresholds_are_not_native_python": True,
        "cases": rows,
        "primitive_cases": {
            k: len(metadata[k])
            for k in ["rgb", "morphology", "components", "histograms"]
        },
        "known_rejected_histogram_guess": {
            "synthetic_case": "g1020_noise_fallback",
            "matlab_low": 0.002,
            "guessed_python_low": 1 / 255,
            "mask_disagreements": 11339,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(report["primitive_cases"]),
        len(rows),
        "automatic cases exact with supplied oracle edges",
    )


if __name__ == "__main__":
    main()
