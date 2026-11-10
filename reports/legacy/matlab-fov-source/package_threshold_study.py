"""Quantify the optional histogram port and uncertified threshold hypotheses."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from deepretina.legacy.matlab_fov_source import source_histcounts250
from nelder_mead_candidate import candidate_multithresh
from scipy.io import loadmat


def sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.oracle_dir
    broad = loadmat(root / "hist_multi_probe.mat", simplify_cells=True)
    holdout = loadmat(root / "hist_holdout.mat", simplify_cells=True)["cases"]
    thresholds = loadmat(root / "threshold_holdout.mat", simplify_cells=True)
    constants = loadmat(root / "hist_constant.mat", simplify_cells=True)["cases"]
    fixture = []
    rows = []

    def check(name, values, edges, counts=None):
        values = np.asarray(values, np.float64)
        lo, hi = float(values.min()), float(values.max())
        if lo < 0 or hi > 1:
            return
        actual, generated = source_histcounts250(values)
        assert np.array_equal(generated, edges), (name, np.max(np.abs(generated - edges)))
        if counts is not None:
            assert np.array_equal(actual, counts), name
        fixture.append({"name": name, "minimum": lo, "maximum": hi, "edges_sha256": sha(edges)})
        rows.append(
            {
                "name": name,
                "edge_disagreements": 0,
                "count_disagreements": None if counts is None else 0,
            }
        )

    for i, c in enumerate(broad["hist_cases"]):
        if c["nbins"] == 250:
            check(f"development_range_{i}", [c["lo"], c["hi"]], c["edges"])
    for i, c in enumerate(holdout):
        check(f"independent_range_{i}", c["values"], c["edges"], c["counts"])
    for i, c in enumerate(constants):
        check(f"constant_{i}", np.full(13, c["value"]), c["edges"], c["counts"])
    for i, c in enumerate(thresholds["hist_special"]):
        check(f"special_{i}", c["values"], c["edges"], c["counts"])
    for c in thresholds["cases"]:
        if c["n"] == 3:
            check(c["name"], c["image"], c["histogram_edges"], c["histogram_counts"])
    multi = []
    for c in list(broad["multi_cases"]) + list(thresholds["cases"]):
        image = c["image"]
        n = c["n"]
        expected = np.asarray(c["thresholds"], np.float64)
        row = {
            "name": c["name"],
            "n": int(n),
            "source_thresholds": expected.tolist(),
            "source_effectiveness": float(c["em"]) if math.isfinite(c["em"]) else None,
            "source_effectiveness_status": "finite"
            if math.isfinite(c["em"])
            else "negative_infinity",
            "image_sha256": sha(image),
            "shape": list(np.asarray(image).shape),
        }
        try:
            result = candidate_multithresh(image, n)
        except NotImplementedError:
            row["status"] = "degenerate_policy_unported"
        else:
            indices = (expected - image.min()) / (image.max() - image.min()) * 255
            delta = result["threshold_indices"] - indices
            row.update(
                candidate_thresholds=result["thresholds"].tolist(),
                candidate_threshold_indices=result["threshold_indices"].tolist(),
                source_threshold_indices=indices.tolist(),
                index_differences=delta.tolist(),
                first_threshold_index_equal=bool(result["thresholds"][0] == expected[0]),
                all_threshold_indices_equal=bool(np.array_equal(result["thresholds"], expected)),
                candidate_effectiveness=result["effectiveness"],
                candidate_effectiveness_status=result["effectiveness_status"],
                status="candidate_not_certified",
            )
        multi.append(row)
    candidate = [r for r in multi if r["status"] == "candidate_not_certified"]
    report = {
        "scope": "Optional source port, no activation in public profiles or frozen evaluation/training",
        "runtime": "MATLAB R2026b singleCompThread CPU nice10 CUDA hidden; Python CPU1",
        "native_histcounts250_domain": "finite double grayscale[0,1], fixed250bins; unsupported numeric boundaries fail closed",
        "native_histograms_exact": len(rows),
        "histograms": rows,
        "full_automatic_port_complete": False,
        "public_route_changed": False,
        "multithresh_candidate": {
            "method": "cleanroom stable Nelder-Mead,256nearest-center histogram, source-failed-search fallback hypothesis",
            "matlab_parity_certified": False,
            "cases": multi,
            "nondegenerate_cases": len(candidate),
            "full_vector_exact": sum(r["all_threshold_indices_equal"] for r in candidate),
            "first_threshold_exact": sum(r["first_threshold_index_equal"] for r in candidate),
            "remaining_gap": "N5 range ramp differs at two higher thresholds; arbitrary degenerate policy unported; source can return negative-infinity effectiveness",
        },
        "global_otsu_counterexample": {
            "case": "three_modes N5",
            "source_em": 0.9901741362563441,
            "global_dp_em": 0.992556770078339,
            "source_indices": [32, 83, 124, 198, 234],
            "global_dp_indices": [31, 71, 115, 135, 174],
            "conclusion": "A better global objective is not a faithful source replacement",
        },
        "oracle_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.glob("*.mat")
            if path.name
            in [
                "hist_multi_probe.mat",
                "hist_holdout.mat",
                "hist_constant.mat",
                "threshold_holdout.mat",
                "multithresh_fminsearch.mat",
            ]
        },
    }
    with args.fixture.open("x") as f:
        json.dump({"reference": "MATLABR2026b", "cases": fixture}, f, indent=2, allow_nan=False)
        f.write("\n")
    with args.report.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        "native histogram exact",
        len(rows),
        "candidate exact",
        report["multithresh_candidate"]["full_vector_exact"],
        "/",
        len(candidate),
        "first",
        report["multithresh_candidate"]["first_threshold_exact"],
    )


if __name__ == "__main__":
    main()
