"""Read existing public-operator goldens; no toolbox source/model/GT access."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sobel(d):
    p = np.pad(d, 1, mode="edge")
    gx = -p[2:, :-2]
    gx -= 2 * p[1:-1, :-2]
    gx -= p[:-2, :-2]
    gx += p[2:, 2:]
    gx += 2 * p[1:-1, 2:]
    gx += p[:-2, 2:]
    gy = p[2:, :-2].copy()
    gy -= p[:-2, :-2]
    gy += 2 * p[2:, 1:-1]
    gy -= 2 * p[:-2, 1:-1]
    gy += p[2:, 2:]
    gy -= p[:-2, 2:]
    return gx, gy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    cases = list(loadmat(a.root / "sobel_operator_goldens.mat", simplify_cells=True)["cases"])
    cases.append(
        dict(
            name="diaret",
            **loadmat(
                a.root / "diaret_photometry_stages.mat",
                simplify_cells=True,
                variable_names=["distance", "gx", "gy"],
            ),
        )
    )
    result = []
    for case in cases:
        row = {"name": case["name"], "shape": list(case["distance"].shape), "axes": {}}
        for name, current in zip(["gx", "gy"], sobel(case["distance"]), strict=True):
            expected = case[name]
            bad = current != expected
            y, x = np.where(bad)
            summary = {
                "unequal": int(bad.sum()),
                "max_abs": float(np.max(np.abs(current - expected))),
                "border_unequal": int(bad[[0, -1]].sum() + bad[1:-1, [0, -1]].sum()),
            }
            if len(y):
                summary.update(
                    bbox=[int(y.min()), int(y.max()), int(x.min()), int(x.max())],
                    column_mod8=np.bincount(x % 8, minlength=8).tolist(),
                    row_mod8=np.bincount(y % 8, minlength=8).tolist(),
                    first_examples=[
                        {
                            "y": int(yy),
                            "x": int(xx),
                            "reference": float(expected[yy, xx]),
                            "current": float(current[yy, xx]),
                            "patch": np.pad(case["distance"], 1, mode="edge")[
                                yy : yy + 3, xx : xx + 3
                            ].tolist(),
                        }
                        for yy, xx in zip(y[:5], x[:5], strict=True)
                    ],
                )
            row["axes"][name] = summary
        result.append(row)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            (
                r["name"],
                r["axes"]["gx"]["unequal"],
                r["axes"]["gy"]["unequal"],
                r["axes"]["gx"]["border_unequal"],
                r["axes"]["gy"]["border_unequal"],
            )
            for r in result
        ]
    )


if __name__ == "__main__":
    main()
