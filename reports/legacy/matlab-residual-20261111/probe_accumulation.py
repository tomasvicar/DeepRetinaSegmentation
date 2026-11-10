"""A small operator-derived candidate set, not arbitrary coefficient permutations."""

import argparse
import json
from pathlib import Path

import numpy as np
from diagnose_sobel import sobel
from scipy.io import loadmat


def methods(d):
    p = np.pad(d, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[1:-1, :-2], p[2:, :-2]
    q, r, s = p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
    top, bot = p[:-2, 1:-1], p[2:, 1:-1]
    termsx = [-a, -2 * b, -c, q, 2 * r, s]
    termsy = [-a, -2 * top, -q, c, 2 * bot, s]

    def add(terms, order):
        result = terms[order[0]].copy()
        for index in order[1:]:
            result += terms[index]
        return result

    result = {"legacy": sobel(d)}
    for name, order in [
        ("column_forward", [0, 1, 2, 3, 4, 5]),
        ("column_reverse", [5, 4, 3, 2, 1, 0]),
        ("row_forward", [0, 3, 1, 4, 2, 5]),
        ("row_reverse", [5, 2, 4, 1, 3, 0]),
    ]:
        result[name] = (add(termsx, order), add(termsy, order))
    result["double_once"] = (
        sum(t.astype(np.float64) for t in termsx).astype(np.float32),
        sum(t.astype(np.float64) for t in termsy).astype(np.float32),
    )
    return result


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
    rows = []
    for case in cases:
        d = case["distance"]
        h, w = d.shape
        all_methods = methods(d)
        region = np.zeros((h, w), bool)
        region[:, [0, -1]] = True
        region[((h - 1) // 16) * 16 :] = True
        row = {
            "name": case["name"],
            "shape": [h, w],
            "uncovered_legacy_errors": {},
            "candidates": {},
        }
        for name, values in all_methods.items():
            stats = {}
            for axis, expected, legacy, value in zip(
                ["gx", "gy"], [case["gx"], case["gy"]], all_methods["legacy"], values, strict=True
            ):
                bad = legacy != expected
                combined = np.where(region, value, legacy)
                stats[axis] = {
                    "total": int(np.count_nonzero(value != expected)),
                    "on_legacy_error": int(np.count_nonzero(value[bad] != expected[bad])),
                    "combined": int(np.count_nonzero(combined != expected)),
                    "uncovered": int(np.count_nonzero(bad & ~region)),
                    "top_interior": int(np.count_nonzero(bad[0, 1:-1])),
                }
            row["candidates"][name] = stats
        rows.append(row)
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        [
            (
                r["name"],
                {n: (v["gx"]["combined"], v["gy"]["combined"]) for n, v in r["candidates"].items()},
            )
            for r in rows
        ]
    )


if __name__ == "__main__":
    main()
