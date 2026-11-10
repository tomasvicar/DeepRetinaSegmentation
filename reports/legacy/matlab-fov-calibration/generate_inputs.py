"""Generate small synthetic-only masks for public MATLAB edge oracle calls."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def make_inputs():
    cases = []
    for name, h, w, cx, cy, rx, ry in [
        ("circle_odd", 61, 71, 35, 30, 22, 22),
        ("circle_even", 60, 70, 34.5, 29.5, 22, 22),
        ("circle_offcenter", 63, 79, 29, 36, 24, 24),
        ("ellipse", 63, 79, 39, 31, 33, 22),
        ("cropped", 63, 79, 10, 31, 35, 35),
        ("touches", 61, 71, 35, 30, 30, 30),
        ("small", 9, 11, 5, 4, 3, 3),
    ]:
        y, x = np.indices((h, w))
        cases.append((name, ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1))
    mask = cases[0][1].copy()
    mask[25:35, 30:40] = False
    cases.append(("hole", mask))
    mask = np.zeros((61, 71), bool)
    mask[8:52, 8:62] = True
    cases.extend(
        [
            ("rectangle", mask),
            ("empty", np.zeros((31, 41), bool)),
            ("full", np.ones((31, 41), bool)),
            ("checkerboard", np.indices((31, 41)).sum(0) % 2 == 0),
            ("random", np.random.default_rng(152).random((35, 43)) > 0.7),
        ]
    )
    rng = np.random.default_rng(337)
    for i in range(60):
        h, w = rng.integers(7, 81, size=2)
        y, x = np.indices((h, w))
        if i < 30:
            mask = rng.random((h, w)) < ((i + 1) / 32)
        else:
            cx, cy = rng.uniform(-0.1, 1.1, size=2) * [w, h]
            rx, ry = rng.uniform(0.1, 0.8, size=2) * [w, h]
            mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        cases.append((f"held_probe_{i:02}", mask))
    result = np.empty(len(cases), dtype=[("name", "O"), ("mask", "O")])
    for i, case in enumerate(cases):
        result[i] = case
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    savemat(args.output, {"inputs": make_inputs()})
