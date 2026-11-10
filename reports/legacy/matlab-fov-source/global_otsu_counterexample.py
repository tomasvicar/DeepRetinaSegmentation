from itertools import pairwise

import numpy as np
from scipy.io import loadmat


def dp_hist(values, n):
    lo = float(values.min())
    hi = float(values.max())
    indices = np.floor((values - lo) / (hi - lo) * 255 + 0.5).astype(np.int64)
    h = np.bincount(indices.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    mass = np.r_[0, p.cumsum()]
    moment = np.r_[0, (p * np.arange(256)).cumsum()]
    score = np.full((256, 256), -np.inf)
    for start in range(256):
        m = mass[start + 1 :] - mass[start]
        s = moment[start + 1 :] - moment[start]
        np.divide(s * s, m, out=score[start, start:], where=m > 0)
    table = np.full((n + 2, 257), -np.inf)
    prev = np.zeros((n + 2, 257), int)
    table[0, 0] = 0
    for k in range(1, n + 2):
        for end in range(k, 257):
            starts = np.arange(k - 1, end)
            s = table[k - 1, starts] + score[starts, end - 1]
            best = int(np.argmax(s))
            table[k, end] = s[best]
            prev[k, end] = starts[best]
    cuts = []
    end = 256
    for k in range(n + 1, 1, -1):
        start = prev[k, end]
        cuts.append(start - 1)
        end = start
    cuts = np.array(cuts[::-1])
    mean = (p * np.arange(256)).sum()
    var = (p * np.arange(256) ** 2).sum() - mean**2

    def effect(t):
        b = np.r_[0, t + 1, 256]
        return sum(score[a, e - 1] for a, e in pairwise(b)) - mean**2

    return cuts, effect(cuts) / var, effect, var


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Independent global Otsu counterexample; not source replacement"
    )
    parser.add_argument("--oracle-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = loadmat(args.oracle_dir / "hist_multi_probe.mat", simplify_cells=True)["multi_cases"]
    case = next(c for c in cases if c["name"] == "three_modes" and c["n"] == 5)
    cuts, em, _, _ = dp_hist(case["image"], 5)
    print(
        {
            "global_indices": cuts.tolist(),
            "global_effectiveness": float(em),
            "source_effectiveness": float(case["em"]),
        }
    )
