"""Optional first-cutoff source contract, not a full multithresh clone.

Measured against public MATLAB R2026b calls and user-owned souradnice7bb0539b.
The FOV caller consumes only min(multithresh(gray,N)); higher cuts are private
search state and have known disagreements. This module changes no public default.
"""

from dataclasses import dataclass

import numpy as np


def _nelder_mead(fun, x0):
    x = np.asarray(x0, float)
    n = len(x)
    vertices = np.tile(x, (n + 1, 1))
    for i in range(n):
        vertices[i + 1, i] = 1.05 * x[i] if x[i] else 0.00025
    values = np.array([fun(v) for v in vertices])
    evaluations = n + 1
    iterations = 1

    def sort():
        nonlocal vertices, values
        order = np.argsort(values, kind="stable")
        vertices = vertices[order]
        values = values[order]

    sort()
    while evaluations < n * 200 and iterations < n * 200:
        if (
            np.max(np.abs(vertices[1:] - vertices[0])) <= 1e-4
            and np.max(np.abs(values[1:] - values[0])) <= 1e-4
        ):
            break
        centroid = np.sum(vertices[:-1], axis=0) / n
        reflected = 2 * centroid - vertices[-1]
        fr = fun(reflected)
        evaluations += 1
        shrink = False
        if fr < values[0]:
            expanded = 3 * centroid - 2 * vertices[-1]
            fe = fun(expanded)
            evaluations += 1
            if fe < fr:
                vertices[-1], values[-1] = expanded, fe
            else:
                vertices[-1], values[-1] = reflected, fr
        elif fr < values[-2]:
            vertices[-1], values[-1] = reflected, fr
        elif fr < values[-1]:
            contracted = 1.5 * centroid - 0.5 * vertices[-1]
            fc = fun(contracted)
            evaluations += 1
            if fc <= fr:
                vertices[-1], values[-1] = contracted, fc
            else:
                shrink = True
        else:
            contracted = 0.5 * centroid + 0.5 * vertices[-1]
            fc = fun(contracted)
            evaluations += 1
            if fc < values[-1]:
                vertices[-1], values[-1] = contracted, fc
            else:
                shrink = True
        if shrink:
            vertices[1:] = vertices[0] + 0.5 * (vertices[1:] - vertices[0])
            for i in range(1, n + 1):
                values[i] = fun(vertices[i])
                evaluations += 1
        sort()
        iterations += 1
    return vertices[0], values[0], iterations, evaluations


def _search_thresholds(image, n):
    """Private search state behind the optional measured first-cutoff backend.

    Histogram, simplex stable ordering and failed-search fallback are measured
    hypotheses. A known N5 ramp disagrees in two returned higher thresholds.
    """
    image = np.asarray(image, np.float64)
    if n not in (3, 4, 5) or not image.size or not np.isfinite(image).all():
        raise ValueError("Study requires finite image and N3/4/5")
    if np.unique(image).size <= n:
        raise NotImplementedError("Degenerate arbitrary-threshold policy is not ported")
    lo, hi = float(image.min()), float(image.max())
    index = np.floor((image - lo) / (hi - lo) * 255 + 0.5).astype(np.int64)
    counts = np.bincount(index.ravel(), minlength=256).astype(np.float64)
    probability = counts / counts.sum()
    mass = np.r_[0, probability.cumsum()]
    moment = np.r_[0, (probability * np.arange(256)).cumsum()]
    mean = (probability * np.arange(256)).sum()
    variance = (probability * np.arange(256) ** 2).sum() - mean**2

    def objective(values):
        cuts = np.floor(values + 0.5).astype(np.int64)
        if cuts.min() < 0 or cuts.max() > 254 or (np.diff(cuts) <= 0).any():
            return 1e10
        bounds = np.r_[0, cuts + 1, 256]
        m = np.diff(mass[bounds])
        s = np.diff(moment[bounds])
        if (m <= 0).any():
            return 1e10
        return -np.sum(s * s / m)

    initial = np.arange(1, n + 1) * 255 / (n + 1)
    values, score, iterations, evaluations = _nelder_mead(objective, initial)
    failed = score >= 1e10
    cuts = initial.copy() if failed else np.floor(values + 0.5).astype(np.int64)
    return {
        "thresholds": lo + (hi - lo) * (cuts / 255),
        "threshold_indices": cuts,
        "effectiveness": None if failed else float((-score - mean**2) / variance),
        "effectiveness_status": "negative_infinity" if failed else "finite",
        "method": "cleanroom_stable_nelder_mead_candidate_not_certified",
        "iterations": iterations,
        "evaluations": evaluations,
        "matlab_parity_certified": False,
    }


@dataclass(frozen=True)
class FirstCutoff:
    value: float
    bin_coordinate: float
    threshold_count: int
    effectiveness_status: str
    method: str = "source_first_cutoff_stable_nelder_mead_r2026b_measured"
    source_ref: str = "7bb0539b90dfce2ed6eaeef1de812072be1155d0"
    reference_runtime: str = "MATLAB R2026b"
    full_multithresh_clone: bool = False
    matlab_parity_certified: bool = False


def first_cutoff_source(gray, n):
    """Return only the source FOV's unscaled first cutoff for N3/4/5.

    Requires finite double grayscale in[0,1]. Arbitrary degenerate-threshold
    policies remain unsupported. Failed searches preserve continuous initial
    coordinates, including fractional63.75/42.5, and retain their status.
    Scope/finite measurements are documented in MATLAB_FIRST_CUTOFF.md;
    this is neither a universal equivalence proof nor a full-vector API.
    """
    values = np.asarray(gray)
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n not in (3, 4, 5)
    ):
        raise ValueError("Source first cutoff requires integer N3/4/5")
    if values.dtype != np.float64 or not values.size or not np.isfinite(values).all():
        raise ValueError("Source first cutoff requires finite nonempty float64 grayscale")
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Source first cutoff grayscale must be in[0,1]")
    result = _search_thresholds(values, n)
    return FirstCutoff(
        float(result["thresholds"][0]),
        float(result["threshold_indices"][0]),
        int(n),
        result["effectiveness_status"],
    )
