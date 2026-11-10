import numpy as np


def nelder_mead(fun, x0):
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


def candidate_multithresh(image, n):
    """Uncertified local-search hypothesis, deliberately not wired into runtime.

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
    values, score, iterations, evaluations = nelder_mead(objective, initial)
    failed = score >= 1e10
    cuts = np.rint(initial).astype(np.int64) if failed else np.floor(values + 0.5).astype(np.int64)
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
