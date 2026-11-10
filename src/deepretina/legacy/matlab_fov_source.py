"""Optional source-policy FOV port, separate from public provided-FOV defaults.

Calibration boundaries and returned processing FOVs are deliberately distinct.
This preserves the original RIM-ONE and pre-hole-fill semantics rather than
silently substituting a robust circle fit or the single_net Python loader.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .matlab_fov import FOVCalibration, binary_sobel_edges


@dataclass(frozen=True)
class MatlabSourceFOV:
    calibration_boundary: np.ndarray
    processing_fov: np.ndarray
    calibration: FOVCalibration
    source_branch: str
    closed_mask: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _binary(mask):
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) < 2 or max(values.shape) <= 5:
        raise ValueError("Source provided-mask branch requires a 2D mask with max dimension >5")
    if not np.isfinite(values).all():
        raise ValueError("Provide a finite binary source FOV")
    levels = np.unique(values)
    if len(levels) > 2 or (len(levels) == 2 and levels[0] != 0) or (levels < 0).any():
        raise ValueError("Threshold source FOV values explicitly before calibration")
    return (values != 0).copy()


def _mode(values):
    values, counts = np.unique(values, return_counts=True)
    if not len(values):
        raise ValueError("Source FOV has no usable paired boundary intersections")
    return values[np.argmax(counts)]


def _calibration(boundary):
    def midpoints(array):
        selected = array[array.sum(axis=1) >= 2]
        if not len(selected):
            return np.array([])
        first = selected.argmax(axis=1)
        last = selected.shape[1] - 1 - selected[:, ::-1].argmax(axis=1)
        return (first + last) / 2 + 1

    col = math.floor(float(_mode(midpoints(boundary))) + 0.5)
    row = math.floor(float(_mode(midpoints(boundary.T))) + 0.5)
    y, x = np.where(boundary)
    distance = np.sqrt((x.astype(float) + 1 - col) ** 2 + (y.astype(float) + 1 - row) ** 2)
    radius = math.floor(float(_mode(distance)) + 0.5) + 1
    return FOVCalibration(row, col, radius, int(boundary.sum()))


def rimone_loader_mask(shape):
    """Original load_rimone left-half rectangular wireframe, not a dense FOV."""
    h, w = shape
    if (
        not isinstance(h, (int, np.integer))
        or not isinstance(w, (int, np.integer))
        or min(h, w) < 1
        or w % 2
    ):
        raise ValueError(
            "Original RIM-ONE loader wireframe requires positive integer shape and even width"
        )
    mask = np.zeros((h, w), bool)
    mask[[0, -1], : w // 2] = True
    mask[:, [0, w // 2 - 1]] = True
    return mask


def provided_fov_source(mask, *, dataset="generic"):
    """Follow souradnice provided branches, without activating any public route.

    `mask` is the supplied source mask. RIM-ONE's loader supplies a wireframe;
    its returned processing FOV is all ones. Erlangen calibrates from vertical
    frame rails but retains the supplied processing mask (normally all ones).
    ROC/RIGA calibrate from Sobel edges of the source's asymmetric inner inset.
    """
    source = _binary(mask)
    name = str(dataset).lower().replace("-", "").replace("_", "")
    processing = source.copy()
    if name in {"rim", "rimone"}:
        boundary = source.copy()
        processing = np.ones_like(source)
        branch = "provided_rimone_wireframe"
    elif name in {"erlangen", "erlangenvideodata"}:
        boundary = np.zeros_like(source)
        boundary[:, [0, -1]] = True
        branch = "provided_erlangen_rails"
    elif name in {"roc", "riga"}:
        inset = source[19:-20, 19:-20]
        if min(inset.shape) < 2:
            raise ValueError("Source ROC/RIGA calibration inset is too small")
        boundary = np.zeros_like(source)
        boundary[19:-20, 19:-20] = binary_sobel_edges(inset)
        branch = "provided_roc_riga_inset"
    else:
        boundary = binary_sobel_edges(source)
        branch = "provided_generic"
    return MatlabSourceFOV(boundary, processing, _calibration(boundary), branch, source)


def source_rgb2gray(image):
    """Measured MATLAB double RGB weights after im2double(uint8RGB)."""
    values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Source automatic FOV currently requires raw RGB uint8")
    values = values.astype(np.float64) / 255
    return (
        values[:, :, 0] * 0.298936021293776
        + values[:, :, 1] * 0.587043074451121
        + values[:, :, 2] * 0.114020904255103
    )


def source_disk10():
    """Public getnhood(strel('disk',10)) oracle, default four-line approximation."""
    widths = (
        11,
        13,
        15,
        17,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        17,
        15,
        13,
        11,
    )
    result = np.zeros((19, 19), bool)
    for row, width in enumerate(widths):
        start = (19 - width) // 2
        result[row, start : start + width] = True
    return result


def source_close(mask):
    """Close with the source approximation and explicit unbounded-background pad."""
    from scipy.ndimage import binary_closing

    padded = np.pad(np.asarray(mask, bool), 9, mode="constant")
    return binary_closing(padded, structure=source_disk10())[9:-9, 9:-9]


def implicit_histcounts250_edges(values):
    """Measured double-grayscale250-bin rule; source rounding quirks retained.

    This fixed250-bin domain is independently checked against synthetic ranges,
    constants and subsampled real images. It is not a general histcounts clone.
    """
    data = np.asarray(values, np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Histogram requires finite nonempty source grayscale")
    lo, hi = float(data.min()), float(data.max())
    if lo < 0 or hi > 1:
        raise ValueError(
            "Native source histogram supports grayscale in [0,1]; use measured edges outside this domain"
        )
    if lo == hi:
        lower = math.floor(2 * lo - 0.5) / 2
        upper = math.ceil(2 * hi + 0.5) / 2
        return lower + np.arange(251, dtype=np.float64) * ((upper - lower) / 250)
    raw_width = (hi - lo) / 250
    if raw_width <= 0:
        raise NotImplementedError("Unrepresentable native histogram range; supply measured edges")
    magnitude = 10.0 ** math.floor(math.log10(raw_width))
    # A log10 result may round exactly to the next decimal exponent. Its
    # leading digit remains at least1; this preserves measured adjacent-double
    # .4/.65 boundary behavior without a numerical epsilon.
    if magnitude == 0:
        raise NotImplementedError(
            "Unrepresentable native histogram magnitude; supply measured edges"
        )
    coarse = max(1, math.floor(raw_width / magnitude)) * magnitude
    lower = math.floor(lo / coarse) * coarse
    target = (hi - lower) / 250
    if target / 250 <= 0:
        raise NotImplementedError(
            "Unrepresentable native histogram resolution; supply measured edges"
        )
    power = math.ceil(math.log10(target / 250))
    for _ in range(16):
        # Recompute decimal powers. Dividing a previous power by10 changes its
        # rounding and loses measured .4/.65-style one-unit width behavior.
        unit = 10.0**power
        if unit == 0:
            break
        width = math.ceil(target / unit) * unit
        if width / 250 >= unit:
            edges = lower + np.arange(251, dtype=np.float64) * width
            if edges[0] <= lo and edges[-1] >= hi and (np.diff(edges) > 0).all():
                return edges
            break
        power -= 1
    raise NotImplementedError("Unresolved native histogram numeric range; supply measured edges")


def source_histcounts250(values, *, oracle_edges=None):
    """Source250-bin grayscale counts, with an explicitly tagged oracle override."""
    data = np.asarray(values, np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Histogram requires finite nonempty source grayscale")
    edges = (
        implicit_histcounts250_edges(data)
        if oracle_edges is None
        else np.asarray(oracle_edges, np.float64)
    )
    if edges.shape != (251,) or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("Provide exactly 251 finite strictly increasing oracle edges")
    if edges[0] > data.min() or edges[-1] < data.max():
        raise ValueError("Oracle edges must cover all source grayscale values")
    return np.histogram(data, bins=edges)[0], edges.copy()


def source_seed_components(mask, seeds):
    """Source 8-connected membership, including background label if seeded."""
    from scipy.ndimage import label

    mask, seeds = np.asarray(mask, bool), np.asarray(seeds, bool)
    if mask.ndim != 2 or seeds.shape != mask.shape:
        raise ValueError("Component mask and seed field must share a 2D shape")
    labels, _ = label(mask, structure=np.ones((3, 3), bool))
    return np.isin(labels, np.unique(labels[seeds]))


def automatic_fov_stages(
    image,
    *,
    dataset="generic",
    multithresh_values=None,
    histogram_edges=None,
    threshold_backend="oracle",
):
    """Source threshold/components/close/fill stages, without inventing a radius.

    N3/4/5 remains guarded by default. Explicit source_first_cutoff uses the
    measured min-only contract; supplied vectors remain available for oracle
    studies. No global multi-Otsu or complete threshold clone is substituted.
    """
    from scipy.ndimage import binary_fill_holes

    gray = source_rgb2gray(image)
    if min(gray.shape) < 2 or max(gray.shape) <= 5:
        raise ValueError(
            "Source FOV study requires image dimensions matching its provided-stage domain"
        )
    name = str(dataset).lower().replace("-", "").replace("_", "")
    name = {
        "drishti": "drishtigs",
        "diaretdb0": "diaretdb",
        "diaretdb1": "diaretdb",
    }.get(name, name)
    n, factor = 0, 1.0
    if name in {"diaretdb", "review"}:
        n, factor = 4, 0.1
    elif name in {"drishtigs", "idrid", "messidor", "riga", "roc", "vicavr", "avrdb"}:
        n = 5
    elif name == "stare":
        n = 3
    elif name in {"eophtha", "ridb"}:
        n, factor = 5, 0.5
    if threshold_backend not in {"oracle", "source_first_cutoff"}:
        raise ValueError("Choose oracle or explicit source_first_cutoff threshold backend")
    if threshold_backend != "oracle" and multithresh_values is not None:
        raise ValueError("Choose measured thresholds or source_first_cutoff, not both")
    cutoff_metadata = None
    if n:
        if multithresh_values is None:
            if threshold_backend == "oracle":
                raise NotImplementedError(
                    f"Source N={n} first-cutoff backend is opt-in; supply measured thresholds or explicit source_first_cutoff"
                )
            from .matlab_firstcut import first_cutoff_source

            cutoff = first_cutoff_source(gray, n)
            low = cutoff.value * factor
            cutoff_metadata = asdict(cutoff)
        else:
            thresholds = np.asarray(multithresh_values, np.float64)
            if thresholds.shape != (n,) or not np.isfinite(thresholds).all():
                raise ValueError(f"Provide exactly {n} finite source multithresh values")
            low = float(thresholds.min()) * factor
    else:
        if multithresh_values is not None:
            raise ValueError("This source profile does not use multithresh")
        low = 0.0
    high = float(gray.max()) * (0.4 if name == "inspireavr" else 0.8)

    def components(cutoff):
        return source_seed_components(gray > cutoff, gray > high)

    selected = components(low)
    initial_closed = source_close(selected)
    white = float(initial_closed.sum() / initial_closed.size)
    expected_black = 1 - white
    tolerance = 0.1
    if name in {"aria", "g1020"}:
        expected_black = 0.1343
    elif name == "diaretdb":
        expected_black = 0.1634
    elif name == "stare":
        expected_black = 0.269
    elif name == "drishtigs":
        expected_black = 0.2929 if gray.shape[1] > 2200 else 0.1568
        tolerance = 0.05
    elif name == "eophtha":
        expected_black = 0.45
        tolerance = 0.15
    fallback = white > (1 - expected_black + tolerance) or white < (1 - expected_black - tolerance)
    closed = initial_closed
    if fallback:
        counts, edges = source_histcounts250(gray, oracle_edges=histogram_edges)
        index = int(np.searchsorted(np.cumsum(counts), expected_black * gray.size, side="right"))
        low = float(edges[index])
        high = float(gray.max()) * 0.8
        selected = components(low)
        closed = source_close(selected)
    filled = binary_fill_holes(closed, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    return {
        "gray": gray,
        "selected": selected,
        "initial_closed": initial_closed,
        "closed": closed,
        "filled": filled,
        "dataset": name,
        "low_threshold": low,
        "high_threshold": high,
        "fallback_used": bool(fallback),
        "multithresh_backend": (
            "source_first_cutoff" if cutoff_metadata is not None else "supplied_oracle_vector"
        )
        if n
        else "not_required",
        "first_cutoff": cutoff_metadata,
        "histogram_backend": (
            "python_source_histcounts250" if histogram_edges is None else "supplied_oracle_edges"
        )
        if fallback
        else "not_required",
        "initial_white_fraction": white,
        "expected_black_fraction": expected_black,
    }


def automatic_fov_source(
    image,
    *,
    dataset="generic",
    multithresh_values=None,
    histogram_edges=None,
    threshold_backend="oracle",
):
    """Optional source automatic FOV for measured primitive/profile subsets only."""
    from dataclasses import replace

    stages = automatic_fov_stages(
        image,
        dataset=dataset,
        multithresh_values=multithresh_values,
        histogram_edges=histogram_edges,
        threshold_backend=threshold_backend,
    )
    name = stages["dataset"]
    # Source RIM uses fov AFTER fill as its boundary; generic/ROI uses final_closed.
    boundary_input = stages["filled"] if name in {"rim", "rimone"} else stages["closed"]
    result = provided_fov_source(boundary_input, dataset=name)
    processing = np.ones_like(stages["filled"]) if name in {"rim", "rimone"} else stages["filled"]
    branch = (
        "automatic_rimone_filled_boundary"
        if name in {"rim", "rimone"}
        else result.source_branch.replace("provided_", "automatic_", 1)
    )
    diagnostics = {key: value for key, value in stages.items() if not isinstance(value, np.ndarray)}
    return replace(
        result,
        processing_fov=processing,
        closed_mask=stages["closed"],
        source_branch=branch,
        diagnostics=diagnostics,
    )
