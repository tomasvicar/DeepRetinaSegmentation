"""Provided-mask calibration following the user's original souradnice.m.

No automatic FOV thresholding, bbox circle fit, or dataset-specific exception is
silently substituted. The binary Sobel stage is independently checked against
public MATLAB edge calls; see docs/MATLAB_FOV_CALIBRATION.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel


@dataclass(frozen=True)
class FOVCalibration:
    center_row_one_based: int
    center_col_one_based: int
    radius_pixels: int
    edge_pixels: int

    @property
    def diameter_pixels(self):
        return 2 * self.radius_pixels


def binary_sobel_edges(mask, *, return_stages=False):
    """Default Sobel edge candidate for an explicitly binary provided FOV.

    Replicate-border Sobel, automatic 2*RMS gradient threshold and directional
    discrete nonmaximum suppression. Plateau ties retain the right/bottom side.
    This implements only the binary-mask contract, not generic grayscale edge.
    """
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-dimensional binary FOV, dimensions >=2")
    levels = np.unique(values)
    if len(levels) > 2 or (len(levels) == 2 and levels[0] != 0) or (levels < 0).any():
        raise ValueError("Threshold the provided FOV explicitly to binary values")
    values = (values != 0).astype(np.float32)
    gx = -sobel(values, axis=1, mode="nearest") / 8
    gy = -sobel(values, axis=0, mode="nearest") / 8
    magnitude = gx * gx + gy * gy
    threshold = 2 * math.sqrt(float(np.mean(magnitude, dtype=np.float64)))
    padded = np.pad(magnitude, 1, mode="constant")
    horizontal = (
        (np.abs(gx) >= np.abs(gy))
        & (magnitude >= padded[1:-1, :-2])
        & (magnitude > padded[1:-1, 2:])
    )
    vertical = (
        (np.abs(gy) >= np.abs(gx))
        & (magnitude >= padded[:-2, 1:-1])
        & (magnitude > padded[2:, 1:-1])
    )
    strong = magnitude.astype(np.float64) > threshold**2
    edges = strong & (horizontal | vertical)
    if return_stages:
        return edges, {"threshold": threshold, "gx": gx, "gy": gy, "unthinned": strong}
    return edges


def _mode_smallest(values):
    values, counts = np.unique(values, return_counts=True)
    if not len(values):
        raise ValueError("Provided FOV has no usable paired edge intersections")
    return values[np.argmax(counts)]


def calibrate_provided_fov(mask, *, dataset=None):
    """Estimate the original generic center/radius from a provided binary mask.

    Coordinates are MATLAB one-based; original `prumer` actually denotes radius.
    ROC/RIGA/RIM-ONE/Erlangen special source branches are explicitly unsupported.
    Empty/full/edge-free masks fail closed instead of returning a bbox heuristic.
    This helper is experimental and does not change inference defaults.
    """
    name = (dataset or "").lower().replace("-", "").replace("_", "")
    if name in {"rim", "rimone", "roc"} or name.startswith(("erlangen", "riga")):
        raise ValueError(
            "This dataset uses a special souradnice branch; supply explicit calibration"
        )
    edges = binary_sobel_edges(mask)

    def midpoints(array):
        counts = array.sum(axis=1)
        selected = array[counts >= 2]
        if not len(selected):
            return np.array([])
        first = selected.argmax(axis=1)
        last = selected.shape[1] - 1 - selected[:, ::-1].argmax(axis=1)
        return (first + last) / 2 + 1

    col = math.floor(float(_mode_smallest(midpoints(edges))) + 0.5)
    row = math.floor(float(_mode_smallest(midpoints(edges.T))) + 0.5)
    y, x = np.where(edges)
    distances = np.sqrt(
        (x.astype(np.float64) + 1 - col) ** 2 + (y.astype(np.float64) + 1 - row) ** 2
    )
    radius = math.floor(float(_mode_smallest(distances)) + 0.5) + 1
    return FOVCalibration(row, col, radius, int(edges.sum()))
