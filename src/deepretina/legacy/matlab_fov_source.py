"""Optional source-policy FOV port, separate from public provided-FOV defaults.

Calibration boundaries and returned processing FOVs are deliberately distinct.
This preserves the original RIM-ONE and pre-hole-fill semantics rather than
silently substituting a robust circle fit or the single_net Python loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .matlab_fov import FOVCalibration, binary_sobel_edges


@dataclass(frozen=True)
class MatlabSourceFOV:
    calibration_boundary: np.ndarray
    processing_fov: np.ndarray
    calibration: FOVCalibration
    source_branch: str
    closed_mask: np.ndarray


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
