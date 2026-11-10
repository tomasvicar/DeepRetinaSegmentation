"""Clean-room scalar-scale bicubic resizing for the MATLAB AV profile.

Uses public cubic-convolution mathematics, half-pixel centers, symmetric boundary
extension and antialiasing while shrinking. No proprietary toolbox source read.
Only float64 image calculations and logical nearest masks are in scope.
"""

from __future__ import annotations

import math

import numpy as np


def _cubic(distance):
    distance = np.abs(distance)
    near = ((1.5 * distance - 2.5) * distance * distance + 1) * (distance <= 1)
    far = ((-0.5 * distance + 2.5) * distance - 4) * distance + 2
    return near + far * ((distance > 1) & (distance < 2))


def imresize_bicubic(image, scale):
    """Resize float HW/HWC with scalar scale, ceil shape and antialiasing.

    Input and output are float64. Overshoot is retained, as in the MATLAB double
    image path: clipping before illumination correction would alter its semantics.
    """
    image = np.asarray(image)
    if (
        image.ndim not in (2, 3)
        or not all(image.shape)
        or not np.issubdtype(image.dtype, np.floating)
    ):
        raise ValueError("Expected a nonempty floating HW/HWC image")
    if not np.isfinite(image).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Image must be finite and scale positive/finite")
    result = image.astype(np.float64, copy=True)
    if scale == 1:
        return result
    support = 4 / scale if scale < 1 else 4
    for axis in (0, 1):
        length = result.shape[axis]
        output_length = math.ceil(length * scale)
        # One-based arithmetic preserves the source coordinate convention.
        position = np.arange(1, output_length + 1) / scale + 0.5 * (1 - 1 / scale) - 1
        left = np.floor(position - support / 2).astype(np.int64)
        indices = left[:, None] + np.arange(math.ceil(support) + 2)[None, :]
        distance = position[:, None] - indices
        weights = scale * _cubic(scale * distance) if scale < 1 else _cubic(distance)
        weights /= weights.sum(axis=1, keepdims=True)
        indices %= 2 * length
        indices = np.where(indices < length, indices, 2 * length - 1 - indices)
        shape = list(result.shape)
        shape[axis] = output_length
        resized = np.zeros(shape, dtype=np.float64)
        weight_shape = [1] * result.ndim
        weight_shape[axis] = output_length
        for tap in range(weights.shape[1]):
            if weights[:, tap].any():
                resized += np.take(result, indices[:, tap], axis=axis) * weights[:, tap].reshape(
                    weight_shape
                )
        result = resized
    return result


def imresize_nearest_mask(mask, shape):
    """Resize a categorical HW mask to explicit shape using nearest centers."""
    mask = np.asarray(mask)
    dimensions = np.asarray(shape)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("Expected a nonempty2D mask")
    if (
        dimensions.shape != (2,)
        or not np.isfinite(dimensions).all()
        or (dimensions < 1).any()
        or not np.equal(dimensions, np.floor(dimensions)).all()
    ):
        raise ValueError("shape must contain two positive integers")
    rows = np.floor((np.arange(int(dimensions[0])) + 0.5) * mask.shape[0] / dimensions[0]).astype(
        np.int64
    )
    columns = np.floor(
        (np.arange(int(dimensions[1])) + 0.5) * mask.shape[1] / dimensions[1]
    ).astype(np.int64)
    return mask[
        np.clip(rows, 0, mask.shape[0] - 1)[:, None],
        np.clip(columns, 0, mask.shape[1] - 1)[None, :],
    ].copy()
