"""Separate, clean-room uniform CLAHE profile for the original MATLAB AV path.

Public algorithm documentation and generated MATLAB R2026b black-box fixtures
are the references. No proprietary toolbox implementation was inspected. This
module does not alter the historical single_net Python normalization profile.
"""

from __future__ import annotations

import math

import numpy as np

AV_KERNEL_PIXELS = 250 / (55 / 25)
AV_CLIP_LIMIT = 0.005
AV_NBINS = 256


def _clip_histogram(histogram: np.ndarray, limit: int) -> np.ndarray:
    """Clip and redistribute counts without losing pixels or exceeding the cap."""
    histogram = np.asarray(histogram, dtype=np.int64)
    result = np.minimum(histogram, limit)
    excess = int((histogram - result).sum())
    increment = excess // len(result)
    additions = np.minimum(limit - result, increment)
    result += additions
    excess -= int(additions.sum())
    while excess:
        previous = excess
        for start in range(len(result)):
            step = max(len(result) // excess, 1)
            for index in range(start, len(result), step):
                if result[index] < limit:
                    result[index] += 1
                    excess -= 1
                if not excess:
                    break
            if not excess:
                break
        if excess == previous:
            raise ValueError("Histogram cap has insufficient capacity")
    return result


def adapthisteq_uniform(image, *, num_tiles=(8, 8), clip_limit=0.01, nbins=256):
    """Uniform/full-range MATLAB-style CLAHE for finite normalized float images.

    Computation and output use float64. This intentionally supports the original
    AV pipeline's double-valued V plane, not every MATLAB class/distribution/Range
    option. Float32 inputs are promoted; integer/color/NaN/out-of-range inputs are
    rejected rather than silently reinterpreted. Tiles are padded symmetrically
    to even sizes. The result retains the original shape and is not globally
    stretched to the observed output extrema.
    """
    image = np.asarray(image)
    if image.ndim != 2 or not all(image.shape) or not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Expected a nonempty2D normalized floating image")
    image = image.astype(np.float64, copy=False)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite intensities in[0,1]")
    tiles = np.asarray(num_tiles)
    if (
        tiles.shape != (2,)
        or not np.isfinite(tiles).all()
        or (tiles < 2).any()
        or not np.equal(tiles, np.floor(tiles)).all()
    ):
        raise ValueError("num_tiles must contain two integers>=2")
    if isinstance(nbins, bool) or not isinstance(nbins, (int, np.integer)) or nbins < 2:
        raise ValueError("nbins must be an integer>=2")
    if not math.isfinite(clip_limit) or not 0 <= clip_limit <= 1:
        raise ValueError("clip_limit must be in[0,1]")
    tiles = tiles.astype(np.int64)
    shape = np.asarray(image.shape, dtype=np.int64)
    size = (shape + tiles - 1) // tiles
    size += size % 2
    extra = size * tiles - shape
    before = extra // 2
    padded = np.pad(image, list(zip(before, extra - before, strict=True)), mode="symmetric")
    indices = np.floor(padded * (nbins - 1) + 0.5).astype(np.int64)
    pixels = int(np.prod(size))
    minimum = math.ceil(pixels / nbins)
    limit = minimum + math.floor((pixels - minimum) * clip_limit + 0.5)
    maps = np.empty((*tiles, nbins), dtype=np.float64)
    for y in range(tiles[0]):
        for x in range(tiles[1]):
            region = indices[y * size[0] : (y + 1) * size[0], x * size[1] : (x + 1) * size[1]]
            histogram = np.bincount(region.ravel(), minlength=nbins)
            maps[y, x] = np.cumsum(_clip_histogram(histogram, limit)) / pixels
    y = (np.arange(padded.shape[0]) - size[0] / 2) / size[0]
    x = (np.arange(padded.shape[1]) - size[1] / 2) / size[1]
    y0, x0 = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    dy, dx = (y - y0)[:, None], (x - x0)[None, :]
    y1, x1 = np.clip(y0 + 1, 0, tiles[0] - 1), np.clip(x0 + 1, 0, tiles[1] - 1)
    y0, x0 = np.clip(y0, 0, tiles[0] - 1), np.clip(x0, 0, tiles[1] - 1)
    result = (
        maps[y0[:, None], x0[None, :], indices] * (1 - dy) * (1 - dx)
        + maps[y0[:, None], x1[None, :], indices] * (1 - dy) * dx
        + maps[y1[:, None], x0[None, :], indices] * dy * (1 - dx)
        + maps[y1[:, None], x1[None, :], indices] * dy * dx
    )
    np.clip(result, 0, 1, out=result)  # Convex interpolation can overshoot by a few ulps.
    return result[before[0] : before[0] + shape[0], before[1] : before[1] + shape[1]]


def av_matlab_clahe_v(value):
    """Apply the user-owned AV MATLAB source's fixed25px photometric parameters.

    The source's local kernel remains fixed even if outer geometry is35px/degree.
    This is a CLAHE-only stage; raw AV preprocessing remains uncertified.
    """
    shape = np.asarray(np.shape(value))
    if shape.shape != (2,):
        raise ValueError("Expected a2D V plane")
    tiles = np.floor(shape / AV_KERNEL_PIXELS + 0.5).astype(np.int64)
    return adapthisteq_uniform(value, num_tiles=tiles, clip_limit=AV_CLIP_LIMIT, nbins=AV_NBINS)
