"""Research coordinate association matching bicubic's one-based inverse map."""

import numpy as np


def resize(mask, shape):
    indices = []
    for source_length, output_length in zip(mask.shape, shape):
        scale = output_length / source_length
        u = np.arange(1, output_length + 1, dtype=np.float64)
        position = u / scale + 0.5 * (1 - 1 / scale) - 1
        indices.append(np.clip(np.floor(position + 0.5).astype(np.int64), 0, source_length - 1))
    return mask[indices[0][:, None], indices[1][None, :]].copy()
