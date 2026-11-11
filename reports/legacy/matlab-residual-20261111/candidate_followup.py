"""Research v2: evidenced row-filter association at the terminal Gy borders."""

import numpy as np
from candidate_sobel import candidate as initial_candidate


def candidate(distance):
    gx, gy = initial_candidate(distance)
    h, w = distance.shape
    if h < 16 or w < 16:
        return gx, gy
    p = np.pad(distance, 1, mode="edge")
    start = ((h - 1) // 16) * 16
    dy = p[2:] - p[:-2]
    left, middle, right = dy[:, :-2], dy[:, 1:-1], dy[:, 2:]
    filtered = (left + 2 * middle) + right
    filtered[:, 0] = left[:, 0] + (2 * middle[:, 0] + right[:, 0])
    gy[start:] = filtered[start:]
    return gx, gy
