"""Research-only observed R2026b CPU path model; NOT a production default.

The 16-row terminal block and left-column traversal are public-oracle findings,
not image/content thresholds. Gy boundary and tiny-frame behavior remain open.
"""

import numpy as np
from diagnose_sobel import sobel


def candidate(distance):
    gx, gy = sobel(distance)
    h, w = distance.shape
    if h < 16 or w < 16:
        return gx, gy
    p = np.pad(distance, 1, mode="edge")
    start = ((h - 1) // 16) * 16
    left = p[2:, 2:].copy()
    left += 2 * p[1:-1, 2:]
    left += p[:-2, 2:]
    left -= p[2:, :-2]
    left -= 2 * p[1:-1, :-2]
    left -= p[:-2, :-2]
    gx[:start, 0] = left[:start, 0]
    smooth = p[:-2] + 2 * p[1:-1] + p[2:]
    tail = smooth[:, 2:] - smooth[:, :-2]
    gx[start:] = tail[start:]
    dy = p[2:] - p[:-2]
    tail = dy[:, :-2] + 2 * dy[:, 1:-1] + dy[:, 2:]
    gy[start:, 1:-1] = tail[start:, 1:-1]
    return gx, gy
