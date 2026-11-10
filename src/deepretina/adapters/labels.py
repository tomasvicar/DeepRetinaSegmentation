"""Shared annotation decoders used by both dataset preparation paths."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def decode_av(rgb: np.ndarray, vessel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode RGB A/V labels; uncertain/crossing/unlabelled vessels are ignored.

    Artery and vein losses are supervised on confidently labelled vessels only.
    The independent vessel channel uses the original binary segmentation.
    """
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    artery = (r > 180) & (g < 100) & (b < 100)
    vein = (b > 180) & (g < 100) & (r < 100)
    known = (artery | vein) & vessel
    target = np.stack([vessel, artery & vessel, vein & vessel]).astype(np.float32)
    valid = np.stack([np.ones_like(vessel, dtype=bool), known, known])
    return target, valid


def decode_refuge(mask: np.ndarray) -> np.ndarray:
    """REFUGE: 255 background, 128 disc rim, 0 cup (disc includes cup)."""
    return np.stack([mask < 150, mask < 50]).astype(np.float32)


def decode_iostar_disc(mask: np.ndarray) -> np.ndarray:
    """IOSTAR OD is the enclosed hole; exclude the exterior connected background."""
    labels, _ = ndimage.label(~mask.astype(bool), np.ones((3, 3), bool))
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    disc = (labels > 0) & ~np.isin(labels, border)
    if not disc.any():
        raise ValueError("IOSTAR optic-disc mask has no enclosed disc region")
    return disc
