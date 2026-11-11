"""Replay retained MATLAB nearest-mask goldens; no oracle/source mutation."""

import argparse
import json
from pathlib import Path

import numpy as np
from nearest_research import resize
from scipy.io import loadmat

from deepretina.legacy.matlab_resize import imresize_nearest_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=Path, required=True)
    p.add_argument("--chain", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path)
    a = p.parse_args()
    rows = []
    fixture = {}
    for i, c in enumerate(loadmat(a.synthetic, simplify_cells=True)["cases"]):
        mask, expected = np.atleast_2d(c["mask"]), np.atleast_2d(c["nearest"])
        fixture.update({f"{i}_input": mask, f"{i}_expected": expected})
        rows.append(
            {
                "name": c["name"],
                "current_unequal": int(
                    np.count_nonzero(imresize_nearest_mask(mask, expected.shape) != expected)
                ),
                "research_unequal": int(np.count_nonzero(resize(mask, expected.shape) != expected)),
            }
        )
    for c in loadmat(a.chain, simplify_cells=True)["cases"]:
        mask, expected = c["processing_fov"], c["resized_fov"]
        rows.append(
            {
                "name": c["name"],
                "current_unequal": int(
                    np.count_nonzero(imresize_nearest_mask(mask, expected.shape) != expected)
                ),
                "research_unequal": int(np.count_nonzero(resize(mask, expected.shape) != expected)),
            }
        )
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
    if a.fixture:
        np.savez_compressed(a.fixture, **fixture)


if __name__ == "__main__":
    main()
