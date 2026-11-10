"""Replay research candidate against synthetic public-MATLAB operator goldens."""

import argparse
import json
from pathlib import Path

import numpy as np
from candidate_sobel import candidate
from diagnose_sobel import sobel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with np.load(args.fixture, allow_pickle=False) as data:
        for index, name in enumerate(data["names"]):
            distance = data[f"{index}_distance"]
            result = {"name": str(name), "shape": list(distance.shape), "methods": {}}
            for label, operator in [("legacy", sobel), ("research_candidate", candidate)]:
                result["methods"][label] = {}
                for axis, value in zip(["gx", "gy"], operator(distance), strict=True):
                    ref = data[f"{index}_{axis}"]
                    delta = np.abs(value.astype(float) - ref.astype(float))
                    result["methods"][label][axis] = {
                        "unequal": int(np.count_nonzero(value != ref)),
                        "max_abs": float(delta.max()),
                    }
            rows.append(result)
    args.output.write_text(
        json.dumps(
            {"production_activation": False, "full_parity_claim": False, "cases": rows}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
