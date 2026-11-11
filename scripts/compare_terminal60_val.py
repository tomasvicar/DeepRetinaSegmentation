"""Explicit fixed terminal60 baseline VAL comparison; never selection or TEST."""

import argparse
import json
from pathlib import Path

from compare_frozen_completed_val import compare as compare_frozen


def compare(**kwargs):
    if "checkpoint_role" in kwargs:
        raise ValueError("This route has exactly one fixed terminal60 role")
    return compare_frozen(**kwargs, checkpoint_role="fixed_terminal60")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "reference",
        "candidate",
        "manifest",
        "completion-qa",
        "reference-context",
        "candidate-context",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("trusted-reference-context-sha256", "trusted-candidate-context-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    a = vars(parser.parse_args())
    output = a.pop("output")
    a["repeats"] = a.pop("draws")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve prior terminal60 comparison")
    result = compare(**a)
    with output.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
