"""Compare two completed runs using validation only; never load test metrics.

uv run python scripts/compare_runs.py runs/drive_bce runs/drive_bce_dice
"""

import argparse
import json
from pathlib import Path


def compare(baseline: Path, candidate: Path) -> dict:
    paths = [baseline, candidate]
    metadata = [json.loads((path / "run.json").read_text()) for path in paths]
    if metadata[0]["manifest_sha256"] != metadata[1]["manifest_sha256"]:
        raise ValueError("Runs must use the identical manifest")
    for key in (
        "task",
        "seed",
        "epochs",
        "batch_size",
        "patch_size",
        "base_channels",
        "depth",
        "samples_per_image",
        "lr",
        "tile_size",
        "overlap",
        "max_steps",
    ):
        if metadata[0]["config"][key] != metadata[1]["config"][key]:
            raise ValueError(f"Runs differ in {key}; this is not a controlled loss comparison")
    metrics = [json.loads((path / "validation.json").read_text()) for path in paths]
    scores = [metric["mean_dice"] for metric in metrics]
    if any(score is None for score in scores):
        raise ValueError("Undefined validation scores")
    best = 1 if scores[1] > scores[0] else 0
    return {
        "selection_split": "val",
        "manifest_sha256": metadata[0]["manifest_sha256"],
        "baseline": {
            "run": str(baseline),
            "loss": metadata[0]["config"]["loss"],
            "mean_dice": scores[0],
        },
        "candidate": {
            "run": str(candidate),
            "loss": metadata[1]["config"]["loss"],
            "mean_dice": scores[1],
        },
        "candidate_minus_baseline": scores[1] - scores[0],
        "selected_checkpoint": str(paths[best] / "best.pt"),
        "caveat": "Single seed; validation-selected best epochs. Not evidence of cross-dataset superiority.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
