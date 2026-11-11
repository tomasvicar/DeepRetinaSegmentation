"""Compare paired saved prediction reports without running models or reading masks."""

import argparse
import json
from pathlib import Path

from deepretina.comparison import compare_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Include source-specific counts and paired intervals",
    )
    parser.add_argument(
        "--include-av",
        action="store_true",
        help="Include conditional and end-to-end A/V paired counts",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier comparisons; choose a new output")
    report = compare_reports(
        args.reference,
        args.candidate,
        args.manifest,
        repeats=args.draws,
        seed=args.seed,
        include_sources=args.include_sources,
        include_av=args.include_av,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["macro_dataset_mean_dice"]))
