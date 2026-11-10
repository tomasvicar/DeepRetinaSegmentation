"""Explicit reviewed stage2 held-out evaluation; general test CLI stays blocked."""

import argparse
from pathlib import Path

import torch
from deepretina.heldout import evaluate_reviewed_heldout
from deepretina.selection import SelectionError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "selection",
        "expected-selection-sha256",
        "checkpoint",
        "manifest",
        "validation-evidence",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Derive last.pt, metrics.csv and training_summary.json; explicit options take priority",
    )
    for name in ("last-checkpoint", "metrics-csv", "training-summary"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--execute-reviewed-heldout",
        action="store_true",
        help="Explicit stage2 execution request; never changes the metadata-only stage1 receipt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = vars(parser.parse_args(argv))
    if not args["execute_reviewed_heldout"]:
        parser.error("--execute-reviewed-heldout is required for this separate stage2 route")
    if args["cpu_threads"] < 1:
        parser.error("--cpu-threads must be positive")
    run_dir = args.pop("run_dir")
    for key, filename in (
        ("last_checkpoint", "last.pt"),
        ("metrics_csv", "metrics.csv"),
        ("training_summary", "training_summary.json"),
    ):
        if args[key] is None:
            if run_dir is None:
                parser.error(f"--{key.replace('_', '-')} or --run-dir is required")
            args[key] = run_dir / filename
    torch.set_num_threads(args.pop("cpu_threads"))
    try:
        evaluate_reviewed_heldout(**args)
    except (SelectionError, ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"unified-test: {error}\n")
    print(args["output"])


if __name__ == "__main__":
    main()
