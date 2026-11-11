"""Create/check frozen selection metadata on CPU; does not run or enable a test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepretina.selection import SelectionError, check_selection, create_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "check"):
        cmd = commands.add_parser(name)
        for option in (
            "checkpoint",
            "last-checkpoint",
            "manifest",
            "validation-evidence",
            "metrics-csv",
            "training-summary",
        ):
            cmd.add_argument("--" + option, type=Path, required=True)
        for option in (
            "completion-family",
            "checkpoint-role",
            "completion-qa",
            "expected-qa-sha256",
        ):
            cmd.add_argument("--" + option, required=True)
        for option in (
            "best-checkpoint",
            "reviewed-protocol",
            "expected-protocol-sha256",
            "original-terminal",
            "original-pool-evidence",
            "expected-pool-evidence-sha256",
            "expected-terminal-sha256",
        ):
            cmd.add_argument("--" + option)
        cmd.add_argument("--cpu-threads", type=int, choices=[1, 2], default=2)
        cmd.add_argument("--threshold", type=float, default=0.5)
        cmd.add_argument("--tile-size", type=int)
        cmd.add_argument("--overlap", type=int)
        if name == "create":
            cmd.add_argument("--reviewer", required=True)
            cmd.add_argument("--rationale", required=True)
            cmd.add_argument("--output", type=Path, required=True)
        else:
            cmd.add_argument("--selection", type=Path, required=True)
            cmd.add_argument("--expected-selection-sha256", required=True)
    args = vars(parser.parse_args())
    action = args.pop("command")
    torch.set_num_threads(args.pop("cpu_threads"))
    try:
        if action == "create":
            result = create_selection(**args)
        else:
            checked = check_selection(**args)
            result = {
                "valid_frozen_selection": True,
                "selection_sha256": checked.selection_sha256,
                "checkpoint_sha256": checked.checkpoint_sha256,
                "test_evaluation_enabled": False,
            }
    except (SelectionError, ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"unified-selection: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
