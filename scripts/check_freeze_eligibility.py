"""Check typed completion/own-weight freeze eligibility on CPU; no selection/TEST."""

import argparse
import json
from pathlib import Path

import torch

from deepretina.freeze_eligibility import PROFILES, ROLES, check_freeze_eligibility
from deepretina.selection import SelectionError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "checkpoint",
        "best-checkpoint",
        "last-checkpoint",
        "manifest",
        "metrics-csv",
        "training-summary",
        "completion-qa",
        "expected-qa-sha256",
        "family",
        "checkpoint-role",
        "output",
    ):
        parser.add_argument("--" + option, required=True)
    for option in (
        "reviewed-protocol",
        "expected-protocol-sha256",
        "original-terminal",
        "original-pool-evidence",
        "expected-pool-evidence-sha256",
        "expected-terminal-sha256",
    ):
        parser.add_argument("--" + option)
    args = vars(parser.parse_args())
    if args["family"] not in PROFILES or args["checkpoint_role"] not in ROLES:
        parser.error("Require explicit supported family and checkpoint role")
    output = Path(args.pop("output"))
    if output.exists() or output.is_symlink():
        parser.error("Preserve previous eligibility proof")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        checked = check_freeze_eligibility(**args)
    except (SelectionError, ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"freeze-eligibility: {error}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(checked.proof, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "checkpoint_sha256": checked.proof["checkpoint_sha256"],
                "checkpoint_role": checked.proof["checkpoint_role"],
                "selection_created": False,
                "test_evaluation_enabled": False,
            }
        )
    )


if __name__ == "__main__":
    main()
