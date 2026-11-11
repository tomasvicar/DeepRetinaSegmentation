"""Validate original curated_fd44 completion on CPU; no inference/freeze/TEST."""

import argparse
import json
from pathlib import Path

import torch

from deepretina.curated_completion import check_curated_completion
from deepretina.selection import SelectionError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "original-index",
        "expected-index-sha256",
        "manifest",
        "checkpoint-role",
        "family",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    output = Path(args.pop("output"))
    if output.exists() or output.is_symlink():
        parser.error("Preserve previous proof")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        checked = check_curated_completion(**args)
    except (SelectionError, ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, "curated-completion: " + str(error) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(checked.proof, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "checkpoint_sha256": checked.proof["checkpoint_sha256"],
                "checkpoint_role": checked.proof["checkpoint_role"],
                "completion_checked": True,
                "selection_created": False,
                "test_evaluation_enabled": False,
            }
        )
    )


if __name__ == "__main__":
    main()
