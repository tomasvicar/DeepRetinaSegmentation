"""Create a new explicit VAL770 reference context; never bridge TEST or rerun a model."""

import argparse
import json
from pathlib import Path

from deepretina.val_bridge import build_bridge, publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "target", "original-reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-parent-content", "expected-target-content"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve existing bridge artifacts")
    for corpus in (a.parent, a.target):
        if a.output.resolve().is_relative_to(corpus.resolve()):
            raise ValueError("Publish bridge outside corpus inputs")
    bridge, reference = build_bridge(
        a.parent,
        a.target,
        a.original_reference,
        expected_parent_content=a.expected_parent_content,
        expected_target_content=a.expected_target_content,
    )
    print(json.dumps(publish(a.output, bridge, reference)))


if __name__ == "__main__":
    main()
