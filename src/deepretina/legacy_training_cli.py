"""Stdlib-only native retraining parser; Torch is imported only at execution."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def add_cli_args(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-content-sha256")
    parser.add_argument("--task", choices=("vessel", "disc", "universal", "av"))
    for key in ("epochs", "epoch-samples", "batch-size", "patch-size", "seed"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    return parser


def train_from_args(args):
    from .legacy_training import LegacyTrainingConfig, train

    values = {}
    if args.config:
        values = (
            json.loads(args.config.read_text())
            if args.config.suffix == ".json"
            else tomllib.loads(args.config.read_text())
        )
    for key in (
        "task",
        "epochs",
        "epoch_samples",
        "batch_size",
        "patch_size",
        "seed",
        "lr",
        "device",
    ):
        if getattr(args, key, None) is not None:
            values[key] = getattr(args, key)
    config = LegacyTrainingConfig(**values)
    checkpoint = train(
        args.manifest,
        args.output,
        config,
        expected_content_sha256=getattr(args, "expected_content_sha256", None),
    )
    return {
        "checkpoint": str(checkpoint),
        "recipe": "native_legacy_canonical_fresh_v1",
        "task": config.task,
        "fresh_initialization": True,
        "historical_optimizer_reproduction": False,
    }


def main(argv=None):
    parser = add_cli_args(argparse.ArgumentParser(description=__doc__))
    print(json.dumps(train_from_args(parser.parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
