"""Command line entry points. Run ``uv run deepretina --help``."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict, fields
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and apply retinal segmentation models.")
    p.add_argument("--version", action="version", version="deepretina 0.1.0")
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert supported dataset ZIPs into a manifest")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="drive rite hrf drhagis chasedb1 iostar refuge drishti",
    )
    prepare.add_argument("--max-size", type=int, default=1024)
    prepare.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train from scratch and select best validation model")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--config", type=Path, help="TOML settings; explicit command options override"
    )
    train.add_argument("--task", choices=["vessels", "av", "disc_cup"])
    for name in (
        "epochs",
        "batch_size",
        "patch_size",
        "base_channels",
        "depth",
        "seed",
        "workers",
        "samples_per_image",
        "max_steps",
        "tile_size",
        "overlap",
    ):
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--loss", choices=["bce", "bce_dice"])
    train.add_argument("--device", help="auto, cpu, cuda, cuda:0, ...")

    evaluate = commands.add_parser("evaluate", help="Score a held-out split with tiled inference")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", type=Path, required=True, help="Metrics JSON")

    predict = commands.add_parser(
        "predict", help="Predict an image or directory at original resolution"
    )
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    for cmd in (evaluate, predict):
        cmd.add_argument("--checkpoint", type=Path, required=True)
        cmd.add_argument("--device", default="auto")
        cmd.add_argument("--tile-size", type=int, default=512)
        cmd.add_argument("--overlap", type=int, default=64)
        cmd.add_argument("--threshold", type=float, default=0.5)
    inspect = commands.add_parser("inspect", help="Summarize a prepared manifest")
    inspect.add_argument("manifest", type=Path)
    commands.add_parser("doctor", help="Check Python, PyTorch and CUDA")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, f"deepretina: {exc}\n")
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


def dispatch(args: argparse.Namespace):
    if args.command == "prepare":
        from .prepare import prepare

        manifest = prepare(args.data_root, args.output, args.datasets, args.max_size, args.seed)
        return {"manifest": str(manifest)}
    if args.command == "train":
        from .engine import TrainConfig, train

        settings = {}
        if args.config:
            settings = tomllib.loads(args.config.read_text())
        allowed = {field.name for field in fields(TrainConfig)}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        for name in allowed:
            value = getattr(args, name, None)
            if value is not None:
                settings[name] = value
        if not settings.get("task"):
            raise ValueError("Specify --task or task in the TOML config")
        config = TrainConfig(**settings)
        print(json.dumps(asdict(config), indent=2), flush=True)
        return {"checkpoint": str(train(args.manifest, args.output, config))}
    if args.command in ("evaluate", "predict"):
        from .engine import evaluate
        from .inference import predict

        options = {
            "device": args.device,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "threshold": args.threshold,
        }
        if args.command == "evaluate":
            result = evaluate(args.checkpoint, args.manifest, split=args.split, **options)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return result
        if args.input.is_dir():
            images = sorted(
                path
                for path in args.input.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
            )
            if not images:
                raise ValueError(f"No supported images in {args.input}")
            return [
                predict(args.checkpoint, path, args.output / path.name, **options)
                for path in images
            ]
        return predict(args.checkpoint, args.input, args.output, **options)
    if args.command == "inspect":
        from collections import Counter

        records = json.loads(args.manifest.read_text())
        counts = Counter((record["dataset"], record["task"], record["split"]) for record in records)
        return {
            "records": len(records),
            "groups": [
                {"dataset": key[0], "task": key[1], "split": key[2], "count": count}
                for key, count in sorted(counts.items())
            ],
        }
    if args.command == "doctor":
        import platform

        import torch

        return {
            "python": sys.version.split()[0],
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpus": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
