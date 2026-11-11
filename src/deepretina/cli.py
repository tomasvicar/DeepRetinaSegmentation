"""Command line entry points. Run ``uv run deepretina --help``."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict, fields
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    from .bundled import model_catalog

    registry = model_catalog()
    model_names = sorted(set(registry["models"]) | set(registry.get("aliases", {})))
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
        help="Dataset selectors; run deepretina datasets for the complete list",
    )
    prepare.add_argument("--max-size", type=int, default=1024)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--limit",
        type=int,
        help="Preview at most N source groups per dataset; not a benchmark dataset",
    )
    prepare.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override a dataset archive/directory, e.g. ubmi=/path/to/images",
    )
    catalog = commands.add_parser(
        "datasets", help="List all migrated dataset selectors and source layouts"
    )
    catalog.add_argument(
        "--json", action="store_true", help="Full machine-readable coverage inventory"
    )

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

    native = commands.add_parser(
        "train-legacy", help="Retrain the exact native original architectures from scratch"
    )
    from .legacy_training_cli import add_cli_args

    add_cli_args(native)

    joint = commands.add_parser(
        "train-unified", help="Experimental five-output training with partial supervision"
    )
    joint.add_argument("--manifest", type=Path, required=True)
    joint.add_argument("--output", type=Path, required=True)
    joint.add_argument("--config", type=Path, help="Flat JSON or TOML UnifiedConfig settings")
    joint.add_argument("--resume", type=Path, help="Resume a unified last.pt checkpoint")
    joint.add_argument("--device")
    joint.add_argument(
        "--mixed-precision",
        choices=["none", "bf16"],
        help="Override unified training precision; CPU/Pascal require none",
    )
    joint.add_argument("--stop-after-epoch", type=int)
    joint.add_argument("--expected-content-sha256", help="Optional externally pinned corpus hash")
    joint.add_argument(
        "--derived-cache", type=Path, help="Explicit verified train/val cache directory"
    )
    joint.add_argument(
        "--expected-cache-fingerprint", help="Externally pinned derived cache identity"
    )
    joint.add_argument(
        "--photometry-cache", type=Path, help="Verified profile-specific full-image RGB cache"
    )
    joint.add_argument("--expected-photometry-cache-fingerprint")
    joint.add_argument("--purpose", choices=["functionality", "full_data"])
    for name in ("epochs", "epoch_samples", "max_steps"):
        joint.add_argument("--" + name.replace("_", "-"), type=int)

    evaluate = commands.add_parser("evaluate", help="Score a held-out split with tiled inference")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate.add_argument("--output", type=Path, required=True, help="Metrics JSON")

    predict = commands.add_parser(
        "predict", help="Predict an image or directory at original resolution"
    )
    predict.add_argument("--input", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    for cmd in (evaluate, predict):
        model_source = cmd.add_mutually_exclusive_group(required=True)
        model_source.add_argument("--checkpoint", type=Path, help="A custom trained checkpoint")
        model_source.add_argument(
            "--model",
            choices=model_names,
            help="Use a model bundled with this package",
        )
        cmd.add_argument("--device", default="auto")
        cmd.add_argument(
            "--tile-size", type=int, help="Override tile size; unified uses checkpoint default"
        )
        cmd.add_argument(
            "--overlap", type=int, help="Override overlap; unified uses checkpoint default"
        )
        cmd.add_argument(
            "--tta",
            choices=["none", "flips2", "rot4", "dihedral8"],
            default="none",
            help="Optional unified full-image inference ensemble (default: none)",
        )
        cmd.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--dataset", help="Camera/dataset profile, e.g. drive, hrf, drishti")
    predict.add_argument(
        "--fov", type=Path, help="Original-frame FOV mask (official mask preferred)"
    )
    predict.add_argument("--field-of-view-degrees", type=float)
    predict.add_argument("--pixels-per-degree", type=float, default=25)
    predict.add_argument(
        "--normalized",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override single_net contrast normalization; otherwise use the saved recipe or historical raw default",
    )
    predict.add_argument(
        "--prepared", action="store_true", help="A/V RGB already has historical preprocessing"
    )
    predict.add_argument(
        "--prepared-array", type=Path, help="Four stored A/V planes in NPY/NPZ (key input)"
    )
    predict.add_argument("--dicoms", type=Path, nargs=4, metavar=("R", "G", "B", "VESSEL"))
    predict.add_argument("--vessel-mask", type=Path, help="Binary vessel mask for original A/V")
    predict.add_argument("--original-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    raw_av_mode = predict.add_mutually_exclusive_group()
    raw_av_mode.add_argument(
        "--av-raw-proxy",
        action="store_true",
        help="Explicit approximate Python preprocessing for raw A/V photographs",
    )
    raw_av_mode.add_argument(
        "--av-raw-matlab",
        action="store_true",
        help="MATLAB-style AV profile with provided --fov; optional explicit diameter overrides calibration",
    )
    raw_av_mode.add_argument(
        "--av-raw-matlab-source",
        action="store_true",
        help="Opt-in Python source FOV/photometry profile; optional --fov, numerical parity not certified",
    )
    predict.add_argument(
        "--vessel-checkpoint", type=Path, help="Override automatic legacy-vessels model for raw A/V"
    )
    predict.add_argument("--vessel-normalized", action="store_true")
    predict.add_argument(
        "--fov-diameter-pixels",
        type=float,
        help="Explicit raw A/V FOV diameter; overrides generic provided-mask calibration",
    )
    inspect = commands.add_parser("inspect", help="Summarize a prepared manifest")
    inspect.add_argument("manifest", type=Path)
    commands.add_parser("doctor", help="Check Python, PyTorch and CUDA")
    commands.add_parser("models", help="List bundled models, quality metrics and checksums")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, f"deepretina: {exc}\n")
    if result is not None:
        print(result if isinstance(result, str) else json.dumps(result, indent=2, allow_nan=False))


def dispatch(args: argparse.Namespace):
    if args.command == "datasets":
        import importlib.resources

        inventory = importlib.resources.files("deepretina.adapters").joinpath("coverage.json")
        details = json.loads(inventory.read_text())
        if args.json:
            return details
        lines = [f"{'SELECTOR':<13} {'TASKS':<23} SOURCE"]
        for entry in details["entries"]:
            tasks = ",".join(entry["supported_tasks"])
            lines.append(
                f"{entry['selector']:<13} {tasks:<23} {', '.join(entry['required_resources'])}"
            )
        return "\n".join(lines)
    if args.command == "prepare":
        from .prepare import prepare

        overrides = {}
        for value in args.source:
            dataset, separator, path = value.partition("=")
            if not separator or not path:
                raise ValueError("--source must be DATASET=PATH")
            if dataset in overrides:
                raise ValueError(f"Duplicate source override for {dataset}")
            overrides[dataset] = Path(path)
        manifest = prepare(
            args.data_root,
            args.output,
            args.datasets,
            args.max_size,
            args.seed,
            limit=args.limit,
            source_overrides=overrides,
        )
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
    if args.command == "train-unified":
        from .unified_cli import train_from_args

        return train_from_args(args)
    if args.command == "train-legacy":
        from .legacy_training_cli import train_from_args

        return train_from_args(args)
    if args.command == "models":
        from .bundled import model_catalog

        return model_catalog()
    if args.command in ("evaluate", "predict"):
        if args.model:
            from .bundled import bundled_checkpoint

            with bundled_checkpoint(args.model) as checkpoint:
                return _apply_model(args, checkpoint)
        return _apply_model(args, args.checkpoint)
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


def _apply_model(args, checkpoint: Path):
    if args.command == "predict":
        from .prediction_output import check_prediction_output

        check_prediction_output(args.output)
    import torch

    family = torch.load(checkpoint, map_location="cpu", weights_only=True).get("family")
    if family == "unified":
        from .unified_cli import apply_unified

        return apply_unified(args, checkpoint)
    if getattr(args, "tta", "none") != "none":
        raise ValueError("--tta requires a unified checkpoint")
    if family in {"legacy_single_net", "legacy_av"}:
        return _apply_legacy(args, checkpoint, family)
    if args.command == "predict" and (
        args.av_raw_proxy or args.av_raw_matlab or args.av_raw_matlab_source
    ):
        raise ValueError("Raw A/V profiles require the original A/V model")
    if args.command == "predict" and args.input is None:
        raise ValueError("--input is required for pilot model prediction")
    from .engine import evaluate
    from .inference import predict

    options = {
        "device": args.device,
        "tile_size": args.tile_size if args.tile_size is not None else 512,
        "overlap": args.overlap if args.overlap is not None else 64,
        "threshold": args.threshold,
    }
    if args.command == "evaluate":
        result = evaluate(checkpoint, args.manifest, split=args.split, **options)
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
        return [predict(checkpoint, path, args.output / path.name, **options) for path in images]
    return predict(checkpoint, args.input, args.output, **options)


def _apply_legacy(args, checkpoint: Path, family: str):
    if args.command != "predict":
        raise ValueError(
            "Original models require raw-input evaluation with their own geometry; the capped pilot manifest is not interchangeable"
        )
    from .legacy.inference import predict_legacy

    if args.input is None and args.dicoms is None and args.prepared_array is None:
        raise ValueError("Supply --input, --dicoms, or --prepared-array")
    if args.prepared_array and (
        args.dicoms
        or args.prepared
        or args.av_raw_proxy
        or args.av_raw_matlab
        or args.av_raw_matlab_source
    ):
        raise ValueError("Choose a single prepared A/V input mode")

    def one(image_path, output):
        if args.av_raw_proxy or args.av_raw_matlab or args.av_raw_matlab_source:
            if family != "legacy_av" or image_path is None:
                raise ValueError(
                    "Raw A/V profiles require the original A/V model and a raw RGB input"
                )
            if args.prepared or args.dicoms or args.prepared_array:
                raise ValueError("Raw A/V profiles cannot be combined with prepared input modes")
            import torch

            profile = (
                torch.load(checkpoint, map_location="cpu", weights_only=True)
                .get("preprocessing", {})
                .get("profile")
            )
            if profile == "legacy25_raw_av4":
                raise ValueError(
                    "This retrained AV model uses canonical raw RGB plus --vessel-mask; "
                    "historical --av-raw-* preprocessing does not match its saved recipe"
                )
            from .legacy.raw_proxy import predict_raw_av

            return predict_raw_av(
                checkpoint,
                image_path,
                output,
                device=args.device,
                dataset=args.dataset,
                fov_path=args.fov,
                field_of_view_degrees=args.field_of_view_degrees,
                vessel_mask_path=args.vessel_mask,
                vessel_checkpoint=args.vessel_checkpoint,
                vessel_normalized=args.vessel_normalized,
                fov_diameter_pixels=args.fov_diameter_pixels,
                pixels_per_degree=args.pixels_per_degree,
                threshold=args.threshold,
                preprocessing_profile="matlab_source"
                if args.av_raw_matlab_source
                else ("matlab" if args.av_raw_matlab else "proxy"),
            )
        inputs = None
        if args.prepared_array:
            import numpy as np

            loaded = np.load(args.prepared_array, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                with loaded:
                    inputs = loaded["input"].copy()
            else:
                inputs = loaded
        return predict_legacy(
            checkpoint,
            image_path,
            output,
            device=args.device,
            dataset=args.dataset,
            fov_path=args.fov,
            field_of_view_degrees=args.field_of_view_degrees,
            normalized=args.normalized,
            threshold=args.threshold,
            vessel_mask_path=args.vessel_mask,
            dicom_paths=args.dicoms,
            prepared=args.prepared,
            prepared_inputs=inputs,
            original_size=tuple(args.original_size) if args.original_size else None,
            pixels_per_degree=args.pixels_per_degree,
        )

    if args.input is not None and args.input.is_dir():
        if args.dicoms or args.prepared_array or args.vessel_mask or args.fov:
            raise ValueError(
                "Batch mode uses automatic masks; supplied per-image masks/planes require individual calls"
            )
        images = sorted(
            p
            for p in args.input.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
        )
        if not images:
            raise ValueError(f"No supported images in {args.input}")
        return [one(image, args.output / image.name) for image in images]
    return one(args.input, args.output)
