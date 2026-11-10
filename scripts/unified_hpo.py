"""Plan and run bounded full-data unified trials, using validation only.

Each invocation runs one trial on one selected device. Scheduling across local
hosts/PBS remains explicit; this script never submits cluster jobs or job arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from deepretina.unified import UnifiedConfig, train_unified

REQUIRED_SOURCES = frozenset(
    {
        "drive",
        "rite",
        "hrf",
        "drhagis",
        "refuge",
        "drishti",
        "chasedb1",
        "iostar",
        "aria",
        "avrdb",
        "g1020",
        "onhsd",
        "riga",
        "rimone",
        "stare",
        "uoadr",
        "erlangen",
    }
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest(root):
    """Fingerprint execution sources, including files not yet tracked by Git."""
    result = hashlib.sha256()
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("scripts/*.py"),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in sorted(files):
        result.update(path.relative_to(root).as_posix().encode())
        result.update(b"\0")
        result.update(path.read_bytes())
    return result.hexdigest()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def make_plan(manifest, *, screen_epochs=8, total_epochs=60, epoch_samples=8192):
    manifest = Path(manifest).resolve()
    if epoch_samples < 1:
        raise ValueError("epoch_samples must be positive")
    if not 1 <= screen_epochs <= total_epochs:
        raise ValueError("Screen epochs must be in1..total_epochs")
    root = Path(__file__).resolve().parents[1]
    manifest_hash = digest(manifest)
    if (manifest.parent / "_SUCCESS").read_text().strip() != manifest_hash:
        raise ValueError("Completed preparation marker differs from manifest")
    audit = json.loads((manifest.parent / "integrity.json").read_text())
    if audit["manifest_sha256"] != manifest_hash or audit["preview_only"]:
        raise ValueError("Require audited full preparation, not a preview")
    rows = json.loads(manifest.read_text())
    if any(r.get("preparation_preview", False) for r in rows):
        raise ValueError("Limited source preparation cannot be used for full-data search")
    train_sources = {r["dataset"] for r in rows if r["split"] == "train"}
    missing = REQUIRED_SOURCES - train_sources
    if missing:
        raise ValueError(f"Full17-source protocol is missing training sources: {sorted(missing)}")
    if audit["annotations_checked"] != len(rows) or audit["photos_checked"] != len(
        {r["photo_id"] for r in rows}
    ):
        raise ValueError("Integrity inventory does not cover the whole manifest")
    for split in ("train", "val"):
        for channel in ("vessel", "artery", "vein", "disc", "cup"):
            support = [
                h for h in audit["head_counts"] if h["split"] == split and h["channel"] == channel
            ]
            if not sum(h["valid_pixels"] for h in support) or not sum(
                h["positive_photos"] for h in support
            ):
                raise ValueError(
                    f"Full-data search lacks positive/valid {split} supervision for {channel}"
                )
    train_photos = {r["photo_id"] for r in rows if r["split"] == "train"}
    if not train_photos:
        raise ValueError("No training photographs")
    base = UnifiedConfig(
        selection_metric="macro_dataset_mean_dice",
        purpose="full_data",
        epochs=total_epochs,
        epoch_samples=max(epoch_samples, len(train_photos)),
    )
    variants = {
        "baseline": base,
        "lr_0003": replace(base, lr=0.0003),
        "lr_003": replace(base, lr=0.003),
        "augmentation_strong": replace(base, augmentation="strong"),
        "width_8": replace(base, base_channels=8),
        "width_16": replace(base, base_channels=16),
        "legacy_topology_scratch": replace(base, architecture="legacy_universal", base_channels=32),
    }
    for config in variants.values():
        config.validate()
    source = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": 1,
        "purpose": "full_data_validation_search",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "prepared_content_sha256": audit["prepared_content_sha256"],
        "source_commit": source,
        "execution_code_sha256": code_digest(root),
        "training_photos": len(train_photos),
        "datasets": sorted({r["dataset"] for r in rows if r["split"] == "train"}),
        "screen_epochs": screen_epochs,
        "total_epochs": total_epochs,
        "candidates": {name: asdict(config) for name, config in variants.items()},
        "selection_policy": "Compare validation learning curves and per-head/per-dataset scores; resume shortlisted trials with unchanged scheduler/config; freeze final selection before test",
        "limitations": [
            "Single seed screening cannot establish robustness",
            "Early rankings may differ after full training",
            "No model quality improvement is established by a plan or screen",
            "Historical original-model exposure requires qualified comparison",
        ],
    }


def run_trial(
    plan_path,
    trial,
    output,
    *,
    manifest=None,
    device=None,
    resume=False,
    final=False,
    stop_after_epoch=None,
):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema_version") != 1 or plan.get("purpose") != "full_data_validation_search":
        raise ValueError("Expected a full-data validation search plan")
    root = Path(__file__).resolve().parents[1]
    if code_digest(root) != plan["execution_code_sha256"]:
        raise ValueError("Execution code changed; review and create a new search plan")
    manifest = Path(manifest or plan["manifest"])
    if digest(manifest) != plan["manifest_sha256"]:
        raise ValueError("Trial manifest differs from frozen plan")
    config = UnifiedConfig(**plan["candidates"][trial])
    if config.legacy_initialization is not None:
        raise ValueError("Independent full-data search excludes historical teacher initialization")
    if device is not None:
        config = replace(config, device=device)
    config.validate()
    until = (
        stop_after_epoch
        if stop_after_epoch is not None
        else (plan["total_epochs"] if final else plan["screen_epochs"])
    )
    if not 1 <= until <= config.epochs:
        raise ValueError("stop_after_epoch must fit the unchanged planned scheduler horizon")
    output = Path(output)
    provenance = {
        "plan_sha256": digest(plan_path),
        "trial": trial,
        "execution_code_sha256": plan["execution_code_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
    }
    provenance_path = output.parent / (output.name + ".search_trial.json")
    if resume:
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Resume search provenance differs")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Choose a new trial output directory")
        write_new(provenance_path, provenance)
    best = train_unified(
        manifest,
        output,
        config,
        resume=output / "last.pt" if resume else None,
        stop_after_epoch=until,
    )
    return {
        "trial": trial,
        "best": str(best),
        "stop_after_epoch": until,
        "test_evaluation_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--screen-epochs", type=int, default=8)
    plan.add_argument("--total-epochs", type=int, default=60)
    plan.add_argument("--epoch-samples", type=int, default=8192)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--trial", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--manifest", type=Path)
    run.add_argument("--device")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--final", action="store_true")
    run.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = make_plan(
                args.manifest,
                screen_epochs=args.screen_epochs,
                total_epochs=args.total_epochs,
                epoch_samples=args.epoch_samples,
            )
            write_new(args.output, result)
            print(json.dumps({"candidates": list(result["candidates"]), "plan": str(args.output)}))
        else:
            print(
                json.dumps(
                    run_trial(
                        args.plan,
                        args.trial,
                        args.output,
                        manifest=args.manifest,
                        device=args.device,
                        resume=args.resume,
                        final=args.final,
                        stop_after_epoch=args.stop_after_epoch,
                    )
                )
            )
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"unified_hpo: {error}\n")


if __name__ == "__main__":
    main()
