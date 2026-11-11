"""Fresh native legacy architectures with a new, explicit canonical-data recipe.

No original tensor transfer or historical optimizer/preprocessing reproduction.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .geometry_policy import capture_geometry_policy
from .integrity import verify_prepared_content
from .legacy.av import LegacyAVNet, checkpoint_payload
from .legacy.single_net import Unet, tiled_logits
from .unified import read_unified_manifest, read_unified_sample

RECIPE = "native_legacy_canonical_fresh_v1"
HEADS = {"vessel": (0,), "disc": (3,), "universal": (0, 3, 4), "av": (1, 2)}
NAMES = {"vessel": ["vessel"], "disc": ["disc"], "universal": ["vessel", "disc", "cup"]}
PATCH = {"vessel": 320, "disc": 192, "universal": 288}
AV_ARCHITECTURE = {
    "encoder_channels": [4, 32, 64, 128, 256],
    "decoder_channels": [256, 128, 64, 32],
    "head_channels": 128,
    "num_classes": 3,
}


@dataclass
class LegacyTrainingConfig:
    task: str = "vessel"
    epochs: int = 60
    epoch_samples: int | None = None
    batch_size: int = 1
    patch_size: int | None = None
    lr: float = 0.0001
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    architecture: dict | None = None

    def validate(self):
        if (
            self.task not in HEADS
            or type(self.epochs) is not int
            or self.epochs < 1
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.seed) is not int
        ):
            raise ValueError("Require native task, positive epochs and batch size")
        if (
            not math.isfinite(self.lr)
            or self.lr <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer parameters")
        if self.epoch_samples is not None and (
            type(self.epoch_samples) is not int or self.epoch_samples < 1
        ):
            raise ValueError("epoch_samples must be a positive integer or omitted")
        if self.task == "av":
            if self.batch_size != 1 or self.patch_size is not None:
                raise ValueError("Native AV requires batch1/full frame, no patch/padding/letterbox")
        elif self.patch_size is not None and (self.patch_size < 16 or self.patch_size % 16):
            raise ValueError("Native single_net patches must be divisible by16")
        spec = architecture_for(self)
        if self.task == "av":
            if spec["encoder_channels"][0] != 4 or spec["num_classes"] != 3:
                raise ValueError("Native AV is four inputs and three logits")
        elif spec["in_size"] != 3 or spec["out_size"] != len(HEADS[self.task]):
            raise ValueError("Native single_net head/input inventory differs")


def architecture_for(config):
    if config.architecture is not None:
        return copy.deepcopy(config.architecture)
    if config.task == "av":
        return copy.deepcopy(AV_ARCHITECTURE)
    return {"filters": 32, "in_size": 3, "out_size": len(HEADS[config.task]), "do": 0.0, "depth": 4}


def build_fresh(config):
    config.validate()
    # Constructors initialize directly. No checkpoint is loaded or reset.
    spec = architecture_for(config)
    return LegacyAVNet(**spec) if config.task == "av" else Unet(**spec)


def av_input(image, target, valid, *, record_id):
    vessel = ((target[0] >= 0.5) & valid[0]).to(image.dtype)
    if not valid[0].any():
        raise ValueError(f"AV row {record_id} requires a binary same-observer vessel target")
    inputs = torch.cat([image - 0.5, vessel[None]], dim=0)
    if (
        min(inputs.shape[-2:]) < 16
        or not torch.isfinite(inputs).all()
        or (inputs.amax(dim=(1, 2)) == inputs.amin(dim=(1, 2))).any()
    ):
        raise ValueError(
            f"AV row {record_id}: constant channel/nonfinite or smaller than16; original normalization is undefined"
        )
    return inputs


def av_known(target, valid):
    """Known class pixels inside valid positive vessels, CHW or batched BCHW."""
    return (
        valid.select(-3, 0)
        & (target.select(-3, 0) >= 0.5)
        & valid.select(-3, 1)
        & valid.select(-3, 2)
        & ((target.select(-3, 1) >= 0.5) ^ (target.select(-3, 2) >= 0.5))
    )


class NativeDataset:
    """Select eligible observers/heads before photo sampling or crop construction."""

    def __init__(self, manifest, split, config):
        if split not in {"train", "val"}:
            raise ValueError("Native training only opens TRAIN/VAL annotations")
        self.root = Path(manifest).parent
        self.config = config
        rows = read_unified_manifest(Path(manifest), split)
        self.groups = defaultdict(lambda: defaultdict(list))
        self.excluded = []
        for row in rows:
            details = row.get("metadata", {})
            if (
                row.get("preparation_preview", details.get("preparation_preview", False))
                or row.get("preparation_max_size") is not None
                or row.get("preprocessing_profile", details.get("preprocessing_profile"))
                != "legacy25_raw"
                or row.get("pixels_per_degree", details.get("pixels_per_degree")) != 25
            ):
                raise ValueError(
                    "Native recipe requires full canonical legacy25_raw25px/degree metadata"
                )
            task = row["task"]
            allowed = (
                task == "av"
                if config.task == "av"
                else task == "disc_cup"
                if config.task == "disc"
                else task in {"av", "vessels"}
                if config.task == "vessel"
                else True
            )
            if not allowed:
                continue
            image, target, valid = read_unified_sample(self.root, row)
            eligible = (
                bool(av_known(target, valid).any())
                if config.task == "av"
                else bool(valid[list(HEADS[config.task])].any())
            )
            if not eligible:
                self.excluded.append(row["id"])
                continue
            if config.task == "av":
                av_input(
                    image, target, valid, record_id=row["id"]
                )  # reject before output/device setup
            group = "disc_cup" if task == "disc_cup" else "vascular"
            self.groups[row["photo_id"]][group].append(row)
        self.groups = {
            pid: {task: sorted(values, key=lambda r: r["id"]) for task, values in groups.items()}
            for pid, groups in sorted(self.groups.items())
        }
        if not self.groups:
            raise ValueError(f"No eligible {config.task} {split} observer bundles")
        self.photo_ids = list(self.groups)
        self.record_ids = {
            r["id"] for g in self.groups.values() for rows in g.values() for r in rows
        }
        self.visits = Counter()
        self.required = {pid: max(map(len, groups.values())) for pid, groups in self.groups.items()}

    def schedule(self, generator, samples=None):
        # Include enough visits to cycle every eligible observer bundle before extras.
        first = [pid for pid in self.photo_ids for _ in range(self.required[pid])]
        if samples is None:
            samples = len(first)
        if samples < len(first):
            raise ValueError("epoch_samples cannot omit coverage-first photo/observer pass")
        order = torch.randperm(len(first), generator=generator).tolist()
        result = [first[i] for i in order]
        for _ in range(samples - len(first)):
            result.append(
                self.photo_ids[int(torch.randint(len(self.photo_ids), (1,), generator=generator))]
            )
        return result

    def sample(self, pid, *, crop=False, generator=None, observer_index=None):
        index = self.visits[pid] if observer_index is None else observer_index
        if observer_index is None:
            self.visits[pid] += 1
        chosen = [rows[index % len(rows)] for rows in self.groups[pid].values()]
        images, targets, supports = [], [], []
        for row in chosen:
            image, target, valid = read_unified_sample(self.root, row)
            images.append(image)
            targets.append(target)
            supports.append(valid)
        if any(not torch.equal(images[0], image) for image in images[1:]):
            raise ValueError("Observers sharing a photo have different canonical RGB bytes")
        image = images[0]
        target = torch.zeros_like(targets[0])
        valid = torch.zeros_like(supports[0])
        for t, v in zip(targets, supports, strict=True):
            target = torch.where(v, t, target)
            valid |= v
        if self.config.task == "av":
            return (
                av_input(image, target, valid, record_id=chosen[0]["id"]),
                target,
                valid,
                [r["id"] for r in chosen],
            )
        image = image - 0.5
        if crop:
            size = self.config.patch_size or PATCH[self.config.task]
            h, w = image.shape[-2:]
            pad = (0, max(0, size - w), 0, max(0, size - h))
            mode = "reflect" if pad[1] < w and pad[3] < h else "replicate"
            image = F.pad(image, pad, mode=mode)
            target = F.pad(target, pad)
            valid = F.pad(valid, pad)
            supports = [F.pad(v, pad) for v in supports]
            h, w = image.shape[-2:]
            locations = torch.nonzero(valid[list(HEADS[self.config.task])].any(0))
            point = locations[int(torch.randint(len(locations), (1,), generator=generator))]
            y = max(0, min(h - size, int(point[0]) - size // 2))
            x = max(0, min(w - size, int(point[1]) - size // 2))
            image = image[:, y : y + size, x : x + size]
            target = target[:, y : y + size, x : x + size]
            valid = valid[:, y : y + size, x : x + size]
            supports = [v[:, y : y + size, x : x + size] for v in supports]
        contributed = [
            row["id"]
            for row, support in zip(chosen, supports, strict=True)
            if support[list(HEADS[self.config.task])].any()
        ]
        return image, target, valid, contributed


def native_loss(logits, target, valid, task):
    if task == "av":
        known = av_known(target, valid)
        if not known.any():
            return logits.sum() * 0
        classes = (target[:, 2] >= 0.5).long()
        return F.cross_entropy(logits[:, 1:3].permute(0, 2, 3, 1)[known], classes[known])
    values = []
    for output, head in enumerate(HEADS[task]):
        support = valid[:, head]
        if not support.any():
            continue
        z, t = logits[:, output][support], target[:, head][support]
        probability = z.sigmoid()
        values.append(
            F.binary_cross_entropy_with_logits(z, t)
            + 1
            - (2 * (probability * t).sum() + 1e-6) / ((probability + t).sum() + 1e-6)
        )
    return torch.stack(values).mean() if values else logits.sum() * 0


def evaluate(model, dataset, config, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for pid, groups in dataset.groups.items():
            for index in range(max(map(len, groups.values()))):
                image, target, valid, _ = dataset.sample(pid, observer_index=index)
                if config.task == "av":
                    logits = model(image[None].to(device))
                else:
                    logits = (
                        tiled_logits(
                            model, image, device, crop_size=config.patch_size or PATCH[config.task]
                        )[None]
                        .float()
                        .to(device)
                    )
                losses.append(
                    float(
                        native_loss(
                            logits, target[None].to(device), valid[None].to(device), config.task
                        )
                    )
                )
    if not losses or not all(math.isfinite(v) for v in losses):
        raise ValueError("Missing/nonfinite VAL loss")
    return sum(losses) / len(losses)


def source_sha256():
    root = Path(__file__).parent
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        result.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return result.hexdigest()


def require_finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite native model/optimizer state")
    elif isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite native optimizer scalar")


def _save_checkpoint(path, payload):
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(manifest, output, config=None, *, expected_content_sha256=None):
    config = config or LegacyTrainingConfig()
    config.validate()
    manifest, output = Path(manifest), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Fresh native training requires a new output directory")
    if output.resolve().is_relative_to(manifest.parent.resolve()):
        raise ValueError("Write native runs outside immutable prepared inputs")
    raw = manifest.read_bytes()
    rows = json.loads(raw)
    geometry = capture_geometry_policy(manifest.parent, rows)
    content = verify_prepared_content(manifest, expected_content_sha256)
    audit = json.loads((manifest.parent / "integrity.json").read_text())
    if audit.get("preview_only") is not False:
        raise ValueError("Native training rejects preparation previews")
    train_data = NativeDataset(manifest, "train", config)
    val_data = NativeDataset(manifest, "val", config)
    generator = torch.Generator().manual_seed(config.seed)
    train_data.schedule(generator, config.epoch_samples)  # fail undersized budget before outputs
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = build_fresh(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    architecture = architecture_for(config)
    source = source_sha256()
    recipe = {
        "name": RECIPE,
        "historical_optimizer_reproduction": False,
        "fresh_initialization": True,
        "config": asdict(config),
        "architecture_matches_original_default": config.architecture is None,
        "selection_metric": "minimum observer-cycled masked VAL loss",
        "av_gt_vessel_training_input": config.task == "av",
    }
    preprocessing = {
        "profile": "legacy25_raw_av4" if config.task == "av" else "single_net",
        "normalization": "internal per-image/channel torch.mean and sample torch.std(correction1); no epsilon"
        if config.task == "av"
        else "RGB/255-0.5",
        "pixels_per_degree": 25,
        "normalized": False,
        "expects_preprocessed_input": True,
        "input_values": "canonical RGB/255-0.5 plus same-observer GT binary vessel"
        if config.task == "av"
        else "canonical RGB/255-0.5",
        "training_recipe": RECIPE,
        "historical_preprocessing_reproduction": False,
    }
    lineage = {
        "initialization": "fresh_native_constructor",
        "loaded_checkpoint": None,
        "historical_tensor_transfer": False,
    }
    identity = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_content_sha256": content,
        "training_source_sha256": source,
        "training_recipe": recipe,
        "lineage": lineage,
        "seed": config.seed,
    }
    if geometry is not None:
        identity["geometry_policy"] = geometry
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "run.json",
        {
            **identity,
            "architecture": architecture,
            "preprocessing": preprocessing,
            "eligibility": {
                split: {
                    "eligible_photos": len(dataset.photo_ids),
                    "eligible_annotation_records": len(dataset.record_ids),
                    "excluded_no_selected_support": dataset.excluded,
                }
                for split, dataset in (("train", train_data), ("val", val_data))
            },
        },
    )
    photo_visits, annotation_visits = Counter(), Counter()
    best = math.inf
    steps = 0
    metrics = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        schedule = train_data.schedule(generator, config.epoch_samples)
        loss_sum = 0
        for start in range(0, len(schedule), config.batch_size):
            pids = schedule[start : start + config.batch_size]
            samples = [
                train_data.sample(pid, crop=config.task != "av", generator=generator)
                for pid in pids
            ]
            images = torch.stack([s[0] for s in samples]).to(device)
            targets = torch.stack([s[1] for s in samples]).to(device)
            valid = torch.stack([s[2] for s in samples]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = native_loss(logits, targets, valid, config.task)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native training loss")
            loss.backward()
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(samples)
            for pid, sample in zip(pids, samples, strict=True):
                photo_visits[pid] += 1
                annotation_visits.update(sample[3])
        require_finite(model.state_dict())
        require_finite(optimizer.state_dict())
        val_loss = evaluate(model, val_data, config, device)
        coverage = {
            "eligible_photos": len(train_data.photo_ids),
            "seen_photos": len(photo_visits),
            "eligible_annotation_records": len(train_data.record_ids),
            "contributing_annotation_records": len(annotation_visits),
            "photo_visits": dict(photo_visits),
            "annotation_visits": dict(annotation_visits),
            "counting": "actual selected-head support delivered to successful optimizer steps",
        }
        coverage["all_eligible_annotations_contributed"] = (
            set(annotation_visits) == train_data.record_ids
        )
        coverage["all_eligible_photos_seen"] = set(photo_visits) == set(train_data.photo_ids)
        if source_sha256() != source or manifest.read_bytes() != raw:
            raise ValueError("Source/manifest changed during native run")
        payload = (
            checkpoint_payload(
                model,
                {"training_recipe": RECIPE},
                preprocessing=preprocessing,
                geometry_policy=geometry,
            )
            if config.task == "av"
            else {
                "format_version": 1,
                "family": "legacy_single_net",
                "architecture": architecture,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "channels": NAMES[config.task],
                "preprocessing": preprocessing,
                "tiling": {
                    "crop_size": config.patch_size or PATCH[config.task],
                    "border": (config.patch_size or PATCH[config.task]) // 8,
                    "blend": "logits",
                    "window": "2*boxcar_fraction_y*boxcar_fraction_x-1, clipped to0.01",
                },
                "provenance": {"training_recipe": RECIPE},
            }
        )
        payload.update(
            identity,
            epoch=epoch,
            steps=steps,
            coverage=coverage,
            validation={"split": "val", "loss": val_loss},
            training_state={
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "sampler_rng": generator.get_state(),
            },
        )
        _save_checkpoint(output / "last.pt", payload)
        if val_loss < best:
            best = val_loss
            _save_checkpoint(output / "best.pt", payload)
        metrics.append(
            {
                "epoch": epoch,
                "steps": steps,
                "train_loss": loss_sum / len(schedule),
                "val_loss": val_loss,
            }
        )
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        _write_json(output / "coverage.json", coverage)
        _write_json(
            output / "training_summary.json",
            {
                "completed_epochs": epoch,
                "planned_epochs": config.epochs,
                "completed_budget": epoch == config.epochs,
                "steps": steps,
                "best_validation_loss": best,
                "recipe": RECIPE,
                "test_evaluation_performed": False,
            },
        )
    return output / "best.pt"


def add_cli_args(parser):
    """Compatibility wrapper; parser-only callers should import legacy_training_cli."""
    from .legacy_training_cli import add_cli_args as add

    return add(parser)


def train_from_args(args):
    from .legacy_training_cli import train_from_args as dispatch

    return dispatch(args)


if __name__ == "__main__":
    from .legacy_training_cli import main

    main()
