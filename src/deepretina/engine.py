"""Reproducible training and full-image evaluation of independent mask channels."""

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import CropDataset, read_manifest, read_sample
from .inference import binary_masks, load_checkpoint, resolve_device, tiled_probabilities
from .model import CHANNELS, UNet


@dataclass
class TrainConfig:
    task: str
    epochs: int = 50
    batch_size: int = 4
    patch_size: int = 256
    base_channels: int = 24
    depth: int = 3
    seed: int = 42
    lr: float = 0.001
    loss: str = "bce"
    device: str = "auto"
    workers: int = 0
    samples_per_image: int = 16
    max_steps: int | None = None
    tile_size: int = 512
    overlap: int = 64


def segmentation_loss(logits, target, valid, kind="bce_dice"):
    if kind not in {"bce", "bce_dice"}:
        raise ValueError("loss must be bce or bce_dice")
    weight = valid.to(logits.dtype)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
    ).sum() / weight.sum().clamp_min(1)
    if kind == "bce":
        return bce
    probability = logits.sigmoid()
    axes = (0, 2, 3)
    intersection = (probability * target * weight).sum(axes)
    denominator = ((probability + target) * weight).sum(axes)
    supported = weight.sum(axes) > 0
    dice_loss = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    return bce + (dice_loss[supported].mean() if supported.any() else logits.sum() * 0)


def _channel_metrics(tp, fp, fn, n):
    def ratio(a, b):
        return float(a / b) if b else None

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "valid_pixels": int(n),
    }


def score_model(
    model, manifest: Path, task: str, split: str, device, tile_size=512, overlap=64, threshold=0.5
):
    rows = read_manifest(manifest, task, split)
    supported = np.asarray(
        getattr(model, "supervised_channels", [True] * len(CHANNELS[task])), dtype=bool
    )
    counts = np.zeros((len(CHANNELS[task]), 4), np.int64)
    per_image = []
    dataset_counts = {}
    dataset_images = {}
    av_correct = av_conditional_correct = av_count = 0
    for row in rows:
        image, target_t, valid_t = read_sample(Path(manifest).parent, row)
        probability = tiled_probabilities(model, image, device, tile_size, overlap)
        predicted = binary_masks(probability, task, threshold)
        target, valid = target_t.numpy() >= 0.5, valid_t.numpy()
        valid[~supported] = False
        predicted[~supported] = False
        sample_metrics = {}
        dataset_name = row["dataset"]
        if dataset_name not in dataset_counts:
            dataset_counts[dataset_name] = np.zeros_like(counts)
            dataset_images[dataset_name] = 0
        dataset_images[dataset_name] += 1
        for c, name in enumerate(CHANNELS[task]):
            p, t, v = predicted[c], target[c], valid[c]
            values = np.array([(p & t & v).sum(), (p & ~t & v).sum(), (~p & t & v).sum(), v.sum()])
            counts[c] += values
            dataset_counts[dataset_name][c] += values
            sample_metrics[name] = _channel_metrics(*values)
        per_image.append({"id": row["id"], "dataset": row["dataset"], "channels": sample_metrics})
        if task == "av":
            labeled = valid[1] & valid[2] & (target[1] ^ target[2])
            # End-to-end AV accuracy includes missed vessel detections as errors.
            av_correct += int(
                ((predicted[1] & target[1] | predicted[2] & target[2]) & labeled).sum()
            )
            av_conditional_correct += int(
                ((probability[1] >= probability[2]) == target[1])[labeled].sum()
            )
            av_count += int(labeled.sum())
    channels = {
        name: _channel_metrics(*values) for name, values in zip(CHANNELS[task], counts, strict=True)
    }
    dices = [v["dice"] for v in channels.values() if v["dice"] is not None]
    by_dataset = {}
    for dataset_name, values in sorted(dataset_counts.items()):
        dataset_channels = {
            name: _channel_metrics(*value)
            for name, value in zip(CHANNELS[task], values, strict=True)
        }
        dataset_dices = [v["dice"] for v in dataset_channels.values() if v["dice"] is not None]
        by_dataset[dataset_name] = {
            "images": dataset_images[dataset_name],
            "channels": dataset_channels,
            "mean_dice": float(np.mean(dataset_dices)) if dataset_dices else None,
        }
    metric_support = {
        name: "reference_class_labeled_vessel_pixels"
        if task == "av" and name in {"artery", "vein"}
        else "annotated_valid_pixels_within_field_of_view"
        for name in CHANNELS[task]
    }
    return {
        "task": task,
        "split": split,
        "threshold": threshold,
        "images": len(rows),
        "channels": channels,
        "metric_support": metric_support,
        "supervised_channels": supported.tolist(),
        "aggregation": "pool pixel confusion counts per channel, then average defined channel Dice",
        "by_dataset": by_dataset,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "av_accuracy": av_correct / av_count if av_count else None,
        "av_conditional_accuracy": av_conditional_correct / av_count if av_count else None,
        "av_accuracy_description": (
            "End-to-end fraction correct on reference class-labeled vessel pixels; "
            "requires vessel and class thresholds plus probability-based class conflict resolution. "
            "Missed vessel detections count as errors; reference-background false positives "
            "are outside this accuracy support."
        )
        if task == "av"
        else None,
        "av_conditional_accuracy_description": (
            "Artery-versus-vein argmax accuracy on reference class-labeled vessel pixels, "
            "independent of predicted vessel membership and class thresholds. "
            "Equal class probabilities choose artery."
        )
        if task == "av"
        else None,
        "av_labeled_pixels": av_count,
        "per_image": per_image,
    }


def evaluate(
    checkpoint: Path,
    manifest: Path,
    split="test",
    device="auto",
    tile_size=512,
    overlap=64,
    threshold=0.5,
):
    model, saved, device = load_checkpoint(checkpoint, device)
    return score_model(
        model, manifest, saved["config"]["task"], split, device, tile_size, overlap, threshold
    )


def train(manifest: Path, output: Path, config: TrainConfig) -> Path:
    if (
        config.task not in CHANNELS
        or config.epochs < 1
        or config.batch_size < 1
        or config.lr <= 0
        or config.workers < 0
    ):
        raise ValueError("Invalid task, epochs, batch_size, learning rate, or worker count")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.loss not in {"bce", "bce_dice"}:
        raise ValueError("loss must be bce or bce_dice")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device(config.device)
    dataset = CropDataset(manifest, config.task, config.patch_size, config.samples_per_image)
    supervised_channels = np.zeros(len(CHANNELS[config.task]), dtype=bool)
    for row in dataset.rows:
        support = row.get("supervised_channels")
        if support is None:
            with np.load(Path(manifest).parent / row["mask"], allow_pickle=False) as labels:
                validity = labels["valid"].astype(bool)
                if "fov" in labels:
                    validity &= labels["fov"][None].astype(bool)
                support = validity.any(axis=(1, 2))
        if np.asarray(support).shape != supervised_channels.shape:
            raise ValueError(f"Invalid supervised_channels for {row['id']}")
        supervised_channels |= np.asarray(support, dtype=bool)
    if not supervised_channels.any():
        raise ValueError("Training split has no supervised channels")
    validation_rows = read_manifest(manifest, config.task, "val")
    preparation_sizes = {r.get("preparation_max_size") for r in dataset.rows + validation_rows}
    if len(preparation_sizes) != 1:
        raise ValueError(
            "Training and validation records use different preparation_max_size values"
        )
    inference_max_size = preparation_sizes.pop()
    if inference_max_size is not None and inference_max_size < 8:
        raise ValueError("preparation_max_size must be >=8 or null")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        generator=generator,
    )
    model = UNet(len(CHANNELS[config.task]), config.base_channels, config.depth).to(device)
    model.supervised_channels = supervised_channels.tolist()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "metrics.csv").exists():
        raise ValueError(f"Training output {output} already contains a run; choose a new directory")
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    metadata = {
        "config": asdict(config),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_hash,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "normalization": "RGB uint8 / 255, then (x - 0.5) / 0.5",
        "channels": CHANNELS[config.task],
        "supervised_channels": supervised_channels.tolist(),
        "initialization": "random; no pretraining",
        "inference_max_size": inference_max_size,
        "determinism": "seeded; unsupported deterministic operators warn",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best = -float("inf")
    steps = 0
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "steps", "train_loss", "val_mean_dice"],
            lineterminator="\n",
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = batches = 0
            for image, target, valid in loader:
                image, target, valid = image.to(device), target.to(device), valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model((image - 0.5) / 0.5), target, valid, config.loss)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
                steps += 1
                if config.max_steps is not None and steps >= config.max_steps:
                    break
            metrics = score_model(
                model, manifest, config.task, "val", device, config.tile_size, config.overlap
            )
            score = metrics["mean_dice"]
            if score is None:
                raise ValueError("Validation has no defined Dice score; inspect valid masks")
            checkpoint = {
                "format_version": 1,
                "config": asdict(config),
                "model": model.state_dict(),
                "epoch": epoch,
                "steps": steps,
                "val_mean_dice": score,
                "channels": CHANNELS[config.task],
                "supervised_channels": supervised_channels.tolist(),
                "normalization": metadata["normalization"],
                "manifest_sha256": manifest_hash,
                "inference_max_size": inference_max_size,
            }
            torch.save(checkpoint, output / "last.pt")
            if score > best:
                best = score
                torch.save(checkpoint, output / "best.pt")
                (output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            row = {
                "epoch": epoch,
                "steps": steps,
                "train_loss": total_loss / batches,
                "val_mean_dice": score,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if config.max_steps is not None and steps >= config.max_steps:
                break
    return output / "best.pt"
