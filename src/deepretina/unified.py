"""One RGB network with vessel, conditional A/V, disc and cup supervision.

Missing annotations stay invalid. Conditional A/V uses a two-class softmax only
on known reference vessel pixels; prediction chooses a class for each detected
vessel pixel. No independent artery/vein sigmoid thresholds are introduced.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .dataset import letterbox_geometry, whole_view
from .derived_cache import VerifiedCache
from .inference import resolve_device, tile_starts
from .integrity import verify_prepared_content
from .model import UNet

CHANNELS = ("vessel", "artery", "vein", "disc", "cup")
SEGMENTATION_CHANNELS = (0, 3, 4)
SOURCE_CHANNELS = {"vessels": (0,), "av": (0, 1, 2), "disc_cup": (3, 4)}


@dataclass
class UnifiedConfig:
    architecture: str = "compact_unet"
    legacy_initialization: str | None = None
    allow_historical_exposure: bool = False
    preprocessing_profile: str = "legacy25_raw"
    pixels_per_degree: float = 25.0
    epochs: int = 60
    epoch_samples: int = 8192
    coverage_first: bool = True
    batch_size: int = 4
    patch_size: int = 384
    base_channels: int = 12
    depth: int = 4
    norm: str = "batch"
    lr: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.05
    seed: int = 42
    workers: int = 2
    device: str = "auto"
    sampling: str = "dataset_task_balanced"
    train_view: str = "patch"
    view_size: int = 512
    foreground_probability: float = 0.5
    augmentation: str = "mild"
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    av_weight: float = 1.0
    tile_size: int = 512
    overlap: int = 64
    require_all_heads: bool = True
    selection_metric: str = "mean_dice"
    purpose: str = "functionality"
    max_steps: int | None = None

    def validate(self):
        if self.architecture not in {"compact_unet", "legacy_universal"}:
            raise ValueError("Unsupported unified architecture")
        if self.legacy_initialization is not None:
            if self.architecture != "legacy_universal":
                raise ValueError("Legacy initialization requires legacy_universal architecture")
            if not self.allow_historical_exposure:
                raise ValueError(
                    "Legacy initialization requires explicit allow_historical_exposure=true; teacher exposure is unknown"
                )
        if self.preprocessing_profile != "legacy25_raw" or self.pixels_per_degree != 25:
            raise ValueError(
                "Unified training uses canonical legacy25_raw RGB geometry at25pixels/degree"
            )
        if min(self.epochs, self.epoch_samples, self.batch_size, self.base_channels) < 1:
            raise ValueError("Epochs, epoch_samples, batch_size and base_channels must be positive")
        if self.patch_size < 32 or self.view_size < 32 or not 1 <= self.depth <= 5:
            raise ValueError("Patch/view sizes must be >=32 and depth in1..5")
        if self.norm not in {"batch", "group"} or self.train_view not in {"patch", "whole"}:
            raise ValueError("Unsupported normalization or training view")
        if self.scheduler not in {"constant", "cosine"} or self.workers < 0:
            raise ValueError("Unsupported scheduler or negative worker count")
        if self.sampling not in {
            "uniform",
            "dataset_balanced",
            "task_balanced",
            "dataset_task_balanced",
        }:
            raise ValueError("Unsupported source/task sampling policy")
        if self.augmentation not in {"basic", "mild", "strong"}:
            raise ValueError("Unsupported augmentation profile")
        if not 0 <= self.foreground_probability <= 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Foreground probability/min_lr_ratio must be in [0,1]")
        values = (self.lr, self.weight_decay, self.bce_weight, self.dice_weight, self.av_weight)
        if any(not math.isfinite(x) or x < 0 for x in values) or self.lr == 0:
            raise ValueError(
                "Learning rate and loss/decay weights must be finite and nonnegative; lr positive"
            )
        if self.bce_weight + self.dice_weight == 0 or self.av_weight == 0:
            raise ValueError("Segmentation and conditional A/V objectives must both be enabled")
        if self.tile_size < 32 or not 0 <= self.overlap < self.tile_size:
            raise ValueError("Invalid inference tile size or overlap")
        if self.selection_metric not in {"mean_dice", "macro_dataset_mean_dice"}:
            raise ValueError("Unsupported validation selection metric")
        if self.purpose not in {"functionality", "full_data"}:
            raise ValueError("purpose must be functionality or full_data")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _photo_id(root, row):
    return str(row.get("photo_id") or (root / row["image"]).resolve())


def read_unified_manifest(manifest: Path, split: str) -> list[dict]:
    root = Path(manifest).parent
    data = json.loads(Path(manifest).read_text())
    rows = data["records"] if isinstance(data, dict) else data
    partitions = {}
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate manifest record ID: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {row['split']}")
        for group in [
            ("source", str(row.get("source_id") or (root / row["image"]).resolve())),
            ("photo", _photo_id(root, row)),
        ]:
            if group in partitions and partitions[group] != row["split"]:
                raise ValueError(f"Original source/photo appears in multiple splits: {group}")
            partitions[group] = row["split"]
    result = [
        r
        for r in rows
        if r["split"] == split
        and r["task"] in SOURCE_CHANNELS
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    ]
    if not result:
        raise ValueError(f"No usable labeled records in {split!r}")
    return result


def map_record_targets(row: dict, target: np.ndarray, valid: np.ndarray):
    """Map an existing task's C,H,W annotation into five partially labeled heads."""
    if row["task"] not in SOURCE_CHANNELS:
        raise ValueError("Only labeled vessel/AV/disc_cup records can be mapped")
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    channels = SOURCE_CHANNELS[row["task"]]
    if target.ndim != 3 or len(target) != len(channels) or valid.shape != target.shape:
        raise ValueError(f"Invalid source annotation geometry for {row['id']}")
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError(f"Invalid source annotation range for {row['id']}")
    mapped = np.zeros((5, *target.shape[-2:]), np.float32)
    support = np.zeros_like(mapped, dtype=bool)
    mapped[list(channels)] = target
    support[list(channels)] = valid
    return mapped, support


def _rgb(root, relative, derived_cache=None):
    if derived_cache is not None:
        if Path(root).resolve() != derived_cache.canonical_root:
            raise ValueError("Cache was verified against a different canonical directory")
        return derived_cache.rgb(relative)
    with Image.open(Path(root) / relative) as image:
        return np.asarray(image.convert("RGB")).copy()


def _labels(root, relative, derived_cache=None):
    return (
        derived_cache.labels(relative)
        if derived_cache is not None
        else np.load(Path(root) / relative, allow_pickle=False)
    )


def read_unified_sample(root: Path, row: dict, *, derived_cache=None):
    """Public row reader for independent evaluation scripts; RGB in [0,1]."""
    image = np.asarray(_rgb(root, row["image"], derived_cache), dtype=np.float32) / 255
    with _labels(root, row["mask"], derived_cache) as labels:
        target = labels["target"].astype(np.float32)
        valid = labels["valid"].astype(bool)
        if "fov" in labels:
            fov = labels["fov"].astype(bool)
            if fov.shape != image.shape[:2]:
                raise ValueError("FOV/image geometry mismatch")
            valid &= fov[None]
    if target.shape[-2:] != image.shape[:2]:
        raise ValueError(f"Image/annotation geometry mismatch: {row['id']}")
    mapped, support = map_record_targets(row, target, valid)
    return (
        torch.from_numpy(image.transpose(2, 0, 1).copy()),
        torch.from_numpy(mapped),
        torch.from_numpy(support),
    )


class CoverageSampler(Sampler):
    """Shuffled complete photo pass followed by source/task-balanced extras."""

    def __init__(self, weights, count, generator, coverage_first=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.count = count
        self.generator = generator
        self.coverage_first = coverage_first

    def __len__(self):
        return self.count

    def __iter__(self):
        first = min(len(self.weights), self.count) if self.coverage_first else 0
        indices = torch.randperm(len(self.weights), generator=self.generator)[:first].tolist()
        if self.count > first:
            indices += torch.multinomial(
                self.weights, self.count - first, replacement=True, generator=self.generator
            ).tolist()
        return iter(indices)


def _supervision_mask(target, valid):
    return (
        valid[0]
        | valid[3]
        | valid[4]
        | (valid[1] & valid[2] & ((target[1] >= 0.5) ^ (target[2] >= 0.5)))
    )


class UnifiedDataset(torch.utils.data.Dataset):
    """One photo per sample, one randomly selected annotator per available task.

    A/V annotations carry their consistent vessel target. Training chooses one
    vascular observer bundle; validation prefers A/V where present. Disc/cup always
    remain an annotator pair. Patient/source IDs constrain splits, not photo merges.
    """

    def __init__(
        self,
        manifest: Path,
        split="train",
        config: UnifiedConfig | None = None,
        *,
        return_trace=False,
        derived_cache=None,
    ):
        self.return_trace = return_trace
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.derived_cache = derived_cache
        if (
            derived_cache is not None
            and derived_cache.manifest.resolve() != self.manifest.resolve()
        ):
            raise ValueError("Derived cache must be verified against this manifest")
        self.split = split
        self.config = config or UnifiedConfig()
        self.config.validate()
        self.rows = read_unified_manifest(manifest, split)
        for row in self.rows:
            details = row.get("metadata", {})
            profile = row.get("preprocessing_profile", details.get("preprocessing_profile"))
            ppd = row.get("pixels_per_degree", details.get("pixels_per_degree"))
            if profile is not None and profile != self.config.preprocessing_profile:
                raise ValueError("Unified manifest contains a different preprocessing profile")
            if ppd is not None and ppd != self.config.pixels_per_degree:
                raise ValueError("Unified manifest physical pixel scale differs")
            if self.config.purpose == "full_data":
                if row.get("preparation_preview", details.get("preparation_preview", False)):
                    raise ValueError(
                        "Full-data training rejects preview/limited preparation records"
                    )
                if profile is None or ppd is None:
                    raise ValueError(
                        "Full-data training requires explicit per-record legacy25_raw/ppd metadata"
                    )
                if row.get("preparation_max_size") is not None:
                    raise ValueError(
                        "Legacy25 physical frames must not use modern max-size cap resizing"
                    )
        groups = defaultdict(list)
        for row in self.rows:
            groups[_photo_id(self.root, row)].append(row)
        self.groups = [sorted(group, key=lambda r: r["id"]) for _, group in sorted(groups.items())]
        self.photo_ids = sorted(groups)
        self._image_cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 128 * 1024**2
        self._verified_aliases = set()
        self._augmentation_generator = None
        self._row_tasks = {}
        self._support_summary = None

    def __len__(self):
        return len(self.groups)

    def _image(self, index):
        key = self.photo_ids[index]
        rows = self.groups[index]
        if key in self._image_cache:
            image = self._image_cache.pop(key)
            self._image_cache[key] = image
            return image
        paths = sorted({str((self.root / r["image"]).resolve()) for r in rows})
        array = _rgb(
            self.root,
            Path(paths[0]).relative_to(self.root.resolve()).as_posix(),
            self.derived_cache,
        )
        if key not in self._verified_aliases:
            for path in paths[1:]:
                other = _rgb(
                    self.root,
                    Path(path).relative_to(self.root.resolve()).as_posix(),
                    self.derived_cache,
                )
                if not np.array_equal(array, other):
                    raise ValueError(f"photo_id aliases contain different RGB pixels: {key}")
            self._verified_aliases.add(key)
        image = torch.from_numpy(array.transpose(2, 0, 1).copy()).float() / 255
        size = image.numel() * image.element_size()
        while self._image_cache and self._cache_bytes + size > self._cache_limit:
            _, previous = self._image_cache.popitem(last=False)
            self._cache_bytes -= previous.numel() * previous.element_size()
        if size <= self._cache_limit:
            self._image_cache[key] = image
            self._cache_bytes += size
        return image

    def sample(self, index, *, random_annotator=False, return_trace=False):
        image = self._image(index)
        h, w = image.shape[-2:]
        target = torch.zeros((5, h, w), dtype=torch.float32)
        valid = torch.zeros_like(target, dtype=torch.bool)
        groups = defaultdict(list)
        for row in self.groups[index]:
            groups[row["task"]].append(row)
        vascular = [task for task in ("av", "vessels") if groups[task]]
        chosen_vascular = (
            vascular[int(torch.randint(len(vascular), (1,)))]
            if random_annotator and len(vascular) > 1
            else (vascular[0] if vascular else None)
        )
        trace = []
        for task in (chosen_vascular, "disc_cup"):
            if task is None:
                continue
            options = groups[task]
            if not options:
                continue
            selected = (
                int(torch.randint(len(options), (1,)))
                if random_annotator and len(options) > 1
                else 0
            )
            row = options[selected]
            trace.append((row["id"], task))
            with _labels(self.root, row["mask"], self.derived_cache) as data:
                raw = data["target"].astype(np.float32)
                support = data["valid"].astype(bool)
                if "fov" in data:
                    fov = data["fov"].astype(bool)
                    if fov.shape != (h, w):
                        raise ValueError("FOV/photo geometry mismatch")
                    support &= fov[None]
            mapped, support = map_record_targets(row, raw, support)
            if mapped.shape[-2:] != (h, w):
                raise ValueError("Annotations assigned to one photo have different geometry")
            mapped = torch.from_numpy(mapped)
            support = torch.from_numpy(support)
            # Dense selection avoids constructing two variable-length index arrays.
            # Keep unsupported values unchanged, including soft annotation values.
            torch.where(support, mapped, target, out=target)
            valid |= support
        result = (image, target, valid)
        return (*result, trace) if return_trace else result

    def support_summary(self):
        if self._support_summary is not None:
            return self._support_summary
        flags = np.zeros(5, bool)
        pixels = np.zeros(5, np.int64)
        classes = np.zeros(2, np.int64)
        for row in self.rows:
            with _labels(self.root, row["mask"], self.derived_cache) as data:
                target = data["target"].astype(np.float32)
                valid = data["valid"].astype(bool)
                if "fov" in data:
                    valid &= data["fov"][None].astype(bool)
            target, valid = map_record_targets(row, target, valid)
            available = valid.any(axis=(1, 2))
            flags |= available
            pixels += valid.sum(axis=(1, 2))
            known = valid[1] & valid[2] & ((target[1] >= 0.5) ^ (target[2] >= 0.5))
            tasks = set()
            for channel, name in ((0, "vessel"), (3, "disc"), (4, "cup")):
                if valid[channel].any():
                    tasks.add(name)
            if known.any():
                tasks.add("av")
            self._row_tasks[row["id"]] = tasks
            classes[0] += int((known & (target[1] >= 0.5)).sum())
            classes[1] += int((known & (target[2] >= 0.5)).sum())
        flags[1:3] = bool(classes.sum())
        self._support_summary = {
            "supervised_channels": flags.tolist(),
            "annotated_pixels_per_record_sum": pixels.tolist(),
            "av_reference_class_pixels": {"artery": int(classes[0]), "vein": int(classes[1])},
            "records": len(self.rows),
            "photos": len(self.groups),
            "datasets": dict(Counter(row["dataset"] for row in self.rows)),
        }

        return self._support_summary

    def sampling_weights(self):
        self.support_summary()
        policy = self.config.sampling
        memberships = []
        for rows in self.groups:
            tasks = {task for row in rows for task in self._row_tasks[row["id"]]}
            declared = {str(row["sampling_group"]) for row in rows if "sampling_group" in row}
            if len(declared) > 1:
                raise ValueError("One photo has conflicting sampling_group assignments")
            source = next(iter(declared)) if declared else rows[0]["dataset"]
            if policy == "uniform":
                keys = {("all",)}
            elif policy == "dataset_balanced":
                keys = {(source,)}
            elif policy == "task_balanced":
                keys = {(task,) for task in tasks}
            else:
                keys = {(source, task) for task in tasks}
            memberships.append(keys)
        populations = Counter(key for keys in memberships for key in keys)
        return [sum(1 / populations[key] for key in keys) for keys in memberships]

    def __getitem__(self, index):
        image, target, valid, trace = self.sample(
            index, random_annotator=self.split == "train", return_trace=True
        )
        native = (image, target, valid)
        if self.split != "train":
            return image, target, valid
        cfg = self.config
        if cfg.train_view == "whole":
            image, target, valid = whole_view(image, target, valid, cfg.view_size)
        else:
            size = cfg.patch_size
            h, w = image.shape[-2:]
            padding = (0, max(0, size - w), 0, max(0, size - h))
            image = F.pad(image, padding, mode="replicate")
            target = F.pad(target, padding)
            valid = F.pad(valid, padding, value=False)
            h, w = image.shape[-2:]
            foreground = None
            if cfg.foreground_probability > 0 and torch.rand(()) < cfg.foreground_probability:
                positives = [(target[c] >= 0.5) & valid[c] for c in SEGMENTATION_CHANNELS]
                positives.append(valid[1] & valid[2] & ((target[1] >= 0.5) ^ (target[2] >= 0.5)))
                positives = [mask for mask in positives if mask.any()]
                if positives:
                    chosen = positives[int(torch.randint(len(positives), (1,)))]
                    foreground = torch.nonzero(chosen)
            if foreground is not None and len(foreground):
                point = foreground[int(torch.randint(len(foreground), (1,)))]
                y = max(0, min(h - size, int(point[0]) - int(torch.randint(size, (1,)))))
                x = max(0, min(w - size, int(point[1]) - int(torch.randint(size, (1,)))))
            else:
                y, x = (
                    int(torch.randint(h - size + 1, (1,))),
                    int(torch.randint(w - size + 1, (1,))),
                )
            image, target, valid = [
                a[..., y : y + size, x : x + size] for a in (image, target, valid)
            ]
        rotation = int(torch.randint(4, (1,)))
        image, target, valid = [torch.rot90(a, rotation, (-2, -1)) for a in (image, target, valid)]
        if bool(torch.randint(2, (1,))):
            image, target, valid = [a.flip(-1) for a in (image, target, valid)]
        if cfg.augmentation != "basic":
            from .augment import augment

            if self._augmentation_generator is None:
                self._augmentation_generator = torch.Generator().manual_seed(
                    (torch.initial_seed() + 104729) % (2**63 - 1)
                )
            image, target, valid = augment(
                image, target, valid, level=cfg.augmentation, generator=self._augmentation_generator
            )
        if not _supervision_mask(target, valid).any():
            # Keep every delivered training photo supervised, even when an affine
            # transform or random crop missed its sparse annotated region.
            image, target, valid = native
            if cfg.train_view == "whole":
                image, target, valid = whole_view(image, target, valid, cfg.view_size)
            else:
                points = torch.nonzero(_supervision_mask(target, valid))
                if not len(points):
                    raise ValueError(f"Photo has no usable supervision: {self.photo_ids[index]}")
                size = cfg.patch_size
                h, w = image.shape[-2:]
                padding = (0, max(0, size - w), 0, max(0, size - h))
                image, target, valid = (
                    F.pad(image, padding, mode="replicate"),
                    F.pad(target, padding),
                    F.pad(valid, padding),
                )
                y, x = points[len(points) // 2].tolist()
                y = max(0, min(image.shape[-2] - size, y - size // 2))
                x = max(0, min(image.shape[-1] - size, x - size // 2))
                image, target, valid = [
                    a[..., y : y + size, x : x + size] for a in (image, target, valid)
                ]
        if self.return_trace:
            used = []
            for record_id, task in trace:
                active = (
                    valid[3:5].any()
                    if task == "disc_cup"
                    else valid[0].any()
                    or (
                        task == "av"
                        and (valid[1] & valid[2] & ((target[1] >= 0.5) ^ (target[2] >= 0.5))).any()
                    )
                )
                if active:
                    used.append(record_id)
            return image, target, valid, index, json.dumps(used)
        return image, target, valid


def unified_loss(logits, target, valid, *, bce_weight=1.0, dice_weight=1.0, av_weight=1.0):
    """Equal task means over actually annotated heads; zero gradient for unknown GT."""
    if logits.shape != target.shape or valid.shape != target.shape or logits.shape[1] != 5:
        raise ValueError("Expected matching B,5,H,W logits/targets/validity")
    components = {}
    weighted = []
    weights = []
    for channel in SEGMENTATION_CHANNELS:
        support = valid[:, channel]
        if not support.any():
            continue
        z, t = logits[:, channel], target[:, channel]
        w = support.to(z.dtype)
        bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * w).sum() / w.sum()
        probability = z.sigmoid()
        intersection = (probability * t * w).sum()
        denominator = ((probability + t) * w).sum()
        dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
        component = bce_weight * bce + dice_weight * dice
        components[CHANNELS[channel]] = component
        weighted.append(component)
        weights.append(1.0)
    known = valid[:, 1] & valid[:, 2] & ((target[:, 1] >= 0.5) ^ (target[:, 2] >= 0.5))
    if known.any():
        labels = (target[:, 2] >= 0.5).long()
        ce = F.cross_entropy(logits[:, 1:3], labels, reduction="none")[known].mean()
        components["av"] = ce
        weighted.append(av_weight * ce)
        weights.append(av_weight)
    return (sum(weighted) / sum(weights) if weighted else logits.sum() * 0), components


def build_unified(config: UnifiedConfig, *, initialize=True):
    config.validate()
    lineage = {
        "initialization": "scratch",
        "historical_exposure": False,
        "independent_model_eligible": True,
    }
    if config.architecture == "legacy_universal":
        from .unified_legacy import LegacyUniversalStudent, initialize_legacy_universal

        model = LegacyUniversalStudent()
        if config.legacy_initialization is not None:
            # An inference/resume checkpoint is self-contained: never reopen the teacher.
            lineage = {
                "initialization": "original_universal_three_to_five",
                "historical_exposure": "unknown",
                "independent_model_eligible": False,
                "accuracy_claim": False,
                "allowed_evaluation_splits": ["train", "val"],
            }
            if initialize:
                lineage = initialize_legacy_universal(model, config.legacy_initialization)
    else:
        model = UNet(5, config.base_channels, config.depth, config.norm)
    model.lineage = lineage
    model.architecture_spec = (
        {
            "name": "legacy_universal",
            "filters": 32,
            "depth": 4,
            "dropout": 0,
            "normalization": "batch",
            "outputs": 5,
        }
        if config.architecture == "legacy_universal"
        else {
            "name": "compact_unet",
            "base_channels": config.base_channels,
            "depth": config.depth,
            "normalization": config.norm,
            "outputs": 5,
        }
    )
    model.unified_config = asdict(config)
    return model


def _activation(logits):
    probability = logits.sigmoid()
    probability[:, 1:3] = logits[:, 1:3].softmax(dim=1) * probability[:, 0:1]
    return probability


def _conditional_probabilities(joint):
    if not np.isfinite(joint).all():
        raise ValueError("Non-finite unified probabilities")
    probability = np.asarray(joint, dtype=np.float32).copy()
    denominator = probability[1:3].sum(axis=0, keepdims=True)
    probability[1:3] = np.divide(
        probability[1:3],
        denominator,
        out=np.full_like(probability[1:3], 0.5),
        where=denominator > 0,
    )
    return probability


def _joint_probabilities(probability):
    joint = np.asarray(probability, dtype=np.float32).copy()
    joint[1:3] *= joint[0:1]
    return joint


def restore_probabilities(probability, geometry):
    """Resize joint AV probabilities, then recover their conditional distribution."""
    restored = geometry.restore(_joint_probabilities(probability), order=1)
    return _conditional_probabilities(restored)


@torch.inference_mode()
def predict_probabilities(model, image, device, config=None, tile_size=512, overlap=64):
    """C=5 probabilities at prepared geometry; artery/vein are conditional softmax."""
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("Expected prepared RGB CHW float in[0,1]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(
            "Expected prepared RGB/255 in[0,1]; do not pass already-centered PreparedImage.tensor()"
        )
    cfg = (
        asdict(config)
        if isinstance(config, UnifiedConfig)
        else config or getattr(model, "unified_config", {})
    )
    model.eval()
    h, w = image.shape[-2:]
    if cfg.get("train_view", "patch") == "whole":
        size = cfg.get("view_size", 512)
        nh, nw, top, left = letterbox_geometry(h, w, size)
        view, _, _ = whole_view(image, size=size)
        probability = _activation(model(view[None].to(device) - 0.5))
        probability = probability[..., top : top + nh, left : left + nw]
        restored = (
            F.interpolate(probability, size=(h, w), mode="bilinear", align_corners=False)[0]
            .cpu()
            .numpy()
        )
        return _conditional_probabilities(restored)
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError("Invalid tiling geometry")
    result = np.zeros((5, h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    for y in tile_starts(h, tile_size, overlap):
        for x in tile_starts(w, tile_size, overlap):
            tile = image[:, y : y + tile_size, x : x + tile_size]
            probability = _activation(model(tile[None].to(device) - 0.5))[0].cpu().numpy()
            th, tw = probability.shape[-2:]
            wy = np.maximum(np.hanning(th), 0.05) if th > 1 else np.ones(1)
            wx = np.maximum(np.hanning(tw), 0.05) if tw > 1 else np.ones(1)
            weight = (wy[:, None] * wx[None]).astype(np.float32)
            result[:, y : y + th, x : x + tw] += probability * weight
            weights[y : y + th, x : x + tw] += weight
    return _conditional_probabilities(result / weights[None])


def decode_probabilities(probability, threshold=0.5, supported=None):
    if probability.shape[0] != 5 or not 0 < threshold < 1:
        raise ValueError("Expected5channels and threshold in(0,1)")
    masks = probability >= threshold
    masks[1] = probability[1] >= probability[2]
    masks[2] = ~masks[1]
    if supported is not None:
        masks[~np.asarray(supported, dtype=bool)] = False
    masks[1] &= masks[0]
    masks[2] &= masks[0]
    masks[4] &= masks[3]
    return masks


def _metrics(counts):
    tp, fp, fn, n = map(int, counts)
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid_pixels": n,
    }


def evaluate_unified_model(
    model, manifest, split="val", device="cpu", config=None, threshold=0.5, derived_cache=None
):
    cfg = (
        UnifiedConfig(**config)
        if isinstance(config, dict)
        else config or UnifiedConfig(**getattr(model, "unified_config", {}))
    )
    historical = (
        cfg.legacy_initialization is not None
        or getattr(model, "lineage", {}).get("historical_exposure") == "unknown"
    )
    if historical and split not in {"train", "val"}:
        raise ValueError(
            "Historical initialization has unknown teacher exposure; exploratory train/val evaluation only"
        )
    dataset = UnifiedDataset(manifest, split, cfg, derived_cache=derived_cache)
    counts = np.zeros((5, 4), np.int64)
    by_dataset = {}
    per_photo = []
    supported = np.asarray(getattr(model, "supervised_channels", [True] * 5), dtype=bool)
    av_correct = av_conditional = av_count = 0
    for index, rows in enumerate(dataset.groups):
        image, target, valid, trace = dataset.sample(
            index, random_annotator=False, return_trace=True
        )
        probability = predict_probabilities(model, image, device, cfg, cfg.tile_size, cfg.overlap)
        masks = decode_probabilities(probability, threshold, supported)
        t, v = target.numpy() >= 0.5, valid.numpy()
        v[~supported] = False
        values = np.zeros((5, 4), np.int64)
        for c in range(5):
            values[c] = [
                (masks[c] & t[c] & v[c]).sum(),
                (masks[c] & ~t[c] & v[c]).sum(),
                (~masks[c] & t[c] & v[c]).sum(),
                v[c].sum(),
            ]
        counts += values
        source = rows[0]["dataset"]
        channel_datasets = {}
        row_lookup = {row["id"]: row for row in rows}
        for record_id, task in trace:
            annotation_source = row_lookup[record_id]["dataset"]
            by_dataset.setdefault(annotation_source, np.zeros_like(counts))
            for channel in SOURCE_CHANNELS[task]:
                by_dataset[annotation_source][channel] += values[channel]
                channel_datasets[CHANNELS[channel]] = annotation_source
        known = v[1] & v[2] & (t[1] ^ t[2])
        photo_av_count = int(known.sum())
        photo_conditional = int(((probability[1] >= probability[2]) == t[1])[known].sum())
        photo_end_to_end = int(((masks[1] & t[1] | masks[2] & t[2]) & known).sum())
        av_count += photo_av_count
        av_conditional += photo_conditional
        av_correct += photo_end_to_end
        per_photo.append(
            {
                "photo_id": dataset.photo_ids[index],
                "record_ids": [r["id"] for r in rows],
                "dataset": source,
                "channel_datasets": channel_datasets,
                "av_conditional_counts": {
                    "correct": photo_conditional,
                    "scored_pixels": photo_av_count,
                    "eligible_pixels": photo_av_count,
                    "end_to_end_correct": photo_end_to_end,
                },
                "channels": {
                    name: _metrics(value) for name, value in zip(CHANNELS, values, strict=True)
                },
            }
        )
    channels = {name: _metrics(value) for name, value in zip(CHANNELS, counts, strict=True)}
    scores = [value["dice"] for value in channels.values() if value["dice"] is not None]
    dataset_metrics = {
        name: {channel: _metrics(value) for channel, value in zip(CHANNELS, values, strict=True)}
        for name, values in by_dataset.items()
    }
    macro_channels = {}
    for channel in CHANNELS:
        available = [
            values[channel]["dice"]
            for values in dataset_metrics.values()
            if values[channel]["dice"] is not None
        ]
        macro_channels[channel] = float(np.mean(available)) if available else None
    macro_scores = [value for value in macro_channels.values() if value is not None]
    return {
        "family": "unified",
        "purpose": cfg.purpose,
        "accuracy_claim": False
        if cfg.purpose == "functionality" or historical
        else "requires fair benchmark",
        "lineage": getattr(model, "lineage", {}),
        "split": split,
        "photos": len(dataset),
        "records": len(dataset.rows),
        "channels": channels,
        "mean_dice": float(np.mean(scores)) if scores else None,
        "macro_dataset_mean_dice": float(np.mean(macro_scores)) if macro_scores else None,
        "macro_dataset_channel_dice": macro_channels,
        "macro_dataset_description": "mean supported dataset Dice per channel, then mean available channels; source follows selected annotation bundle",
        "av_accuracy": av_correct / av_count if av_count else None,
        "av_conditional_accuracy": av_conditional / av_count if av_count else None,
        "av_labeled_pixels": av_count,
        "supervised_channels": supported.tolist(),
        "metric_support": {
            name: "reference_class_labeled_vessel_pixels"
            if i in (1, 2)
            else "annotated_valid_pixels_within_field_of_view"
            for i, name in enumerate(CHANNELS)
        },
        "evaluation_policy": "one photo once; first sorted record per task; AV vessel annotation takes priority; all splits fixed by manifest",
        "av_decoding": "argmax of two conditional logits gated by vessel probability >= threshold; no independent AV class threshold",
        "by_dataset": dataset_metrics,
        "per_photo": per_photo,
        "threshold": threshold,
    }


def load_unified(path, device="auto"):
    """Load one immutable byte snapshot from a path or binary stream, safely."""
    snapshot = path.read() if hasattr(path, "read") else Path(path).read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    payload["checkpoint_sha256"] = hashlib.sha256(snapshot).hexdigest()
    if payload.get("format_version") != 1 or payload.get("family") != "unified":
        raise ValueError("Expected unified format_version1 checkpoint")
    if tuple(payload["channels"]) != CHANNELS:
        raise ValueError("Unified output channel order mismatch")
    if payload.get("normalization") != "RGB/255-0.5":
        raise ValueError("Unified normalization contract mismatch")
    config = UnifiedConfig(**payload["config"])
    model = build_unified(config, initialize=False)
    model.lineage = payload.get("lineage", model.lineage)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.supervised_channels = payload.get("supervised_channels", [True] * 5)
    if len(model.supervised_channels) != 5:
        raise ValueError("Unified supervised channel metadata mismatch")
    device = resolve_device(device)
    return model.to(device).eval(), payload, device


def inference_config(payload, *, tile_size=None, overlap=None):
    """Use checkpoint geometry; explicit tiling overrides apply only to patch models."""
    from dataclasses import replace

    config = UnifiedConfig(**payload["config"])
    if config.train_view == "whole" and (tile_size is not None or overlap is not None):
        raise ValueError("Whole-view unified checkpoints do not accept tiling overrides")
    settings = {
        key: value
        for key, value in {"tile_size": tile_size, "overlap": overlap}.items()
        if value is not None
    }
    config = replace(config, **settings)
    config.validate()
    return config


def predict_unified(
    checkpoint: Path,
    image_path: Path,
    output: Path,
    *,
    device="auto",
    dataset=None,
    fov=None,
    field_of_view_degrees=None,
    threshold=0.5,
    tile_size=None,
    overlap=None,
):
    """Complete raw-photo prediction with the verified legacy25_raw geometry.

    Returns original-resolution probabilities, five masks and overlays. Camera
    field angle/official FOV can be supplied exactly as for the reference model.
    """
    from .legacy.preprocessing import prepare_single_net

    model, payload, device = load_unified(checkpoint, device)
    with Image.open(image_path) as photo:
        original = np.asarray(photo.convert("RGB")).copy()
    config = inference_config(payload, tile_size=tile_size, overlap=overlap)
    if not 0 < threshold < 1:
        raise ValueError("Unified threshold must be in (0,1)")
    prepared = prepare_single_net(
        original,
        dataset=dataset,
        fov=fov,
        field_of_view_degrees=field_of_view_degrees,
        pixels_per_degree=config.pixels_per_degree,
        normalized=False,
    )
    # PreparedImage.tensor() is already centered; low-level inference expects01.
    tensor = torch.from_numpy(prepared.image.transpose(2, 0, 1).copy()).float() / 255
    probability = predict_probabilities(
        model, tensor, device, config, config.tile_size, config.overlap
    )
    probability[list(SEGMENTATION_CHANNELS)] *= prepared.fov[None]
    probability = restore_probabilities(probability, prepared.geometry)
    domain = prepared.geometry.restore(prepared.fov.astype(np.uint8), order=0) > 0
    probability[list(SEGMENTATION_CHANNELS)] *= domain[None]
    probability[1:3, ~domain] = 0.5
    masks = decode_probabilities(probability, threshold, payload["supervised_channels"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"probabilities": str(output / "probabilities.npz")}
    np.savez_compressed(
        output / "probabilities.npz",
        probability=probability,
        channels=np.asarray(CHANNELS),
        supervised_channels=np.asarray(payload["supervised_channels"], bool),
        fov=domain,
        conditional_channels=np.asarray([1, 2], dtype=np.int64),
    )
    colors = np.zeros_like(original)
    palette = [(50, 210, 70), (255, 45, 45), (45, 90, 255), (255, 190, 40), (40, 220, 255)]
    for name, mask, color in zip(CHANNELS, masks, palette, strict=True):
        path = output / f"{name}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        paths[name] = str(path)
        colors[mask] = color
    av = np.zeros_like(original)
    av[masks[1]] = (255, 0, 0)
    av[masks[2]] = (0, 0, 255)
    Image.fromarray(av).save(output / "av.png")
    paths["av"] = str(output / "av.png")
    active = masks.any(axis=0)
    overlay = original.copy()
    overlay[active] = np.round(0.55 * original[active] + 0.45 * colors[active]).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.png")
    paths["overlay"] = str(output / "overlay.png")
    metadata = {
        "family": "unified",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "lineage": model.lineage,
        "architecture": model.architecture_spec,
        "accuracy_claim": payload.get("accuracy_claim", False),
        "image": str(image_path),
        "channels": list(CHANNELS),
        "supervised_channels": payload["supervised_channels"],
        "purpose": payload["purpose"],
        "preprocessing": {
            "profile": config.preprocessing_profile,
            "pixels_per_degree": config.pixels_per_degree,
            "field_of_view_degrees": prepared.field_of_view_degrees,
            "normalized": False,
            "normalization": "RGB/255-0.5",
            "fov_source": prepared.fov_source,
            "geometry": prepared.geometry.as_dict(),
        },
        "probability_semantics": [
            "P(vessel|RGB)",
            "P(artery|vessel,RGB)",
            "P(vein|vessel,RGB)",
            "P(disc|RGB)",
            "P(cup|RGB)",
        ],
        "fusion": "joint AV probabilities interpolated/blended before recovering conditional AV",
        "decoding": "conditional artery/vein argmax gated by vessel threshold; cup within disc; outside FOV segmentation probabilities zero, conditional probabilities0.5",
        "threshold": threshold,
        "inference": {
            "mode": "whole" if config.train_view == "whole" else "tiled",
            "tile_size": config.tile_size if config.train_view == "patch" else None,
            "overlap": config.overlap if config.train_view == "patch" else None,
            "view_size": config.view_size if config.train_view == "whole" else None,
            "checkpoint_tile_size": payload["config"].get("tile_size", 512),
            "checkpoint_overlap": payload["config"].get("overlap", 64),
            "overrides": {
                key: value
                for key, value in {"tile_size": tile_size, "overlap": overlap}.items()
                if value is not None
            },
        },
        "outputs": paths,
    }
    _write_json(output / "prediction.json", metadata)
    paths["metadata"] = str(output / "prediction.json")
    return paths


def _rng_state(device, dataset, generator):
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "sampler": generator.get_state(),
        "augmentation": dataset._augmentation_generator.get_state()
        if dataset._augmentation_generator is not None
        else None,
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, device, dataset, generator):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    generator.set_state(state["sampler"])
    if state["augmentation"] is not None:
        dataset._augmentation_generator = torch.Generator().set_state(state["augmentation"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _training_source_sha256():
    """Identity of executable model/data/training sources, independent of checkout path."""
    root = Path(__file__).parent
    result = hashlib.sha256()
    for name in (
        "unified.py",
        "unified_legacy.py",
        "model.py",
        "dataset.py",
        "augment.py",
        "inference.py",
        "integrity.py",
        "derived_cache.py",
        "legacy/single_net.py",
        "legacy/preprocessing.py",
    ):
        path = root / name
        result.update(name.encode() + b"\0" + path.read_bytes())
    return result.hexdigest()


def _normalized_config(config):
    result = asdict(config if isinstance(config, UnifiedConfig) else UnifiedConfig(**config))
    result.pop("device")
    return result


def _reconcile_resume_outputs(output, saved, config, fields):
    """last.pt commits training progress; best.pt may safely lead by one epoch.

    A kill after publishing an improved model must not permit a weaker replay to
    overwrite it. Validation is embedded in best.pt so its sidecar is repairable.
    """
    best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if (
        best_payload.get("family") != "unified"
        or best_payload.get("manifest_sha256") != saved["manifest_sha256"]
        or best_payload.get("prepared_content_sha256") != saved.get("prepared_content_sha256")
        or _normalized_config(best_payload["config"]) != _normalized_config(config)
        or best_payload.get("training_source_sha256") != saved.get("training_source_sha256")
        or best_payload.get("derived_cache") != saved.get("derived_cache")
        or best_payload.get("lineage") != saved.get("lineage")
    ):
        raise ValueError("Resume best checkpoint provenance differs from committed last.pt")
    best_score = best_payload.get("val_selection_score", best_payload["val_mean_dice"])
    committed_best = saved["training_state"].get(
        "best_validation_score",
        saved["training_state"].get("best_validation_mean_dice", -float("inf")),
    )
    if (
        not math.isfinite(best_score)
        or best_score < committed_best
        or not 1 <= best_payload["epoch"] <= saved["epoch"] + 1
    ):
        raise ValueError("Resume best checkpoint score/epoch contradicts committed training state")
    validation = best_payload.get("validation_metrics")
    if validation is None:
        validation = json.loads((output / "validation.json").read_text())
        if (
            validation.get("selection_score") != best_score
            or validation.get("selection_metric") != config.selection_metric
        ):
            raise ValueError("Old best checkpoint lacks recoverable matching validation metadata")
    if (
        validation.get("selection_score") != best_score
        or validation.get("selection_metric") != config.selection_metric
    ):
        raise ValueError("Embedded best validation does not match checkpoint selection score")
    with (output / "metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError("Resume metrics.csv columns differ from training schema")
        rows = list(reader)
    committed = []
    for row in rows:
        try:
            epoch = int(row["epoch"])
        except (TypeError, ValueError):
            if len(committed) < saved["epoch"]:
                raise ValueError("Resume metrics.csv lost committed epoch rows") from None
            continue  # A process may have been interrupted during a trailing row write.
        if epoch <= saved["epoch"]:
            if any(row.get(field) in (None, "") for field in fields):
                raise ValueError("Resume metrics.csv has incomplete committed epoch rows")
            committed.append(row)
    if [int(row["epoch"]) for row in committed] != list(range(1, saved["epoch"] + 1)):
        raise ValueError("Resume metrics.csv does not contain exactly the committed epochs")
    temporary = output / "metrics.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(committed)
    temporary.replace(output / "metrics.csv")
    _write_json(output / "validation.json", validation)
    _write_json(output / "coverage.json", saved["training_state"].get("coverage", {}))
    _write_json(
        output / "resume_recovery.json",
        {
            "committed_epoch": saved["epoch"],
            "discarded_csv_rows": len(rows) - len(committed),
            "retained_best_epoch": best_payload["epoch"],
            "retained_best_score": best_score,
            "best_ahead_of_committed_epoch": best_payload["epoch"] > saved["epoch"],
            "training_source_sha256": saved.get("training_source_sha256"),
        },
    )
    return best_score


def train_unified(
    manifest: Path,
    output: Path,
    config: UnifiedConfig,
    *,
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
    validation_fn=None,
    expected_content_sha256: str | None = None,
    derived_cache_root: Path | None = None,
    expected_cache_fingerprint: str | None = None,
):
    """Train the shared network; purpose=functionality never implies model quality.

    Resume is supported at completed epoch boundaries using last.pt, including
    optimizer/scheduler and RNG state. best.pt is an inference-only checkpoint.
    """
    config.validate()
    manifest = Path(manifest)
    output = Path(output)
    derived_cache = None
    if derived_cache_root is not None:
        if not expected_content_sha256 or not expected_cache_fingerprint:
            raise ValueError("Derived cache requires explicit canonical and cache fingerprint pins")
        derived_cache = VerifiedCache(
            derived_cache_root,
            manifest,
            expected_cache_fingerprint=expected_cache_fingerprint,
            expected_canonical_fingerprint=expected_content_sha256,
        )
        required_rows = [
            row for row in json.loads(manifest.read_text()) if row["split"] in {"train", "val"}
        ]
        derived_cache.require_coverage(required_rows)
    elif expected_cache_fingerprint is not None:
        raise ValueError("A cache fingerprint requires an explicit runtime cache directory")
    cache_identity = derived_cache.identity if derived_cache is not None else None
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    training_source_sha256 = _training_source_sha256()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = UnifiedDataset(
        manifest, "train", config, return_trace=True, derived_cache=derived_cache
    )
    if config.purpose == "full_data" and (
        not config.coverage_first or config.epoch_samples < len(dataset)
    ):
        raise ValueError(
            "Full-data training requires coverage_first and epoch_samples >= train photos"
        )
    validation_rows = read_unified_manifest(manifest, "val")
    support = dataset.support_summary()
    validation_support = None
    if config.purpose == "full_data" and config.require_all_heads:
        validation_support = UnifiedDataset(
            manifest, "val", config, derived_cache=derived_cache
        ).support_summary()
        if (
            not all(validation_support["supervised_channels"])
            or min(validation_support["av_reference_class_pixels"].values()) == 0
        ):
            raise ValueError(
                "Full-data model selection requires validation GT for all5heads and both AV classes"
            )
    flags = support["supervised_channels"]
    if config.require_all_heads and (
        not all(flags) or min(support["av_reference_class_pixels"].values()) == 0
    ):
        raise ValueError("Training data must supervise all5outputs and both conditional AV classes")
    caps = {row.get("preparation_max_size") for row in dataset.rows + validation_rows}
    if len(caps) != 1:
        raise ValueError("Training/validation preparation scales differ")
    inference_max_size = caps.pop()
    if inference_max_size is not None and inference_max_size < 16:
        raise ValueError("Invalid preparation_max_size")
    prepared_content_sha256 = (
        verify_prepared_content(manifest, expected=expected_content_sha256)
        if config.purpose == "full_data" or expected_content_sha256 is not None
        else None
    )
    generator = torch.Generator().manual_seed(config.seed)
    sampler = CoverageSampler(
        dataset.sampling_weights(), config.epoch_samples, generator, config.coverage_first
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        generator=generator,
    )
    model = build_unified(config, initialize=resume is None).to(device)
    model.supervised_channels = flags
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * config.min_lr_ratio
        )
        if config.scheduler == "cosine"
        else None
    )
    first_epoch = 1
    steps = 0
    best = -float("inf")
    photo_visits = Counter()
    annotation_visits = Counter()
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("family") != "unified" or "training_state" not in saved:
            raise ValueError("Resume requires unified last.pt with training state")
        if saved["manifest_sha256"] != digest:
            raise ValueError("Resume manifest changed")
        if saved.get("prepared_content_sha256") != prepared_content_sha256:
            raise ValueError("Resume prepared image/annotation content changed")
        if saved.get("derived_cache") != cache_identity:
            raise ValueError("Resume derived-cache identity changed; runtime paths may relocate")
        if saved.get("training_source_sha256", training_source_sha256) != training_source_sha256:
            raise ValueError("Resume executable training sources changed")
        original = asdict(UnifiedConfig(**saved["config"]))
        current = asdict(config)
        original.pop("device")
        current.pop("device")
        if original != current:
            raise ValueError("Resume training configuration changed")
        model.load_state_dict(saved["state_dict"])
        model.lineage = saved.get("lineage", model.lineage)
        state = saved["training_state"]
        photo_visits.update(state.get("coverage", {}).get("photo_visits", {}))
        annotation_visits.update(state.get("coverage", {}).get("annotation_visits", {}))
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        first_epoch = saved["epoch"] + 1
        steps = saved["steps"]
        best = state.get(
            "best_validation_score", state.get("best_validation_mean_dice", -float("inf"))
        )
        _restore_rng(state["rng"], device, dataset, generator)
        if not (output / "metrics.csv").is_file() or not (output / "best.pt").is_file():
            raise ValueError("Resume in the original run directory with metrics.csv and best.pt")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Choose a new unified run directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        _write_json(
            output / "run.json",
            {
                "family": "unified",
                "lineage": model.lineage,
                "architecture": model.architecture_spec,
                "config": asdict(config),
                "manifest": str(manifest.resolve()),
                "manifest_sha256": digest,
                "prepared_content_sha256": prepared_content_sha256,
                "derived_cache": cache_identity,
                "training_source_sha256": training_source_sha256,
                "normalization": "RGB/255-0.5",
                "preprocessing": {
                    "profile": config.preprocessing_profile,
                    "pixels_per_degree": config.pixels_per_degree,
                    "normalized": False,
                },
                "channels": list(CHANNELS),
                "activation": [
                    "sigmoid",
                    "conditional_softmax",
                    "conditional_softmax",
                    "sigmoid",
                    "sigmoid",
                ],
                "support": support,
                "validation_support": validation_support,
                "parameters": sum(p.numel() for p in model.parameters()),
                "torch": str(torch.__version__),
                "inference_max_size": inference_max_size,
                "purpose": config.purpose,
                "accuracy_claim": False
                if config.purpose == "functionality" or config.legacy_initialization is not None
                else "requires independent benchmark",
            },
        )
    fields = [
        "epoch",
        "steps",
        "lr",
        "train_loss",
        "val_mean_dice",
        "val_macro_dataset_mean_dice",
        "val_selection_score",
        "train_seconds",
        "validation_seconds",
    ]
    if resume is not None:
        best = _reconcile_resume_outputs(output, saved, config, fields)
    started = time.perf_counter()
    completed_epoch = first_epoch - 1
    with (output / "metrics.csv").open("a" if resume else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if resume is None:
            writer.writeheader()
        for epoch in range(first_epoch, config.epochs + 1):
            if config.max_steps is not None and steps >= config.max_steps:
                break
            model.train()
            total = 0.0
            batches = 0
            train_started = time.perf_counter()
            for image, target, valid, indices, traces in loader:
                image, target, valid = image.to(device), target.to(device), valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = unified_loss(
                    model(image - 0.5),
                    target,
                    valid,
                    bce_weight=config.bce_weight,
                    dice_weight=config.dice_weight,
                    av_weight=config.av_weight,
                )
                if not components:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite unified training loss")
                loss.backward()
                optimizer.step()
                # Count only samples delivered to a successful optimizer update;
                # worker prefetch never advances these counters.
                for index, trace, sample_target, sample_valid in zip(
                    indices.tolist(), traces, target, valid
                ):
                    if _supervision_mask(sample_target, sample_valid).any():
                        photo_visits[dataset.photo_ids[index]] += 1
                        annotation_visits.update(json.loads(trace))
                steps += 1
                total += float(loss.detach())
                batches += 1
                if config.max_steps is not None and steps >= config.max_steps:
                    break
            if not batches:
                raise ValueError("No supervised pixels were sampled in this epoch")
            train_seconds = time.perf_counter() - train_started
            val_started = time.perf_counter()
            metrics = (
                validation_fn(model, manifest, split="val", device=device, config=config)
                if validation_fn is not None
                else evaluate_unified_model(
                    model,
                    manifest,
                    split="val",
                    device=device,
                    config=config,
                    **({"derived_cache": derived_cache} if derived_cache is not None else {}),
                )
            )
            score = metrics[config.selection_metric]
            if score is None or not math.isfinite(score):
                raise ValueError(f"Undefined unified validation metric: {config.selection_metric}")
            metrics = dict(metrics, selection_metric=config.selection_metric, selection_score=score)
            val_seconds = time.perf_counter() - val_started
            by_dataset = {}
            for name in sorted({row["dataset"] for row in dataset.rows}):
                eligible = {
                    dataset.photo_ids[i]
                    for i, rows in enumerate(dataset.groups)
                    if any(row["dataset"] == name for row in rows)
                }
                by_dataset[name] = {
                    "eligible_photos": len(eligible),
                    "seen_photos": sum(photo_visits[key] > 0 for key in eligible),
                }
            coverage = {
                "eligible_photos": len(dataset),
                "seen_photos": len(photo_visits),
                "eligible_annotation_records": len(dataset.rows),
                "contributing_annotation_records": len(annotation_visits),
                "photo_visits": dict(photo_visits),
                "annotation_visits": dict(annotation_visits),
                "by_dataset": by_dataset,
                "counting": "actual supervised samples delivered to successful optimizer updates",
            }
            _write_json(output / "coverage.json", coverage)
            payload = {
                "format_version": 1,
                "family": "unified",
                "lineage": model.lineage,
                "architecture": model.architecture_spec,
                "accuracy_claim": False
                if config.legacy_initialization is not None or config.purpose == "functionality"
                else "requires independent benchmark",
                "channels": list(CHANNELS),
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "manifest_sha256": digest,
                "prepared_content_sha256": prepared_content_sha256,
                "derived_cache": cache_identity,
                "training_source_sha256": training_source_sha256,
                "supervised_channels": flags,
                "inference_max_size": inference_max_size,
                "normalization": "RGB/255-0.5",
                "preprocessing": {
                    "profile": config.preprocessing_profile,
                    "pixels_per_degree": config.pixels_per_degree,
                    "normalized": False,
                },
                "purpose": config.purpose,
                "probability_fusion": "joint_av_mixture_then_conditional",
                "coverage": coverage,
                "epoch": epoch,
                "steps": steps,
                "val_mean_dice": metrics["mean_dice"],
                "val_macro_dataset_mean_dice": metrics.get("macro_dataset_mean_dice"),
                "val_selection_score": score,
                "validation_metrics": metrics,
            }
            if score > best:
                best = score
                best_temporary = output / "best.pt.tmp"
                torch.save(payload, best_temporary)
                best_temporary.replace(output / "best.pt")
                _write_json(output / "validation.json", metrics)
            row = {
                "epoch": epoch,
                "steps": steps,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": total / batches,
                "val_mean_dice": metrics["mean_dice"],
                "val_macro_dataset_mean_dice": metrics.get("macro_dataset_mean_dice"),
                "val_selection_score": score,
                "train_seconds": train_seconds,
                "validation_seconds": val_seconds,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if scheduler is not None:
                scheduler.step()
            last = dict(
                payload,
                training_state={
                    "coverage": coverage,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "best_validation_score": best,
                    "selection_metric": config.selection_metric,
                    "rng": _rng_state(device, dataset, generator),
                },
            )
            temporary = output / "last.pt.tmp"
            torch.save(last, temporary)
            temporary.replace(output / "last.pt")
            completed_epoch = epoch
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
    _write_json(
        output / "training_summary.json",
        {
            "seconds_this_invocation": time.perf_counter() - started,
            "steps": steps,
            "best_validation_score": best,
            "selection_metric": config.selection_metric,
            "parameters": sum(p.numel() for p in model.parameters()),
            "purpose": config.purpose,
            "completed_budget": (output / "last.pt").exists()
            and (
                steps >= config.max_steps if config.max_steps else completed_epoch >= config.epochs
            ),
        },
    )
    return output / "best.pt"


def main(argv=None):
    """Experimental unified CLI; production quality requires the full-data benchmark."""
    import argparse
    import tomllib
    from dataclasses import fields

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train all five outputs jointly")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--device")
    training.add_argument("--purpose", choices=["functionality", "full_data"])
    training.add_argument("--epochs", type=int)
    training.add_argument("--epoch-samples", type=int)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--stop-after-epoch", type=int)
    training.add_argument("--derived-cache", type=Path)
    training.add_argument("--expected-cache-fingerprint")
    training.add_argument("--expected-content-sha256")
    prediction = commands.add_parser(
        "predict", help="Predict a raw fundus photograph with reversible legacy25 geometry"
    )
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="auto")
    prediction.add_argument("--dataset")
    prediction.add_argument("--field-of-view-degrees", type=float)
    prediction.add_argument("--fov", type=Path)
    prediction.add_argument("--threshold", type=float, default=0.5)
    inspect = commands.add_parser("inspect", help="Inspect labeled photo groups and head support")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            settings = {}
            if args.resume:
                settings = dict(
                    torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
                )
            if args.config:
                settings.update(
                    json.loads(args.config.read_text())
                    if args.config.suffix.lower() == ".json"
                    else tomllib.loads(args.config.read_text())
                )
            allowed = {field.name for field in fields(UnifiedConfig)}
            unknown = set(settings) - allowed
            if unknown:
                raise ValueError(f"Unknown unified settings: {sorted(unknown)}")
            for name in ("device", "purpose", "epochs", "epoch_samples", "max_steps"):
                if getattr(args, name) is not None:
                    settings[name] = getattr(args, name)
            config = UnifiedConfig(**settings)
            result = {
                "checkpoint": str(
                    train_unified(
                        args.manifest,
                        args.output,
                        config,
                        resume=args.resume,
                        stop_after_epoch=args.stop_after_epoch,
                        **{
                            key: value
                            for key, value in {
                                "derived_cache_root": args.derived_cache,
                                "expected_cache_fingerprint": args.expected_cache_fingerprint,
                                "expected_content_sha256": args.expected_content_sha256,
                            }.items()
                            if value is not None
                        },
                    )
                ),
                "purpose": config.purpose,
            }
        elif args.command == "predict":
            fov = None
            if args.fov:
                with Image.open(args.fov) as image:
                    fov = np.asarray(image.convert("L")) > 0
            result = predict_unified(
                args.checkpoint,
                args.input,
                args.output,
                device=args.device,
                dataset=args.dataset,
                fov=fov,
                field_of_view_degrees=args.field_of_view_degrees,
                threshold=args.threshold,
            )
        else:
            dataset = UnifiedDataset(args.manifest, args.split)
            result = {"split": args.split, **dataset.support_summary()}
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, f"unified: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
