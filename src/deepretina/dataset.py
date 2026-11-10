"""Manifest loading and seeded, spatially aligned training crops."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .model import CHANNELS


def read_manifest(manifest: Path, task: str, split: str) -> list[dict]:
    rows = json.loads(Path(manifest).read_text())
    if isinstance(rows, dict):
        rows = rows["records"]
    groups = {}
    image_splits = {}
    for row in rows:
        if row["image"] in image_splits and image_splits[row["image"]] != row["split"]:
            raise ValueError(f"Photograph {row['image']!r} appears in multiple splits")
        image_splits[row["image"]] = row["split"]
        group = row.get("source_id", row["image"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Source image {group!r} appears in multiple splits")
        groups[group] = row["split"]
    selected = [
        r
        for r in rows
        if r["task"] == task and r["split"] == split and r.get("usable_for_training", True)
    ]
    if not selected:
        raise ValueError(f"No {task} records in split {split!r} of {manifest}")
    return selected


def read_sample(root: Path, row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(root / row["image"]) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    with np.load(root / row["mask"], allow_pickle=False) as data:
        target = data["target"].astype(np.float32)
        valid = data["valid"].astype(bool)
        if "fov" in data:
            valid = valid & data["fov"][None].astype(bool)
    expected = (len(CHANNELS[row["task"]]), *image.shape[:2])
    if target.shape != expected or valid.shape != expected:
        raise ValueError(
            f"Invalid mask geometry for {row['id']}: expected {expected}, got {target.shape}/{valid.shape}"
        )
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError(f"Mask values outside [0,1] for {row['id']}")
    return (
        torch.from_numpy(image.transpose(2, 0, 1).copy()),
        torch.from_numpy(target),
        torch.from_numpy(valid.copy()),
    )


def letterbox_geometry(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """Rounded aspect-preserving geometry shared by training and inference."""
    if min(height, width) < 1 or size < 8:
        raise ValueError("Image dimensions must be positive and letterbox size >=8")
    scale = min(size / height, size / width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    return resized_h, resized_w, (size - resized_h) // 2, (size - resized_w) // 2


def whole_view(image, target=None, valid=None, size=512):
    """Symmetric zero letterbox; labels/validity use nearest, padded validity false."""
    height, width = image.shape[-2:]
    nh, nw, top, left = letterbox_geometry(height, width, size)
    padding = (left, size - nw - left, top, size - nh - top)
    image = F.interpolate(image[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    image = F.pad(image, padding, value=0)
    if target is not None:
        target = F.interpolate(target[None].float(), size=(nh, nw), mode="nearest")[0]
        target = F.pad(target, padding, value=0)
    if valid is not None:
        valid = F.interpolate(valid[None].float(), size=(nh, nw), mode="nearest")[0] > 0.5
        valid = F.pad(valid, padding, value=False)
    return image, target, valid


class CropDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        task: str,
        patch_size: int,
        samples_per_image: int = 16,
        *,
        train_view="patch",
        view_size=512,
        foreground_probability=0.0,
        augmentation="basic",
    ):
        self.root = Path(manifest).parent
        self.rows = read_manifest(manifest, task, "train")
        if patch_size < 8 or samples_per_image < 1:
            raise ValueError("patch_size must be >=8 and samples_per_image positive")
        if train_view not in {"patch", "whole"} or augmentation not in {"basic", "mild", "strong"}:
            raise ValueError("Unsupported view or augmentation profile")
        if not 0 <= foreground_probability <= 1 or view_size < 32:
            raise ValueError("Invalid foreground probability or whole view size")
        self.train_view, self.view_size = train_view, view_size
        self.foreground_probability, self.augmentation = foreground_probability, augmentation
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        # Bounded independently per DataLoader worker; tensors are never modified in place.
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 128 * 1024**2
        self._augmentation_generator = None

    def __len__(self):
        return len(self.rows) * self.samples_per_image

    def __getitem__(self, index):
        row_index = index % len(self.rows)
        if row_index in self._cache:
            sample = self._cache.pop(row_index)
            self._cache[row_index] = sample
        else:
            sample = read_sample(self.root, self.rows[row_index])
            sample_bytes = sum(t.numel() * t.element_size() for t in sample)
            while self._cache and self._cache_bytes + sample_bytes > self._cache_limit:
                _, previous = self._cache.popitem(last=False)
                self._cache_bytes -= sum(t.numel() * t.element_size() for t in previous)
            if sample_bytes <= self._cache_limit:
                self._cache[row_index] = sample
                self._cache_bytes += sample_bytes
        image, target, valid = sample
        if self.train_view == "whole":
            image, target, valid = whole_view(image, target, valid, self.view_size)
            items = [image, target, valid]
        else:
            size = self.patch_size
            h, w = image.shape[-2:]
            padding = (0, max(0, size - w), 0, max(0, size - h))
            image = F.pad(image, padding, mode="replicate")
            target = F.pad(target, padding)
            valid = F.pad(valid, padding, value=False)
            h, w = image.shape[-2:]
            foreground = None
            if self.foreground_probability > 0 and torch.rand(()) < self.foreground_probability:
                foreground = torch.nonzero(((target >= 0.5) & valid).any(dim=0))
            if foreground is not None and len(foreground):
                center = foreground[int(torch.randint(len(foreground), (1,)))]
                y = max(0, min(h - size, int(center[0]) - int(torch.randint(size, (1,)))))
                x = max(0, min(w - size, int(center[1]) - int(torch.randint(size, (1,)))))
            else:
                # Preserve the original two RNG draws when foreground_probability=0.
                y, x = (
                    int(torch.randint(h - size + 1, (1,))),
                    int(torch.randint(w - size + 1, (1,))),
                )
            items = [a[..., y : y + size, x : x + size] for a in (image, target, valid)]
        k = int(torch.randint(4, (1,)))
        items = [torch.rot90(a, k, (-2, -1)) for a in items]
        if bool(torch.randint(2, (1,))):
            items = [a.flip(-1) for a in items]
        if self.augmentation != "basic":
            from .augment import augment

            if self._augmentation_generator is None:
                self._augmentation_generator = torch.Generator().manual_seed(
                    (torch.initial_seed() + 104729) % (2**63 - 1)
                )
            items = list(
                augment(*items, level=self.augmentation, generator=self._augmentation_generator)
            )
        return tuple(items)
