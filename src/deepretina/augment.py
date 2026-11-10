"""Seeded joint augmentation for image, categorical targets and supervision masks.

Existing crop rotations/flips stay in CropDataset. These optional profiles add
small affine changes and bounded photometric variation for measured HPO trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AugmentProfile:
    rotation_degrees: float
    min_scale: float
    max_scale: float
    brightness: float
    contrast: float
    gamma: float
    channel_jitter: float
    noise_std: float
    affine_probability: float = 0.75
    photometric_probability: float = 0.9


PROFILES = {
    "mild": AugmentProfile(10, 0.9, 1.1, 0.06, 0.1, 0.1, 0.025, 0.01),
    "strong": AugmentProfile(20, 0.8, 1.2, 0.15, 0.2, 0.2, 0.08, 0.025),
}


def _check_geometry(image, target, valid):
    if image.ndim != 3 or target.ndim != 3 or valid.shape != target.shape:
        raise ValueError("Expected image CHW, target CHW and matching valid CHW")
    if image.shape[-2:] != target.shape[-2:] or min(image.shape[-2:]) < 2:
        raise ValueError("Image, target and valid must have the same nonempty spatial shape")
    if not image.is_floating_point() or not target.is_floating_point() or valid.dtype != torch.bool:
        raise TypeError("Image and target must be floating tensors; valid must be bool")
    if image.device != target.device or image.device != valid.device:
        raise ValueError("All input tensors must be on the same device")


def warp_joint(image, target, valid, *, angle_degrees=0.0, scale=1.0):
    """Apply one physical rotation/zoom to all channels, retaining the CHW frame.

    Image padding reflects the input. Target and validity padding are zero/false.
    Nearest-neighbour sampling preserves every categorical target value exactly.
    ``scale>1`` zooms in; angles use the image rotation convention of torch.rot90.
    """
    _check_geometry(image, target, valid)
    if not math.isfinite(angle_degrees) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Affine angle must be finite and scale finite and positive")
    height, width = image.shape[-2:]
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle) / scale, math.sin(angle) / scale
    theta = torch.tensor(
        [[cosine, -sine * height / width, 0.0], [sine * width / height, cosine, 0.0]],
        dtype=torch.float32,
        device=image.device,
    )[None]
    grid = F.affine_grid(theta, (1, image.shape[0], height, width), align_corners=False)
    # CPU grid_sample does not support half precision; return original dtypes.
    image_work = image if image.dtype in (torch.float32, torch.float64) else image.float()
    image_grid = grid.to(image_work.dtype)
    transformed_image = F.grid_sample(
        image_work[None],
        image_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )[0].to(image.dtype)
    target_work = target if target.dtype in (torch.float32, torch.float64) else target.float()
    stacked = torch.cat([target_work, valid.to(target_work.dtype)], dim=0)
    categorical = F.grid_sample(
        stacked[None],
        grid.to(target_work.dtype),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    channels = target.shape[0]
    transformed_target = categorical[:channels].to(target.dtype)
    transformed_valid = categorical[channels:] > 0.5
    # Make the outside-domain rule explicit, including coordinate edge cases.
    inside = (grid[0].abs() <= 1).all(dim=-1)
    transformed_valid &= inside[None]
    return tuple(x.contiguous() for x in (transformed_image, transformed_target, transformed_valid))


def _uniform(low, high, generator):
    return low + (high - low) * float(torch.rand((), generator=generator, device="cpu"))


def augment(image, target, valid, level="mild", generator=None):
    """Return augmented CHW tensors without modifying the inputs.

    Random draws use the seeded CPU torch RNG, or an explicit CPU Generator.
    DataLoader worker seeding therefore controls this helper without NumPy RNGs.
    Profiles are candidates for validation, not proven accuracy improvements.
    """
    _check_geometry(image, target, valid)
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("Augmentation uses a CPU torch.Generator")
    if level == "basic":
        return tuple(x.contiguous() for x in (image, target, valid))
    if level not in PROFILES:
        raise ValueError(f"Unknown augmentation level {level!r}; choose basic, mild or strong")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Augmentation expects finite image values in [0,1]")
    profile = PROFILES[level]
    if _uniform(0, 1, generator) < profile.affine_probability:
        image, target, valid = warp_joint(
            image,
            target,
            valid,
            angle_degrees=_uniform(-profile.rotation_degrees, profile.rotation_degrees, generator),
            scale=_uniform(profile.min_scale, profile.max_scale, generator),
        )
    if _uniform(0, 1, generator) < profile.photometric_probability:
        brightness = _uniform(-profile.brightness, profile.brightness, generator)
        contrast = _uniform(1 - profile.contrast, 1 + profile.contrast, generator)
        gamma = _uniform(1 - profile.gamma, 1 + profile.gamma, generator)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - mean) * contrast + mean + brightness).clamp(0, 1).pow(gamma)
        channel_factors = torch.tensor(
            [
                _uniform(1 - profile.channel_jitter, 1 + profile.channel_jitter, generator)
                for _ in range(image.shape[0])
            ],
            dtype=image.dtype,
            device=image.device,
        )[:, None, None]
        sigma = _uniform(0, profile.noise_std, generator)
        noise = torch.randn(image.shape, generator=generator, device="cpu", dtype=torch.float32)
        image = image * channel_factors + sigma * noise.to(device=image.device, dtype=image.dtype)
    image = image.clamp(0, 1)
    return tuple(x.contiguous() for x in (image, target, valid))
