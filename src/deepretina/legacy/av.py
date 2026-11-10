"""Numerical port of the original Ves_Class ``Network.Net`` artery/vein model.

Reference: artery_vein_class@7bb0539, Ves_Class/Network.py. The v8/v9 files
serialize this residual U-Net, not the attention network in Network_v9.py.
RGB plus an existing binary vessel segmentation are required. No preprocessing
normalization is added outside the model: its encoder applies the original
per-image/channel sample standardization without an epsilon.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REFERENCE_REF = "7bb0539b90dfce2ed6eaeef1de812072be1155d0"
REFERENCE_SOURCE_SHA256 = "8f99b450f38a3d01dbeb75a110f81ae13a39c0eeb1d4a8f785b38a33eaf8d7a9"
KNOWN_CHECKPOINTS = {
    "b35e9ba997a595330fc1215525804549673c0c6463adc138ecb078206bccabc4": "net_v0_0_8",
    "75c5048aed6adb9f783f8fabb056185c4eb81c23be45b3a152794f59dc8cb1c0": "net_v0_0_9",
}


def center_crop(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Match torchvision CenterCrop, including banker rounding and zero padding."""
    height, width = size
    source_h, source_w = tensor.shape[-2:]
    if width > source_w or height > source_h:
        left = max((width - source_w) // 2, 0)
        top = max((height - source_h) // 2, 0)
        right = max((width - source_w + 1) // 2, 0)
        bottom = max((height - source_h + 1) // 2, 0)
        tensor = F.pad(tensor, (left, right, top, bottom))
        source_h, source_w = tensor.shape[-2:]
    top = round((source_h - height) / 2.0)
    left = round((source_w - width) / 2.0)
    return tensor[..., top : top + height, left : left + width]


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate")
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x = self.conv1(self.BN(x))
        x = self.relu(x)
        residual = x
        x = self.conv2(x)
        return self.relu(x) + residual


class Encoder(nn.Module):
    def __init__(self, channels=(4, 32, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(a, b) for a, b in pairwise(channels)])
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = []
        mean, std = torch.mean(x, (2, 3)), torch.std(x, (2, 3))
        x = (x - mean[:, :, None, None]) / std[:, :, None, None]
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels=(256, 128, 64, 32)):
        super().__init__()
        self.chs = tuple(channels)
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(a, b, 2, 2) for a, b in pairwise(channels)]
        )
        self.dec_blocks = nn.ModuleList([Block(a, b) for a, b in pairwise(channels)])

    def forward(self, x, encoder_features):
        for index in range(len(self.chs) - 1):
            x = self.upconvs[index](x)
            skip = center_crop(encoder_features[index], tuple(x.shape[-2:]))
            x = self.dec_blocks[index](torch.cat([x, skip], dim=1))
        return x


class BottleNeck(nn.Module):
    def __init__(self, channels=(256, 256)):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(channels[0], channels[1], 1, padding=0, padding_mode="replicate")

    def forward(self, x):
        return self.conv1x1_1(x)


class LegacyAVNet(nn.Module):
    def __init__(
        self,
        encoder_channels=(4, 32, 64, 128, 256),
        decoder_channels=(256, 128, 64, 32),
        head_channels=128,
        num_classes=3,
    ):
        super().__init__()
        if encoder_channels[0] != 4 or num_classes != 3:
            raise ValueError("Legacy A/V requires four inputs and three original logits")
        self.architecture = {
            "encoder_channels": list(encoder_channels),
            "decoder_channels": list(decoder_channels),
            "head_channels": int(head_channels),
            "num_classes": int(num_classes),
        }
        self.encoder = Encoder(encoder_channels)
        self.bottleneck = BottleNeck((encoder_channels[-1], encoder_channels[-1]))
        self.decoder = Decoder(decoder_channels)
        self.head1 = nn.Conv2d(decoder_channels[-1], head_channels, 3, padding=1)
        self.head2 = nn.Conv2d(head_channels, num_classes, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 4 or min(x.shape[-2:]) < 16:
            raise ValueError("Expected N,4,H,W input with height and width at least 16")
        features = self.encoder(x)
        bottleneck = self.bottleneck(features[::-1][0])
        out = self.decoder(bottleneck, features[::-1][1:])
        out = self.head2(self.head1(out))
        return F.interpolate(out, (x.size(2), x.size(3)), mode="nearest")


class _CheckpointContainer(nn.Module):
    """Only the original module attributes/state are needed during conversion."""


def _load_original_state(path: Path):
    aliases = [
        (_CheckpointContainer, "Network." + name)
        for name in ("Net", "Block", "Encoder", "Decoder", "BottleNeck")
    ]
    layers = [
        nn.ReLU,
        nn.BatchNorm2d,
        nn.ModuleList,
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.MaxPool2d,
        set,
    ]
    with torch.serialization.safe_globals([*aliases, *layers]):
        original = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(original, _CheckpointContainer):
        raise TypeError("Expected the original saved Network.Net model object")
    state = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
    return state


def _architecture_from_state(state):
    encoder = []
    index = 0
    while (key := f"encoder.enc_blocks.{index}.conv1.weight") in state:
        weight = state[key]
        if index == 0:
            encoder.append(int(weight.shape[1]))
        encoder.append(int(weight.shape[0]))
        index += 1
    decoder = []
    index = 0
    while (key := f"decoder.upconvs.{index}.weight") in state:
        weight = state[key]
        if index == 0:
            decoder.append(int(weight.shape[0]))
        decoder.append(int(weight.shape[1]))
        index += 1
    if not encoder or not decoder or "head1.weight" not in state or "head2.weight" not in state:
        raise ValueError("Checkpoint does not match the original residual A/V architecture")
    return {
        "encoder_channels": encoder,
        "decoder_channels": decoder,
        "head_channels": int(state["head1.weight"].shape[0]),
        "num_classes": int(state["head2.weight"].shape[0]),
    }


def checkpoint_payload(
    model: LegacyAVNet,
    provenance: dict[str, Any] | None = None,
    *,
    preprocessing: dict[str, Any] | None = None,
    geometry_policy: dict[str, Any] | None = None,
):
    payload = {
        "format_version": 1,
        "family": "legacy_av",
        "architecture": model.architecture,
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "channels": ["background_logit_unused_for_classification", "artery", "vein"],
        "input_channels": ["red", "green", "blue", "vessel_mask"],
        "preprocessing": {
            "profile": "ves_class",
            "pixels_per_degree": 25,
            "normalization": "internal per-image/channel torch.mean and sample torch.std(correction1); no epsilon",
            "expects_preprocessed_input": True,
            "input_values": "stored preprocessed RGB and binary vessel channel, cast to float32 without offset/scale",
            "rgb_preprocessing": "MATLAB local_contrast_and_clahe on a 25 px/degree frame; exact raw reproduction requires original DICOM or independent preprocessing goldens",
            "raw_matlab_pipeline_certified": False,
        },
        "tiling": {
            "mode": "full_image",
            "reason": "original normalization/context are full-image; tiling changes predictions",
        },
        "provenance": dict(provenance or {}),
    }
    if preprocessing is not None:
        import copy

        payload["preprocessing"] = copy.deepcopy(preprocessing)
    if geometry_policy is not None:
        from ..geometry_policy import validate_geometry_policy

        validate_geometry_policy(geometry_policy)
        import copy

        payload["geometry_policy"] = copy.deepcopy(geometry_policy)
    return payload


def convert_av(source: Path, destination: Path, *, require_known=True):
    """Convert the owned v8/v9 pickle using only an explicit weights-only allowlist.

    No historical imports, CUDA pickles, or arbitrary pickle globals are executed.
    ``require_known=False`` is an explicit escape for a separately verified owned
    checkpoint with this architecture; it does not relax the pickle allowlist.
    """
    source, destination = Path(source), Path(destination)
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if require_known and sha256 not in KNOWN_CHECKPOINTS:
        raise ValueError(
            "Unrecognized original A/V checkpoint SHA256; expected the owned v8/v9 artifact"
        )
    state = _load_original_state(source)
    architecture = _architecture_from_state(state)
    model = LegacyAVNet(**architecture)
    model.load_state_dict(state, strict=True)
    payload = checkpoint_payload(
        model,
        {
            "source_filename": source.name,
            "source_sha256": sha256,
            "source_size": source.stat().st_size,
            "source_version": KNOWN_CHECKPOINTS.get(sha256, "separately_verified_owned_checkpoint"),
            "source_ref": REFERENCE_REF,
            "source_path": "Ves_Class/Network.py",
            "source_code_sha256": REFERENCE_SOURCE_SHA256,
            "conversion": "torch.load(weights_only=True), explicit Network class/layer aliases; state dictionary only",
            "forward_reference_validation": "not implied by serialization conversion; see independent numeric parity report",
        },
    )
    if destination.exists():
        raise FileExistsError(f"Converted checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def load_av(path: Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("family") != "legacy_av":
        raise ValueError("Expected a safe converted legacy_av checkpoint")
    if payload.get("geometry_policy") is not None:
        from ..geometry_policy import validate_geometry_policy

        validate_geometry_policy(payload["geometry_policy"])
    model = LegacyAVNet(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def prepare_input(rgb, vessel_mask):
    """Stack stored values exactly; no RGB rescale or vessel threshold is added."""
    rgb, vessel_mask = np.asarray(rgb), np.asarray(vessel_mask)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or vessel_mask.shape != rgb.shape[:2]:
        raise ValueError("Expected matching H,W,3 RGB and H,W vessel mask")
    stacked = np.concatenate([rgb.transpose(2, 0, 1), vessel_mask[None]], axis=0).astype(np.float32)
    return torch.from_numpy(stacked[None].copy())


def restore_classes(classes, original_size):
    classes = np.asarray(classes, dtype=np.uint8)
    if original_size is None or tuple(original_size) == classes.shape:
        return classes.copy()
    if len(original_size) != 2 or min(original_size) < 1:
        raise ValueError("original_size must contain positive height,width")
    result = F.interpolate(
        torch.from_numpy(classes.astype(np.float32))[None, None],
        size=tuple(original_size),
        mode="nearest",
    )
    return result[0, 0].numpy().astype(np.uint8)


@torch.inference_mode()
def predict_av_input(model, inputs, device="cpu", original_size=None):
    """Classify one already-preprocessed image, preserving the old full-image path."""
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    if inputs.ndim == 3:
        inputs = inputs[None]
    if inputs.ndim != 4 or inputs.shape[:2] != (1, 4) or min(inputs.shape[-2:]) < 16:
        raise ValueError("Expected [4,H,W] or [1,4,H,W] input with height/width at least 16")
    if not torch.isfinite(inputs).all():
        raise ValueError("Legacy A/V inputs must be finite")
    std = inputs.std(dim=(2, 3))
    if (std == 0).any():
        raise ValueError(
            "Original A/V normalization is undefined for a constant channel, including an empty/full vessel mask"
        )
    vessel_values = torch.unique(inputs[0, 3])
    if (
        len(vessel_values) != 2
        or not bool((vessel_values == 0).any())
        or bool((vessel_values < 0).any())
    ):
        raise ValueError(
            "A/V requires a binary vessel mask with zero background and one positive stored value"
        )
    model = model.to(device).eval()
    logits = model(inputs.to(device))[0].cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Original A/V model produced non-finite logits")
    vessel = inputs[0, 3].cpu().numpy() != 0
    values = logits.numpy()
    classes = np.zeros(vessel.shape, dtype=np.uint8)
    classes[(values[1] > values[2]) & vessel] = 1
    classes[(values[2] > values[1]) & vessel] = 2
    conditional = logits[1:3].softmax(dim=0).numpy()
    return {
        "prepared_logits": values,
        "prepared_conditional_probabilities": conditional,
        "prepared_classes": classes,
        "classes": restore_classes(classes, original_size),
        "prepared_shape": list(classes.shape),
        "output_shape": list(classes.shape if original_size is None else original_size),
    }


def predict_av(model, rgb, vessel_mask, device="cpu", original_size=None):
    return predict_av_input(model, prepare_input(rgb, vessel_mask), device, original_size)
