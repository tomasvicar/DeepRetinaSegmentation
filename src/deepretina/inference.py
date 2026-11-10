"""Overlapping tile inference with explicit probability and mask outputs.

Independent output probabilities are retained unchanged. Exported binary masks enforce
cup within disc and artery/vein within vessels; AV ties are resolved by probability.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .dataset import letterbox_geometry, whole_view
from .model import CHANNELS, UNet


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_checkpoint(checkpoint: Path, device: str = "auto"):
    device = resolve_device(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError(
            "Unsupported checkpoint format; legacy pickle models require explicit migration"
        )
    config = saved["config"]
    model = UNet(
        len(CHANNELS[config["task"]]),
        config["base_channels"],
        config["depth"],
        config.get("norm", "group"),
    )
    model.train_view = config.get("train_view", "patch")
    model.view_size = config.get("view_size", 512)
    model.load_state_dict(saved["model"])
    model.supervised_channels = saved.get(
        "supervised_channels", [True] * len(CHANNELS[config["task"]])
    )
    if len(model.supervised_channels) != len(CHANNELS[config["task"]]):
        raise ValueError("Invalid checkpoint channel support metadata")
    return model.to(device).eval(), saved, device


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    last = max(0, length - tile)
    return sorted({*range(0, last + 1, tile - overlap), last})


@torch.inference_mode()
def tiled_probabilities(
    model, image: torch.Tensor, device, tile_size: int = 512, overlap: int = 64
) -> np.ndarray:
    """Return C,H,W float32 probabilities at input geometry; bound GPU memory by tile."""
    if tile_size < 8 or not 0 <= overlap < tile_size:
        raise ValueError("tile_size must be >=8 and overlap in [0,tile_size)")
    model.eval()
    h, w = image.shape[-2:]
    result = None
    weights = np.zeros((h, w), np.float32)
    for y in tile_starts(h, tile_size, overlap):
        for x in tile_starts(w, tile_size, overlap):
            tile = image[:, y : y + tile_size, x : x + tile_size]
            probability = model((tile[None].to(device) - 0.5) / 0.5).sigmoid()[0].cpu().numpy()
            th, tw = probability.shape[-2:]
            # Positive edge weights preserve coverage even on single-pixel edges.
            wy = np.maximum(np.hanning(th), 0.05) if th > 1 else np.ones(1)
            wx = np.maximum(np.hanning(tw), 0.05) if tw > 1 else np.ones(1)
            weight = (wy[:, None] * wx[None]).astype(np.float32)
            if result is None:
                result = np.zeros((len(probability), h, w), np.float32)
            result[:, y : y + th, x : x + tw] += probability * weight
            weights[y : y + th, x : x + tw] += weight
    return result / weights[None]


@torch.inference_mode()
def whole_probabilities(model, image, device, view_size=512):
    model.eval()
    height, width = image.shape[-2:]
    resized_h, resized_w, top, left = letterbox_geometry(height, width, view_size)
    prepared, _, _ = whole_view(image, size=view_size)
    probability = model((prepared[None].to(device) - 0.5) / 0.5).sigmoid()
    probability = probability[..., top : top + resized_h, left : left + resized_w]
    return (
        F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)[0]
        .cpu()
        .numpy()
    )


def probabilities_for_config(model, image, device, config=None, tile_size=512, overlap=64):
    """Whole-photo or tiled inference at prepared geometry, using recorded policy."""
    if config is None:
        view = getattr(model, "train_view", "patch")
        size = getattr(model, "view_size", 512)
    elif isinstance(config, dict):
        view, size = config.get("train_view", "patch"), config.get("view_size", 512)
    else:
        view, size = config.train_view, config.view_size
    model.eval()
    if view == "whole":
        return whole_probabilities(model, image, device, size)
    if view != "patch":
        raise ValueError(f"Unknown training view {view!r}")
    return tiled_probabilities(model, image, device, tile_size, overlap)


def binary_masks(probability: np.ndarray, task: str, threshold: float = 0.5) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    mask = probability >= threshold
    if task == "disc_cup":
        mask[1] &= mask[0]
    elif task == "av":
        artery_wins = probability[1] >= probability[2]
        mask[1] &= mask[0] & artery_wins
        mask[2] &= mask[0] & ~artery_wins
    return mask


def predict(
    checkpoint: Path,
    image_path: Path,
    output: Path,
    device: str = "auto",
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.5,
) -> dict:
    model, saved, device = load_checkpoint(checkpoint, device)
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    inference_max_size = saved.get("inference_max_size")
    h, w = image.shape[:2]
    scale = min(1.0, inference_max_size / max(h, w)) if inference_max_size else 1.0
    inference_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    prepared = np.asarray(Image.fromarray(image).resize(inference_size, Image.Resampling.LANCZOS))
    tensor = torch.from_numpy(prepared.transpose(2, 0, 1).copy()).float() / 255
    probability = probabilities_for_config(
        model, tensor, device, tile_size=tile_size, overlap=overlap
    )
    if probability.shape[-2:] != (h, w):
        probability = F.interpolate(
            torch.from_numpy(probability)[None], size=(h, w), mode="bilinear", align_corners=False
        )[0].numpy()
    task = saved["config"]["task"]
    masks = binary_masks(probability, task, threshold)
    supported = np.asarray(model.supervised_channels, dtype=bool)
    masks[~supported] = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "probabilities.npz",
        probability=probability,
        channels=np.asarray(CHANNELS[task]),
        supervised_channels=supported,
    )
    color = np.zeros_like(image)
    palette = {
        "vessels": [(50, 220, 80)],
        "av": [(60, 190, 60), (255, 50, 50), (50, 100, 255)],
        "disc_cup": [(255, 190, 40), (50, 220, 255)],
    }[task]
    paths = {"probabilities": str(output / "probabilities.npz")}
    for name, mask, rgb in zip(CHANNELS[task], masks, palette, strict=True):
        path = output / f"{name}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        color[mask] = rgb
        paths[name] = str(path)
    active = masks.any(axis=0)
    overlay = image.copy()
    overlay[active] = np.round(0.55 * image[active] + 0.45 * color[active]).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.png")
    paths["overlay"] = str(output / "overlay.png")
    metadata = {
        "task": task,
        "channels": CHANNELS[task],
        "supervised_channels": supported.tolist(),
        "unsupported_channels": [
            name for name, keep in zip(CHANNELS[task], supported, strict=True) if not keep
        ],
        "unsupported_mask_policy": "all-zero mask placeholders; raw probabilities untrained and invalid for interpretation",
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "shape": list(image.shape[:2]),
        "inference_shape": list(prepared.shape[:2]),
        "inference_max_size": inference_max_size,
        "train_view": model.train_view,
        "view_size": model.view_size,
        "whole_view_policy": "symmetric zero letterbox; RGB bilinear align_corners=False; restore probabilities before threshold"
        if model.train_view == "whole"
        else None,
        "resampling": "LANCZOS RGB downsample; bilinear probability restoration before threshold",
        "tile_size": tile_size,
        "overlap": overlap,
        "outputs": paths,
    }
    (output / "prediction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths
