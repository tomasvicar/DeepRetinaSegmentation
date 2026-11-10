"""One-time conversion of the user's trusted original pickled single_net models.

Runtime loading uses weights_only=True. Original class-name mappings are local to
an Unpickler, avoiding Config/Unet module collisions with the separate A/V port.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .single_net import Unet, unetConv2, unetConvT2, unetUp

SOURCE_COMMIT = "7a152c31c2947ddf760716dfa8a6dfc4d2b881c9"


class _Config:
    pass


class _Log:
    pass


class _OriginalUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        mapped = {
            ("Unet", "Unet"): Unet,
            ("Unet", "unetConv2"): unetConv2,
            ("Unet", "unetConvT2"): unetConvT2,
            ("Unet", "unetUp"): unetUp,
            ("Config", "Config"): _Config,
            ("utils.Log", "Log"): _Log,
        }
        if (module, name) in mapped:
            return mapped[module, name]
        return super().find_class(module, name)


_PICKLE = types.ModuleType("deepretina_legacy_single_net_pickle")
_PICKLE.Unpickler = _OriginalUnpickler
_PICKLE.load = pickle.load
_PICKLE.loads = pickle.loads
_PICKLE.dump = pickle.dump
_PICKLE.dumps = pickle.dumps


def _primitive(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported original metadata type: {type(value).__name__}")


def load_trusted_original(source: Path):
    """Explicit migration operation for original user-owned pickle checkpoints only."""
    model = torch.load(source, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    if not isinstance(model, Unet):
        raise TypeError("Expected original Unet.Unet checkpoint")
    return model.cpu().eval()


def _channels(model, instance, channels):
    if channels is not None:
        result = list(channels)
    elif model.out_size == 3:
        result = ["vessel", "disc", "cup"]
    elif model.out_size == 1 and "vessels_seg" in instance.get("method", ""):
        result = ["vessel"]
    elif model.out_size == 1 and "disk" in instance.get("method", ""):
        result = ["disc"]
    else:
        raise ValueError("Ambiguous original output channels; supply channels explicitly")
    if len(result) != model.out_size or len(set(result)) != len(result):
        raise ValueError("Output channel names must be unique and match the original head count")
    return result


def inspect_single_net(source: Path, *, channels=None):
    source = Path(source)
    model = load_trusted_original(source)
    instance = _primitive(vars(model.config)) if hasattr(model, "config") else {}
    training_index = _primitive(getattr(model, "dataset_dict", {}))
    architecture = {
        "filters": int(model.filters[0]),
        "in_size": int(model.conv1[0].conv.in_channels),
        "out_size": int(model.out_size),
        "do": float(model.do),
        "depth": len(model.filters),
    }
    patch = int(instance.get("patch_size", 256))
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    payload = {
        "format_version": 1,
        "family": "legacy_single_net",
        "architecture": architecture,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "channels": _channels(model, instance, channels),
        "preprocessing": {
            "profile": "single_net",
            "normalization": "RGB/255-0.5",
            "pixels_per_degree": 25,
            "normalized": False,
            "normalization_status": "source-inferred: Config.dataset_fname=data_25.hdf5; inherited value was not serialized, raw/normalized history is not independently verified",
        },
        "tiling": {
            "crop_size": patch,
            "border": patch // 8,
            "blend": "logits",
            "window": "2*boxcar_fraction_y*boxcar_fraction_x-1, clipped to0.01",
        },
        "original_config_instance": instance,
        "training_index": training_index,
        "provenance": {
            "source_filename": source.name,
            "source_path": str(source.resolve()),
            "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "legacy_ref": SOURCE_COMMIT,
            "parameters": sum(p.numel() for p in model.parameters()),
            "datasets": dict(Counter(v["database_name"] for v in training_index.values())),
            "splits": dict(Counter(v["split"] for v in training_index.values())),
            "training_images": len(training_index),
            "channel_order_basis": "original evaluate_single_vs_multi.py and checkpoint method/head count",
            "normalization_verified": False,
        },
    }
    return model, payload


def convert_single_net(source: Path, destination: Path, *, channels=None) -> dict:
    """Convert the trusted original to portable tensors and primitive metadata."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserving existing converted model: {destination}")
    _, payload = inspect_single_net(source, channels=channels)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload


def load_single_net(path: Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("family") != "legacy_single_net":
        raise ValueError("Expected converted legacy_single_net format_version1")
    if payload.get("geometry_policy") is not None:
        from ..geometry_policy import validate_geometry_policy

        validate_geometry_policy(payload["geometry_policy"])
    model = Unet(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if len(payload["channels"]) != model.out_size:
        raise ValueError("Output channel metadata mismatch")
    return model.to(device).eval(), payload


def export_training_index(payload: dict, destination: Path):
    """Export exact original image IDs, annotation names and split assignments."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Preserving existing training index: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "family": payload["family"],
        "source_sha256": payload["provenance"]["source_sha256"],
        "source_filename": payload["provenance"]["source_filename"],
        "channels": payload["channels"],
        "training_images": len(payload["training_index"]),
        "datasets": payload["provenance"]["datasets"],
        "splits": payload["provenance"]["splits"],
        "images": payload["training_index"],
    }
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return destination
