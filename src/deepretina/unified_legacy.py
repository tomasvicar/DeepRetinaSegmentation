"""Optional original universal topology for the five-output student.

The historical reference implementation is unchanged. This student fixes its
axis-swapped padding at the input boundary, before running divisible geometry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .legacy.single_net import Unet


class LegacyUniversalStudent(Unet):
    """Original width32/depth4 topology with five outputs and safe arbitrary sizes."""

    def __init__(self):
        super().__init__(filters=32, in_size=3, out_size=5, do=0, depth=4)

    def forward(self, inputs):
        h, w = inputs.shape[-2:]
        if min(h, w) < 1:
            raise ValueError("Image dimensions must be positive")
        divisor = 2 ** len(self.filters)
        ph, pw = (-h) % divisor, (-w) % divisor
        if ph or pw:
            mode = "reflect" if ph < h and pw < w else "replicate"
            inputs = F.pad(inputs, (0, pw, 0, ph), mode=mode)
        return super().forward(inputs)[..., :h, :w]


def initialize_legacy_universal(model, checkpoint):
    """Copy the original three heads exactly; initialize conditional AV to neutral.

    Only safe converted, metadata-checked checkpoints are accepted. The caller
    must explicitly authorize uncertain historical exposure before calling this.
    """
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"filters": 32, "in_size": 3, "out_size": 3, "do": 0.0, "depth": 4}
    if (
        payload.get("format_version") != 1
        or payload.get("family") != "legacy_single_net"
        or payload.get("architecture") != expected
        or payload.get("channels") != ["vessel", "disc", "cup"]
    ):
        raise ValueError(
            "Warmstart requires the safe original universal vessel/disc/cup checkpoint"
        )
    preprocessing = payload.get("preprocessing", {})
    if (
        preprocessing.get("normalization") != "RGB/255-0.5"
        or preprocessing.get("pixels_per_degree") != 25
    ):
        raise ValueError("Historical universal preprocessing metadata does not match legacy25")
    state = dict(payload["state_dict"])
    for name in ("final.weight", "final.bias"):
        original = state[name]
        expanded = torch.zeros_like(model.state_dict()[name])
        expanded[[0, 3, 4]] = original
        state[name] = expanded
    model.load_state_dict(state, strict=True)
    index_bytes = json.dumps(payload.get("training_index", {}), sort_keys=True).encode()
    return {
        "initialization": "original_universal_three_to_five",
        "historical_exposure": "unknown",
        "independent_model_eligible": False,
        "accuracy_claim": False,
        "allowed_evaluation_splits": ["train", "val"],
        "warning": "Historical teacher membership may overlap current validation/test; exploratory validation only",
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_provenance": payload.get("provenance", {}),
        "training_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "training_index_records": len(payload.get("training_index", {})),
        "copied_heads": {"vessel": [0, 0], "disc": [1, 3], "cup": [2, 4]},
        "new_heads": "artery/vein weights and biases zero; conditional softmax initially uniform",
    }
