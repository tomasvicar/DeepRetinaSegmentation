"""Small U-Net initialized from scratch; no external model downloads."""

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = {"vessels": ["vessel"], "av": ["vessel", "artery", "vein"], "disc_cup": ["disc", "cup"]}


def block(in_channels: int, out_channels: int, norm: str = "group") -> nn.Module:
    if norm not in {"group", "batch"}:
        raise ValueError("norm must be group or batch")
    normalize = lambda: (
        nn.GroupNorm(1, out_channels) if norm == "group" else nn.BatchNorm2d(out_channels)
    )
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        normalize(),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        normalize(),
        nn.SiLU(),
    )


class UNet(nn.Module):
    def __init__(
        self, out_channels: int, base_channels: int = 24, depth: int = 3, norm: str = "group"
    ):
        super().__init__()
        if base_channels < 1 or not 1 <= depth <= 5:
            raise ValueError("base_channels must be positive and depth must be between 1 and 5")
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [block(3 if i == 0 else widths[i - 1], w, norm) for i, w in enumerate(widths)]
        )
        self.decoders = nn.ModuleList(
            [block(widths[i + 1] + widths[i], widths[i], norm) for i in reversed(range(depth))]
        )
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        self.divisor = 2**depth

    def forward_features(self, x: torch.Tensor):
        """Decode the padded frame; callers must apply their head before cropping."""
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.divisor, 0, (-h) % self.divisor), mode="replicate")
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1]), strict=True):
            x = decoder(
                torch.cat(
                    (
                        F.interpolate(
                            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ),
                    dim=1,
                )
            )
        return x, (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, (h, w) = self.forward_features(x)
        return self.head(features)[..., :h, :w]
