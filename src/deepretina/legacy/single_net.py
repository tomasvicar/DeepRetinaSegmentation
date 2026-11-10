"""Exact original single_net U-Net and logit blending from commit 7a152c3.

The linear channel widths, module names, registered unused BatchNorm layers,
transpose convolutions, activations and historical forward padding order are
retained for checkpoint compatibility. Use tiled_logits for arbitrary image sizes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.ndimage import convolve
from torch import nn
from torch.nn import functional as F


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=1, pad=1, do_batch=1, dov=0):
        super().__init__()
        self.do_batch = do_batch
        self.conv = nn.Conv2d(in_size, out_size, filter_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_size, momentum=0.1)
        self.dov = dov
        if dov > 0:
            self.do = nn.Sequential(nn.Dropout(dov), nn.Dropout2d(dov))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)
        outputs = F.relu(outputs)
        if self.dov > 0:
            outputs = self.do(outputs)
        return outputs


class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=2, pad=1, out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_size, out_size, filter_size, stride=stride, padding=pad, output_padding=out_pad
        )

    def forward(self, inputs):
        return F.relu(self.conv(inputs))


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.up = unetConvT2(in_size, out_size)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        shape1, shape2 = inputs1.shape, inputs2.shape
        # Historical axis order; trained inference uses square, divisible tiles.
        inputs2 = F.pad(inputs2, (0, shape1[-2] - shape2[-2], 0, shape1[-1] - shape2[-1]))
        return torch.cat([inputs1, inputs2], 1)


class Unet(nn.Module):
    """Original constructor signature and state_dict keys, not a new architecture."""

    def __init__(self, filters=64, in_size=3, out_size=1, do=0, depth=5):
        super().__init__()
        if filters < 1 or not 2 <= depth <= 5 or not 0 <= do < 1:
            raise ValueError("Require positive filters, depth2..5 and dropout in [0,1)")
        self.do, self.out_size = do, out_size
        self.filters = (np.arange(depth) + 1) * filters
        widths = self.filters
        for level in range(depth - 1):
            incoming = in_size if level == 0 else int(widths[level - 1])
            width = int(widths[level])
            setattr(
                self,
                f"conv{level + 1}",
                nn.Sequential(
                    unetConv2(incoming, width, dov=0 if level == 0 else do),
                    unetConv2(width, width, dov=do),
                    unetConv2(width, width, dov=do),
                ),
            )
        self.center = nn.Sequential(
            unetConv2(int(widths[-2]), int(widths[-1]), dov=do),
            unetConv2(int(widths[-1]), int(widths[-1]), dov=do),
        )
        for level in reversed(range(1, depth)):
            upper, lower = int(widths[level]), int(widths[level - 1])
            setattr(self, f"up_concat{level}", unetUp(upper, upper))
            setattr(
                self,
                f"up_conv{level}",
                nn.Sequential(
                    unetConv2(lower + upper, lower, dov=do),
                    unetConv2(
                        lower, lower, do_batch=0 if level == 1 else 1, dov=0 if level == 1 else do
                    ),
                ),
            )
        self.final = nn.Conv2d(int(widths[0]), out_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        height, width = inputs.shape[-2:]
        divisor = 2 ** len(self.filters)
        pad_h = (-height) % divisor
        pad_w = (-width) % divisor
        # Keep original forward behavior; callers tile into trained square patches.
        inputs = F.pad(inputs, (0, pad_h, 0, pad_w), mode="reflect")
        inputs = inputs.to(next(self.parameters()).device)
        skips = []
        x = inputs
        for level in range(1, len(self.filters)):
            x = getattr(self, f"conv{level}")(x)
            skips.append(x)
            x = F.max_pool2d(x, 2, 2)
        x = self.center(x)
        for level, skip in zip(reversed(range(1, len(self.filters))), reversed(skips), strict=True):
            x = getattr(self, f"up_concat{level}")(skip, x)
            x = getattr(self, f"up_conv{level}")(x)
        return self.final(x)[..., :height, :width]


LegacyUNet = Unet


@lru_cache(maxsize=8)
def blend_window(crop_size: int, border: int) -> np.ndarray:
    """Cache the exact historical float64 SciPy-convolved logit blending window."""
    if crop_size < 1 or not 0 <= border < crop_size:
        raise ValueError("Require positive crop_size and border in [0,crop_size)")
    weights = 2 * np.ones((crop_size, crop_size), dtype=np.float64)
    kernel = np.ones((2 * border + 1, 2 * border + 1), dtype=np.float64)
    kernel /= kernel.sum()
    weights = convolve(weights, kernel, mode="constant") - 1
    weights[weights < 0.01] = 0.01
    weights.setflags(write=False)
    return weights


def tile_corners(height: int, width: int, crop_size: int, border: int):
    """Retain original interior, right edge, bottom edge, corner traversal order."""
    stride = crop_size - border
    if min(height, width) < crop_size or stride < 1:
        raise ValueError("Pad image to at least crop_size first; border must be smaller")
    rows = list(range(0, height - crop_size, stride))
    cols = list(range(0, width - crop_size, stride))
    return (
        [(row, col) for row in rows for col in cols]
        + [(row, width - crop_size) for row in rows]
        + [(height - crop_size, col) for col in cols]
        + [(height - crop_size, width - crop_size)]
    )


@torch.inference_mode()
def tiled_logits(
    model, image: torch.Tensor, device="cpu", crop_size: int = 256, border: int | None = None
) -> torch.Tensor:
    """Blend logits in float64 at input geometry, then let caller apply sigmoid.

    Input must already be RGB CHW float32/255 -0.5. Small images are padded before
    tiling, extending the original function's domain without changing regular tiles.
    """
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("Expected preprocessed RGB CHW floating tensor")
    border = crop_size // 8 if border is None else border
    window = blend_window(crop_size, border)
    height, width = image.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive")
    pad_h, pad_w = max(0, crop_size - height), max(0, crop_size - width)
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    padded_h, padded_w = image.shape[-2:]
    model = model.to(device).eval()
    summed = None
    weights = np.zeros((padded_h, padded_w), dtype=np.float64)
    for row, col in tile_corners(padded_h, padded_w, crop_size, border):
        patch = image[:, row : row + crop_size, col : col + crop_size]
        logits = model(patch[None].to(device))[0].cpu().numpy()
        if summed is None:
            summed = np.zeros((logits.shape[0], padded_h, padded_w), np.float64)
        summed[:, row : row + crop_size, col : col + crop_size] += logits * window
        weights[row : row + crop_size, col : col + crop_size] += window
    return torch.from_numpy((summed / weights[None])[..., :height, :width].copy())
