"""Optional scratch-only residual capacity for conditional artery/vein logits."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import UNet


class AVResidualUNet(UNet):
    """Keep five base logits and add a small zero-initialized A/V correction.

    Backbone state keys are identical to UNet. The side branch consumes decoded
    features, centered input RGB and detached internal vessel probability. It
    never receives an external vessel mask. Vessel/disc/cup logits stay direct.
    """

    def __init__(self, base_channels=12, depth=4, norm="batch"):
        super().__init__(5, base_channels, depth, norm)
        self.av_hidden_channels = max(8, base_channels)
        # Initializing extra capacity must not change the following crop/augment
        # RNG stream relative to the same seeded baseline constructor.
        with torch.random.fork_rng(devices=[]):
            private = torch.Generator().manual_seed((torch.initial_seed() + 104729) % (2**63 - 1))
            torch.set_rng_state(private.get_state())
            hidden = self.av_hidden_channels
            self.av_residual = nn.Sequential(
                nn.Conv2d(base_channels + 4, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, 2, 1),
            )
            nn.init.zeros_(self.av_residual[-1].weight)
            nn.init.zeros_(self.av_residual[-1].bias)

    def forward(self, inputs):
        features, (h, w) = self.forward_features(inputs)
        base = self.head(features)
        rgb = F.pad(
            inputs, (0, features.shape[-1] - w, 0, features.shape[-2] - h), mode="replicate"
        )
        internal_vessel = base[:, 0:1].sigmoid().detach()
        residual = self.av_residual(torch.cat((features, rgb, internal_vessel), dim=1))
        logits = torch.cat((base[:, :1], base[:, 1:3] + residual, base[:, 3:]), dim=1)
        return logits[..., :h, :w]
