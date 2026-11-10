"""Unmodified project-owned pre-v2 loss reference, source e92d011."""
import torch
from torch.nn import functional as F
CHANNELS=("vessel","artery","vein","disc","cup")
SEGMENTATION_CHANNELS=(0,3,4)

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
