"""Reference metrics must preserve unknown labels and missed-vessel errors."""

import importlib.util
from pathlib import Path

import numpy as np


def reference_module():
    path = Path(__file__).parents[1] / "scripts/evaluate_original.py"
    spec = importlib.util.spec_from_file_location("reference_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_cup_predictions_have_no_metric_support():
    module = reference_module()
    target = np.zeros((5, 2, 3), bool)
    target[3, 0, 0] = True
    valid = np.zeros_like(target)
    valid[3] = True  # This photograph has only a disc annotation.
    prediction = target.copy()
    prediction[4] = True
    prediction[0] = True
    counts = module.confusion(prediction, target, valid)
    assert counts[3].tolist() == [1, 0, 0, 6]
    assert not counts[[0, 1, 2, 4]].any()


def test_missed_vessels_count_as_end_to_end_av_false_negatives():
    module = reference_module()
    target = np.zeros((5, 2, 3), bool)
    target[1, 0, 0] = True
    target[2, 0, 1] = True
    valid = np.zeros_like(target)
    valid[1:3, 0, :2] = True
    prediction = np.zeros_like(target)
    counts = module.confusion(prediction, target, valid)
    assert counts[1].tolist() == [0, 0, 1, 2]
    assert counts[2].tolist() == [0, 0, 1, 2]


def test_task_specific_originals_override_only_their_own_heads():
    import torch

    helper = reference_module()

    class Constant(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values

        def forward(self, image):
            value = image.new_tensor(self.values).view(1, -1, 1, 1)
            return value.expand(image.shape[0], -1, *image.shape[-2:])

    def original(values):
        return Constant(values), {"tiling": {"crop_size": 8, "border": 1}}

    masks = helper.segmentation_masks(
        torch.zeros(3, 12, 14),
        original([-2, 2, 2]),
        "cpu",
        vessel=original([2]),
        disc=original([-2]),
    )
    assert masks[0].all()
    assert not masks[3].any()
    assert masks[4].all()  # Original independent cup is not clipped to another disc model.
    assert not masks[1:3].any()
