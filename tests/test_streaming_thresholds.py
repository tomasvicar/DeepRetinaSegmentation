import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "thresholds", Path(__file__).parents[1] / "scripts/calibrate_vascular2_thresholds.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("threshold", m.THRESHOLDS)
def test_exact_native_decoder_counts_unknown_and_containment(threshold):
    p = np.array(
        [
            [[0.25, 0.5, 0.75], [0.4, 0.6, 0.8]],
            [[0.5, 0.7, 0.3], [0.9, 0.2, 0.6]],
            [[0.5, 0.3, 0.7], [0.1, 0.8, 0.4]],
            [[0.2, 0.5, 0.8], [0.6, 0.4, 0.7]],
            [[0.8, 0.9, 0.4], [0.3, 0.7, 0.8]],
        ],
        np.float32,
    )
    target = torch.tensor(
        [
            [[0, 1, 1], [1, 0, 1]],
            [[0, 1, 0], [1, 0, 1]],
            [[0, 0, 1], [0, 0, 0]],
            [[1, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0]],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1:3, 0, 0] = False
    rows = [
        {"id": "av", "dataset": "test", "task": "av"},
        {"id": "dc", "dataset": "test", "task": "disc_cup"},
    ]
    trace = [("av", "av"), ("dc", "disc_cup")]
    result = m.photo_counts(p, target, valid, rows, trace, "photo", [True] * 5, threshold)
    masks = m.u.decode_probabilities(p, threshold, [True] * 5)
    for i, c in enumerate(m.u.CHANNELS):
        t = target[i].numpy() >= 0.5
        v = valid[i].numpy()
        assert result["channels"][c] == m.u._metrics(
            [
                np.sum(masks[i] & t & v),
                np.sum(masks[i] & ~t & v),
                np.sum(~masks[i] & t & v),
                v.sum(),
            ]
        )
    assert not (masks[4] & ~masks[3]).any()
    assert not (masks[1] & masks[2]).any()
    all_counts = [
        m.photo_counts(p, target, valid, rows, trace, "photo", [True] * 5, t)[
            "av_conditional_counts"
        ]
        for t in m.THRESHOLDS
    ]
    assert len({(c["correct"], c["scored_pixels"]) for c in all_counts}) == 1


def test_same_forward_native_producer_parity():
    p = np.random.default_rng(42).random((5, 8, 9)).astype(np.float32)
    target = torch.from_numpy((p > 0.5).astype(np.float32))
    valid = torch.ones_like(target, dtype=torch.bool)
    rows = [
        {"id": "a", "dataset": "synthetic", "task": "av"},
        {"id": "d", "dataset": "synthetic", "task": "disc_cup"},
    ]
    ds = SimpleNamespace(
        groups=[rows],
        photo_ids=["p"],
        manifest=Path("unused"),
        split="train",
        sample=lambda *args, **kwargs: (
            torch.zeros(3, 8, 9),
            target,
            valid,
            [("a", "av"), ("d", "disc_cup")],
        ),
    )
    model = SimpleNamespace(supervised_channels=[True] * 5, lineage={})
    assert m.native_parity(model, ds, 0, p, m.u.UnifiedConfig(), "cpu")["status"] == "PASS"
