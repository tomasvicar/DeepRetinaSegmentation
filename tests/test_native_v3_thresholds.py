"""Synthetic probability/count tests only; no real NN/GT/data access."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import calibrate_native_v3_thresholds as m


@pytest.fixture
def sample():
    image = torch.zeros(3, 2, 3)
    target = torch.zeros(5, 2, 3)
    target[0, 0, :] = 1
    target[1, 0, :2] = 1
    target[2, 0, 2] = 1
    target[3, :, :] = 1
    target[4, 0, 0] = 1
    valid = torch.ones(5, 2, 3, dtype=torch.bool)
    valid[1:3, 1, :] = False
    p = np.array(
        [
            [[0.2, 0.5, 0.8], [0.3, 0.6, 0.4]],
            [[0.8, 0.3, 0.2], [0.4, 0.7, 0.8]],
            [[0.2, 0.7, 0.8], [0.6, 0.3, 0.2]],
            [[0.2, 0.5, 0.8], [0.3, 0.6, 0.4]],
            [[0.2, 0.5, 0.8], [0.3, 0.6, 0.4]],
        ],
        np.float32,
    )
    rows = [
        {"id": "av0", "dataset": "synthetic", "task": "av"},
        {"id": "dc0", "dataset": "synthetic", "task": "disc_cup"},
    ]
    trace = [("av0", "av"), ("dc0", "disc_cup")]
    return (image, target, valid, trace), p, rows


def test_same_probability_8thresholds_conditional_invariant_and_tradeoff(sample):
    (_, target, valid, trace), p, rows = sample
    result = m.threshold_counts(p, target, valid, rows, trace, "photo", [True] * 5)
    assert tuple(result) == m.THRESHOLDS
    assert (
        len(
            {
                (v["av_conditional_counts"]["correct"], v["av_conditional_counts"]["scored_pixels"])
                for v in result.values()
            }
        )
        == 1
    )
    assert result[0.25]["channels"]["vessel"]["fp"] > result[0.6]["channels"]["vessel"]["fp"]
    for h in m.u.CHANNELS:
        assert (
            len(
                {
                    (
                        v["channels"][h]["valid_pixels"],
                        v["channels"][h]["tp"] + v["channels"][h]["fn"],
                    )
                    for v in result.values()
                }
            )
            == 1
        )
    assert m.aggregate([result[0.5]], 0.5)["checkpoint_sha256"] == m.BEST


@pytest.mark.parametrize("tta", ["none", "rot4", "dihedral8"])
def test_actual_native_counter_same_injected_probability_no_forward(sample, monkeypatch, tta):
    value, p, rows = sample
    monkeypatch.setattr(
        torch.nn.Module, "_call_impl", lambda *a, **k: pytest.fail("No model forward")
    )
    config = m.u.UnifiedConfig(purpose="full_data")
    model = SimpleNamespace(supervised_channels=[True] * 5, unified_config={}, lineage={})
    dataset = SimpleNamespace(groups=[rows], photo_ids=["photo"], manifest=Path("not-opened.json"))
    expected = m.threshold_counts(p, value[1], value[2], rows, value[3], "photo", [True] * 5)[0.5]
    m.native_counter_parity(model, dataset, 0, value, p, config, "cpu", tta, expected)


def test_nonfinite_or_nonFP32_probability_refused(sample):
    value, p, rows = sample
    with pytest.raises(ValueError):
        m.threshold_counts(
            p.astype(np.float64), value[1], value[2], rows, value[3], "p", [True] * 5
        )
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        m.threshold_counts(p, value[1], value[2], rows, value[3], "p", [True] * 5)


def test_false_packet_stops_before_paths_or_forward():
    with pytest.raises(ValueError, match="False packet"):
        m.run({"execution_authorized": False})


def test_worst6_is_one_minimum_not_macro():
    ref = {"channels": {h: {"dice": 0.8} for h in m.u.CHANNELS}, "av_conditional_accuracy": 0.8}
    report = {"channels": {h: {"dice": 0.9} for h in m.u.CHANNELS}, "av_conditional_accuracy": 0.7}
    value, ratios = m.worst6(report, ref)
    assert len(ratios) == 6 and value == pytest.approx(0.7 / 0.8)
