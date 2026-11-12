"""Synthetic tensors/scripted logits only: zero trained networks, images or GT files."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from deepretina import unified as u

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experimental_dual", HERE / "dual.py")
dual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)
CFG = {"train_view": "patch", "mixed_precision": "none"}


class FormulaReplay:
    supervised_channels = [True] * 5

    def __init__(self):
        self.inputs = []
        self.evals = 0

    def eval(self):
        self.evals += 1
        return self

    def __call__(self, x):
        self.inputs.append((tuple(x.shape), hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()))
        a, b, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return torch.cat((a * 3, b * 5, -b * 2, c * 4, -c * 3), dim=1)


def image(h, w):
    return torch.linspace(0, 1, 3 * h * w, dtype=torch.float32).reshape(3, h, w)


@pytest.mark.parametrize("h,w", [(17, 29), (515, 33), (1, 513)])
def test_control_bitwise_input_order_shared_calls_and_rng(h, w):
    x = image(h, w)
    before = x.clone()
    rng = torch.random.get_rng_state().clone()
    old = FormulaReplay()
    new = FormulaReplay()
    expected = u.predict_probabilities(old, x, "cpu", CFG, 512, 64, tta="rot4")
    result, trace = dual.dual_probabilities(new, x, "cpu", CFG, u=u)
    assert np.array_equal(result["probability_control"], expected)
    assert old.inputs == new.inputs and old.evals == new.evals == 4
    assert (
        trace["model_calls"]
        == len(old.inputs)
        == 4 * len(u.tile_starts(h, 512, 64)) * len(u.tile_starts(w, 512, 64))
    )
    assert torch.equal(x, before) and torch.equal(rng, torch.random.get_rng_state())
    for p in result.values():
        assert p.dtype == np.float32 and p.shape == (5, h, w) and np.isfinite(p).all()


class ConstantReplay(FormulaReplay):
    def __call__(self, x):
        self.inputs.append(tuple(x.shape))
        return (
            torch.tensor([0.7, 1.0, -1.0, 0.3, -0.4], dtype=torch.float32)[None, :, None, None]
            .expand(1, 5, *x.shape[-2:])
            .clone()
        )


def test_constant_logits_agree_up_to_fp32_weight_rounding():
    p, _ = dual.dual_probabilities(ConstantReplay(), image(515, 33), "cpu", CFG, u=u)
    np.testing.assert_allclose(
        p["probability_control"], p["spatial_logit_alternative"], atol=3e-7, rtol=0
    )


class ViewReplay(FormulaReplay):
    def __call__(self, x):
        n = len(self.inputs)
        self.inputs.append(tuple(x.shape))
        high = n % 2 == 0
        return (
            torch.tensor(
                [6.0 if high else -2.0, 2.0 if high else -2.0, -2.0 if high else 2.0, 0.0, 0.0],
                dtype=torch.float32,
            )[None, :, None, None]
            .expand(1, 5, *x.shape[-2:])
            .clone()
        )


def test_joint_AV_and_probability_not_logit_TTA_across_views():
    p, _ = dual.dual_probabilities(ViewReplay(), image(3, 5), "cpu", CFG, u=u)
    hi = torch.sigmoid(torch.tensor(6.0)).item()
    lo = torch.sigmoid(torch.tensor(-2.0)).item()
    ca = torch.softmax(torch.tensor([2.0, -2.0]), 0)[0].item()
    cb = 1 - ca
    expected_v = (hi + lo) / 2
    expected_a = (hi * ca + lo * cb) / (hi + lo)
    for branch in p.values():
        np.testing.assert_allclose(branch[0], expected_v, atol=2e-7, rtol=0)
        np.testing.assert_allclose(branch[1], expected_a, atol=2e-7, rtol=0)
        assert abs(branch[0, 0, 0] - torch.sigmoid(torch.tensor(2.0)).item()) > 0.2
        assert abs(branch[1, 0, 0] - 0.5) > 0.3


class ZeroVessel(FormulaReplay):
    def __call__(self, x):
        return (
            torch.tensor([-1000.0, 1000.0, -1000.0, 0.0, 0.0], dtype=torch.float32)[
                None, :, None, None
            ]
            .expand(1, 5, *x.shape[-2:])
            .clone()
        )


def test_zero_vessel_finite_conditional_fallback():
    p, _ = dual.dual_probabilities(ZeroVessel(), image(3, 5), "cpu", CFG, u=u)
    for branch in p.values():
        assert np.all(branch[0] == 0) and np.all(branch[1:3] == 0.5) and np.isfinite(branch).all()


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 3, 5),
        torch.zeros(3, 3, 5, dtype=torch.float64),
        torch.full((3, 3, 5), float("nan")),
        torch.full((3, 3, 5), 2.0),
    ],
)
def test_invalid_prepared_image_before_any_fake_call(bad):
    model = FormulaReplay()
    with pytest.raises(ValueError):
        dual.dual_probabilities(model, bad, "cpu", CFG, u=u)
    assert not model.inputs


def test_whole_view_rejected_before_fake_call():
    model = FormulaReplay()
    with pytest.raises(ValueError):
        dual.dual_probabilities(model, image(3, 5), "cpu", {"train_view": "whole"}, u=u)
    assert not model.inputs


def test_wrong_control_reference_refuses_before_dataset_or_model():
    with pytest.raises(ValueError, match="reference bytes"):
        dual.stream_counts(FormulaReplay(), None, "cpu", CFG, {}, b"{}", u=u, counters=None)


def test_reused_counter_matches_frozen_native_partial_labels(monkeypatch):
    # Root stages the byte-exact historical counter beside these synthetic tests.
    spec = importlib.util.spec_from_file_location(
        "frozen_counter",
        Path(u.__file__).resolve().parents[2] / "scripts/calibrate_vascular2_thresholds.py",
    )
    counter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    target = torch.zeros(5, 3, 5)
    target[0, :, 1:4] = 1
    target[1, :, 1:3] = 1
    target[2, :, 3:4] = 1
    target[3, :, 1:4] = 1
    target[4, :, 2:3] = 1
    valid = torch.ones(5, 3, 5, dtype=torch.bool)
    valid[1:3, :, 0] = False
    valid[1:3, :, 4] = False
    valid[4, 0] = False
    rows = [
        {"id": "av", "task": "av", "dataset": "synthetic-av", "split": "val"},
        {"id": "disc", "task": "disc_cup", "dataset": "synthetic-disc", "split": "val"},
    ]
    trace = [("av", "av"), ("disc", "disc_cup")]
    probability, _ = dual.dual_probabilities(FormulaReplay(), image(3, 5), "cpu", CFG, u=u)
    probability = probability["probability_control"]
    photo = counter.photo_counts(
        probability, target, valid, rows, trace, "synthetic-photo", [True] * 5, 0.35
    )
    expected = counter.aggregate([photo], 0.35)

    class Dataset:
        def __init__(self):
            self.groups = [rows]
            self.photo_ids = ["synthetic-photo"]
            self.rows = rows

        def __len__(self):
            return 1

        def sample(self, *a, **kw):
            return image(3, 5), target.clone(), valid.clone(), trace

    monkeypatch.setattr(u, "UnifiedDataset", lambda *a, **kw: Dataset())
    monkeypatch.setattr(u, "predict_probabilities", lambda *a, **kw: probability)
    native = u.evaluate_unified_model(
        FormulaReplay(),
        "UNOPENED",
        split="val",
        device="cpu",
        config=u.UnifiedConfig(),
        threshold=0.35,
    )
    for field in dual.FIELDS:
        assert native[field] == expected[field]
    assert valid[4, 0].sum() == 0  # scorer does not mutate caller validity


class ContextReplay(FormulaReplay):
    def __call__(self, x):
        n = len(self.inputs)
        self.inputs.append(tuple(x.shape))
        values = [6.0, 2.0, -2.0, 4.0, 3.0] if n % 2 == 0 else [-2.0, -2.0, 2.0, -3.0, -4.0]
        return (
            torch.tensor(values, dtype=torch.float32)[None, :, None, None]
            .expand(1, 5, *x.shape[-2:])
            .clone()
        )


def test_hybrid_segmentation_and_conditional_identities_no_added_calls():
    model = ContextReplay()
    result, observed = dual.dual_probabilities(model, image(515, 33), "cpu", CFG, u=u)
    control = result["probability_control"]
    alternative = result["spatial_logit_alternative"]
    hybrid = result["segmentation_logit_control_AV"]
    assert len(result) == 3 and observed["model_calls"] == len(model.inputs) == 8
    assert np.array_equal(hybrid[[0, 3, 4]], alternative[[0, 3, 4]])
    assert np.array_equal(hybrid[1:3], control[1:3])
    assert np.max(np.abs(control[0] - alternative[0])) > 1e-3
    assert np.isfinite(hybrid).all() and np.all((hybrid >= 0) & (hybrid <= 1))
    joint = u._joint_probabilities(hybrid)
    np.testing.assert_allclose(joint[1:3].sum(0), hybrid[0], atol=2e-7, rtol=0)
