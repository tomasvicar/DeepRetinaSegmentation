"""Synthetic CPU tile-batch contracts; no real checkpoint/image/GT/GPU benchmark."""

import ast
import importlib.util
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from deepretina import unified as u


class Coordinates(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image):
        self.calls.append(tuple(image.shape))
        y = torch.linspace(-0.7, 0.7, image.shape[-2], device=image.device)[None, None, :, None]
        x = torch.linspace(-0.9, 0.9, image.shape[-1], device=image.device)[None, None, None, :]
        return torch.cat(
            (
                image[:, :1] + x + y,
                image[:, 1:2] + x,
                image[:, 2:3] - y,
                image[:, :1] - x,
                image[:, 2:3] + y,
            ),
            1,
        )


@pytest.fixture
def frozen():
    code = subprocess.check_output(
        ["git", "show", "1f723ad:src/deepretina/unified.py"],
        cwd=Path(__file__).parents[1],
        text=True,
    )
    name = "deepretina._tile_batch_reference1f"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = u.__file__
    sys.modules[name] = module
    module.reference_code = code
    exec(compile(code, u.__file__, "exec"), module.__dict__)  # noqa: S102 - immutable git reference
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize("tta", u.TTA_OPERATORS)
@pytest.mark.parametrize("view", ["patch", "whole"])
def test_batch1_exact_frozen_path_schema_and_rng(frozen, tta, view):
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    cfg = asdict(u.UnifiedConfig(train_view=view, view_size=32))
    state = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    expected = frozen.predict_probabilities(Coordinates(), image, "cpu", cfg, 32, 8, tta=tta)
    actual = u.predict_probabilities(
        Coordinates(), image, "cpu", cfg, 32, 8, tta=tta, tile_batch_size=1
    )
    assert np.array_equal(actual, expected) and u.tile_batch_protocol(1) == {}
    assert torch.equal(state[0], torch.get_rng_state()) and state[1] == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(state[2], np.random.get_state(), strict=True))
    old = next(
        n
        for n in ast.parse(frozen.reference_code).body
        if isinstance(n, ast.FunctionDef) and n.name == "_predict_probabilities"
    )
    current = next(
        n
        for n in ast.parse(Path(u.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_predict_probabilities"
    )
    assert ast.dump(old, include_attributes=False) == ast.dump(current, include_attributes=False)


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("shape", [(35, 49), (17, 71), (73, 19), (1, 3)])
@pytest.mark.parametrize("tta", ["none", "rot4", "dihedral8"])
def test_coordinate_sensitive_odd_rectangles_preserve_tiling_and_inverse(batch, shape, tta):
    h, w = shape
    image = torch.linspace(0, 1, 3 * h * w).reshape(3, h, w)
    reference = Coordinates()
    model = Coordinates()
    expected = u.predict_probabilities(reference, image, "cpu", tile_size=32, overlap=8, tta=tta)
    actual = u.predict_probabilities(
        model, image, "cpu", tile_size=32, overlap=8, tta=tta, tile_batch_size=batch
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
    assert actual.shape == (5, h, w) and actual.dtype == np.float32 and np.isfinite(actual).all()
    assert max(c[0] for c in model.calls) <= batch
    assert len(model.calls) <= len(reference.calls)
    np.testing.assert_allclose(actual[1:3].sum(0), 1, atol=1e-6)


@pytest.mark.parametrize("norm", ["batch", "group"])
@pytest.mark.parametrize("batch", [2, 4])
def test_native_eval_BN_GN_padding_state_and_BF16_checkpoint_inference_FP32(tmp_path, norm, batch):
    cfg = u.UnifiedConfig(
        base_channels=2, depth=2, norm=norm, mixed_precision="bf16", tile_size=32, overlap=8
    )
    model = u.build_unified(cfg).eval()
    path = tmp_path / "synthetic-bf16.pt"
    torch.save(
        {
            "format_version": 1,
            "family": "unified",
            "channels": list(u.CHANNELS),
            "normalization": "RGB/255-0.5",
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
        },
        path,
    )
    model, payload, device = u.load_unified(path, "cpu")
    image = torch.linspace(0, 1, 3 * 17 * 71).reshape(3, 17, 71)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    expected = u.predict_probabilities(model, image, device, cfg, 32, 8, tta="rot4")
    actual = u.predict_probabilities(
        model, image, device, cfg, 32, 8, tta="rot4", tile_batch_size=batch
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        nested = u.predict_probabilities(
            model, image, device, cfg, 32, 8, tta="rot4", tile_batch_size=batch
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-7)
    assert np.array_equal(actual, nested) and actual.dtype == np.float32
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert payload["config"]["mixed_precision"] == "bf16" and not torch.cuda.is_initialized()


@pytest.mark.parametrize("batch", [True, False, 0, -1, 8, 1.5, float("nan"), "2", None])
def test_invalid_batch_rejected_before_model(batch):
    with pytest.raises(ValueError, match="tile_batch_size"):
        u.predict_probabilities(None, None, "cpu", tile_batch_size=batch)


def test_whole_view_batching_explicitly_unsupported_before_model():
    with pytest.raises(ValueError, match="whole-view"):
        u.predict_probabilities(
            None, None, "cpu", u.UnifiedConfig(train_view="whole"), tile_batch_size=2
        )


def test_overlap_averages_joint_av_weighted_by_vessel_not_conditional():
    class Confidence(torch.nn.Module):
        def forward(self, image):
            left = image[:, :1].mean((1, 2, 3)) < 0
            vessel = torch.where(left, 0.9, 0.1)
            artery = torch.where(left, 0.8, 0.1)
            logits = torch.stack(
                (
                    torch.logit(vessel),
                    artery.log(),
                    (1 - artery).log(),
                    torch.zeros_like(vessel),
                    torch.zeros_like(vessel),
                ),
                1,
            )
            return logits[:, :, None, None].expand(-1, -1, *image.shape[-2:])

    image = torch.linspace(0, 1, 49)[None, None, :].expand(3, 32, 49)
    p = u.predict_probabilities(
        Confidence(), image, "cpu", tile_size=32, overlap=15, tile_batch_size=2
    )
    assert p[0, 16, 24] == pytest.approx(0.5, abs=1e-6)
    assert p[1, 16, 24] == pytest.approx(0.73, abs=1e-6) and p[1, 16, 24] != pytest.approx(0.45)


def test_different_edge_shapes_flush_without_padding_or_reordering(monkeypatch):
    monkeypatch.setattr(u, "tile_starts", lambda length, tile, overlap: list(range(0, length, 16)))
    image = torch.linspace(0, 1, 3 * 45 * 49).reshape(3, 45, 49)
    expected = u.predict_probabilities(Coordinates(), image, "cpu", tile_size=32, overlap=8)
    model = Coordinates()
    actual = u.predict_probabilities(
        model, image, "cpu", tile_size=32, overlap=8, tile_batch_size=4
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-7)
    assert len({shape[-2:] for shape in model.calls}) > 1
    assert max(shape[0] for shape in model.calls) <= 4
