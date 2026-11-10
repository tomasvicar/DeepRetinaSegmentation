"""Optional v2 vessel emphasis: masked gradients and v1 numerical compatibility."""

import importlib.util
import io
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from deepretina.unified import CHANNELS, UnifiedConfig, build_unified, load_unified, unified_loss

spec = importlib.util.spec_from_file_location(
    "v1_loss", Path(__file__).parent / "reference/unified_v1/loss.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("support", ["vessel", "disc_cup", "av", "all", "none"])
def test_default_loss_and_gradients_bitwise_equal_v1(support):
    generator = torch.Generator().manual_seed(1957)
    target = torch.rand((2, 5, 7, 9), generator=generator)
    valid = torch.rand((2, 5, 7, 9), generator=generator) > 0.3
    channels = {
        "vessel": {0},
        "disc_cup": {3, 4},
        "av": {0, 1, 2},
        "all": set(range(5)),
        "none": set(),
    }[support]
    for c in set(range(5)) - channels:
        valid[:, c] = False
    z = torch.randn((2, 5, 7, 9), generator=generator, requires_grad=True)
    old = z.detach().clone().requires_grad_()
    kwargs = {"bce_weight": 0.7, "dice_weight": 0.3, "av_weight": 1.7}
    loss, parts = unified_loss(z, target, valid, **kwargs)
    expected, old_parts = reference.unified_loss(old, target, valid, **kwargs)
    loss.backward()
    expected.backward()
    assert torch.equal(loss, expected)
    assert torch.equal(z.grad, old.grad)
    assert set(parts) == set(old_parts)
    assert all(torch.equal(parts[k], old_parts[k]) for k in parts)


def example():
    z = torch.zeros((1, 5, 1, 4), requires_grad=True)
    target = torch.zeros_like(z)
    valid = torch.zeros_like(z, dtype=torch.bool)
    target[:, 0, 0, 0] = 1
    valid[:, 0, 0, :2] = True
    return z, target, valid


def test_vessel_multiplier_acts_on_vessel_only_batches_without_cancelling():
    z, t, v = example()
    old = z.detach().clone().requires_grad_()
    base, _ = unified_loss(old, t, v)
    base.backward()
    weighted, parts = unified_loss(z, t, v, vessel_loss_weight=2)
    weighted.backward()
    assert torch.equal(weighted, 2 * base)
    assert torch.equal(z.grad, 2 * old.grad)
    assert torch.equal(parts["vessel"], base)


def test_positive_bce_weight_scales_only_positive_vessel_gradient():
    z, t, v = example()
    old = z.detach().clone().requires_grad_()
    base, _ = unified_loss(old, t, v, dice_weight=0)
    base.backward()
    weighted, _ = unified_loss(z, t, v, dice_weight=0, vessel_pos_weight=3)
    weighted.backward()
    assert z.grad[0, 0, 0, 0] == 3 * old.grad[0, 0, 0, 0]
    assert z.grad[0, 0, 0, 1] == old.grad[0, 0, 0, 1]
    assert torch.count_nonzero(z.grad[~v]) == 0


def test_positive_bce_weight_respects_fractional_targets():
    z, t, v = example()
    v[:] = False
    v[:, 0, 0, 0] = True
    t[:, 0, 0, 0] = 0.25
    loss, _ = unified_loss(z, t, v, dice_weight=0, vessel_pos_weight=3)
    loss.backward()
    assert torch.allclose(loss, torch.tensor(1.5) * torch.log(torch.tensor(2.0)))
    assert z.grad[0, 0, 0, 0] == 0


def test_emphasis_does_not_change_av_disc_cup_gradients_or_unknown_support():
    z, t, v = example()
    v[:, 1:3, 0, :2] = True
    t[:, 1, 0, 0] = 1
    t[:, 2, 0, 1] = 1
    v[:, 3:, 0, :2] = True
    t[:, 3:, 0, 0] = 1
    base = z.detach().clone().requires_grad_()
    original, _ = unified_loss(base, t, v)
    original.backward()
    loss, _ = unified_loss(z, t, v, vessel_loss_weight=2, vessel_pos_weight=3)
    loss.backward()
    assert torch.equal(z.grad[:, 1:], base.grad[:, 1:])
    assert torch.count_nonzero(z.grad[~v]) == 0
    other = z.detach().clone()
    other[~v] = 100
    changed = t.clone()
    changed[~v] = 0.8
    ignored, _ = unified_loss(other, changed, v, vessel_loss_weight=2, vessel_pos_weight=3)
    assert torch.equal(loss, ignored)


@pytest.mark.parametrize("knob", ["vessel_loss_weight", "vessel_pos_weight"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, None, "2"])
def test_invalid_vessel_weights_fail_in_config_and_loss(knob, value):
    with pytest.raises(ValueError, match="finite and positive"):
        UnifiedConfig(**{knob: value}).validate()
    with pytest.raises(ValueError, match="finite and positive"):
        unified_loss(*example(), **{knob: value})


def test_old_checkpoint_without_new_fields_loads_exact_model_and_state_keys():
    cfg = UnifiedConfig(base_channels=2, depth=1, norm="group")
    model = build_unified(cfg).eval()
    old_config = asdict(cfg)
    old_config.pop("vessel_loss_weight")
    old_config.pop("vessel_pos_weight")
    payload = {
        "format_version": 1,
        "family": "unified",
        "channels": list(CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": old_config,
        "state_dict": model.state_dict(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded, saved, _ = load_unified(buffer, device="cpu")
    assert UnifiedConfig(**saved["config"]).vessel_loss_weight == 1
    assert UnifiedConfig(**saved["config"]).vessel_pos_weight == 1
    x = torch.linspace(-0.5, 0.5, 3 * 37 * 41).reshape(1, 3, 37, 41)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    assert model.state_dict().keys() == loaded.state_dict().keys()
