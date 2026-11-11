"""Tiny CPU synthetic forward only; no actual TRAIN/VAL/TEST arrays or weights."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "vascular2_bn_research", Path(__file__).parents[1] / "scripts/recalibrate_vascular2_bn.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(1)


def model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(), torch.nn.Conv2d(4, 5, 1)
    ).eval()


def test_actual_update_bn_only_changes_allowed_buffers_and_restores_modes_momenta():
    net = model()
    net[1].running_mean.fill_(90)
    net[1].momentum = 0.27
    before = copy.deepcopy(net.state_dict())
    net[3].train()
    batches = [torch.ones(4, 3, 8, 8) * i for i in [1.0, 2.0, 3.0]]
    report = m.recalibrate(net, batches)
    assert set(report["changed_bn_keys"]) <= m.bn_keys(net)
    for key, v in net.state_dict().items():
        if key not in m.bn_keys(net):
            assert m.tensor_bytes(v) == m.tensor_bytes(before[key])
    assert report["non_bn_state_bitidentical"] and report["gradient_updates"] == 0
    assert net[1].num_batches_tracked.item() == 3
    assert net.training is False and net[3].training is True and net[1].momentum == 0.27
    assert all(p.grad is None for p in net.parameters())


def test_update_is_cumulative_not_original_momentum_ema():
    bn = torch.nn.BatchNorm2d(1, momentum=0.99).eval()
    m.recalibrate(bn, [torch.zeros(4, 1, 3, 3), torch.ones(4, 1, 3, 3) * 4])
    assert bn.running_mean.item() == 2
    assert bn.num_batches_tracked.item() == 2
    assert bn.momentum == 0.99


def test_centered_batches_count_parent_delivered_indices_only():
    loader = [(torch.ones(4, 3, 4, 4), None, None, torch.tensor([0, 1, 0, 2]), None)]
    batches = m.CenteredTrainingBatches(loader, ["a", "b", "c"])
    images = list(batches)
    assert torch.equal(images[0], torch.full((4, 3, 4, 4), 0.5))
    assert batches.delivered == 4 and batches.visits == {"a": 2, "b": 1, "c": 1}


def test_failure_restores_bn_momentum_and_mode():
    bn = torch.nn.BatchNorm2d(1, momentum=0.13).eval()

    def fail():
        yield torch.ones(4, 1, 3, 3)
        raise RuntimeError("synthetic input failure")

    with pytest.raises(RuntimeError, match="input failure"):
        m.recalibrate(bn, fail())
    assert bn.momentum == 0.13 and bn.training is False


def test_non_bn_side_effect_rejected():
    class Invalid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(1)
            self.register_buffer("unrelated", torch.tensor(0))

        def forward(self, x):
            self.unrelated.add_(1)
            return self.bn(x)

    with pytest.raises(ValueError, match="Non-BN"):
        m.recalibrate(Invalid(), [torch.ones(4, 1, 3, 3)])


def test_derived_payload_is_prediction_only_and_does_not_rewrite_parent_history():
    net = model()
    parent = {
        "family": "unified",
        "format_version": 1,
        "lineage": {"initialization": "scratch"},
        "epoch": 60,
        "steps": 122880,
        "state_dict": copy.deepcopy(net.state_dict()),
        "config": {"augmentation": "mild"},
        "validation_metrics": {"fake": 1},
        "val_selection_score": 0.8,
        "val_mean_dice": 0.7,
        "val_macro_dataset_mean_dice": 0.8,
        "training_state": {"optimizer": "parent only"},
        "checkpoint_sha256": "parent",
        "coverage": {"parent_annotation_count": 5700},
    }
    original = copy.deepcopy(parent)
    m.recalibrate(net, [torch.ones(4, 3, 8, 8)])
    output = m.derived_payload(parent, net, parent_sha="p" * 64, report={"gradient_updates": 0})
    assert output["epoch"] == 60 and output["steps"] == 122880
    assert output["config"] == parent["config"] == {"augmentation": "mild"}
    assert not any(k.startswith("val_") for k in output)
    assert "validation_metrics" not in output and "training_state" not in output
    assert output["lineage"]["predict_only"] and not output["lineage"]["training_resume_eligible"]
    assert output["lineage"]["calibration_gradient_updates"] == 0
    assert parent["validation_metrics"] == original["validation_metrics"]
    for key, v in parent["state_dict"].items():
        assert m.tensor_bytes(v) == m.tensor_bytes(original["state_dict"][key])


def test_same_seed_synthetic_recipe_repeats_exact_bn_state():
    a = model()
    b = copy.deepcopy(a)
    batches = [
        torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(i)) for i in range(3)
    ]
    m.recalibrate(a, batches)
    m.recalibrate(b, batches)
    assert all(
        m.tensor_bytes(a.state_dict()[k]) == m.tensor_bytes(b.state_dict()[k])
        for k in a.state_dict()
    )


def test_modified_same_version_update_bn_cannot_silently_run(monkeypatch):
    monkeypatch.setattr(m, "UPDATER_SHA", "0" * 64)
    with pytest.raises(ValueError, match="implementation"):
        m.recalibrate(model(), [])


def test_tiny_derived_checkpoint_safe_roundtrip_matches_predictions(tmp_path):
    import io
    from dataclasses import asdict

    from deepretina.unified import UnifiedConfig, build_unified, load_unified

    cfg = UnifiedConfig(base_channels=2, depth=2, norm="batch", purpose="functionality")
    net = build_unified(cfg).eval()
    parent = {
        "family": "unified",
        "format_version": 1,
        "channels": ["vessel", "artery", "vein", "disc", "cup"],
        "normalization": "RGB/255-0.5",
        "config": asdict(cfg),
        "lineage": net.lineage,
        "epoch": 60,
        "steps": 122880,
        "state_dict": copy.deepcopy(net.state_dict()),
        "validation_metrics": {"stale": True},
    }
    report = m.recalibrate(net, [torch.randn(4, 3, 32, 32)])
    result = m.derived_payload(parent, net, parent_sha="p" * 64, report=report)
    data = io.BytesIO()
    torch.save(result, data)
    loaded, payload, _ = load_unified(io.BytesIO(data.getvalue()), device="cpu")
    image = torch.randn(1, 3, 33, 35)
    net.eval()
    with torch.no_grad():
        assert torch.equal(net(image), loaded(image))
    assert payload["lineage"]["predict_only"] and "validation_metrics" not in payload
