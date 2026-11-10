"""CPU numerical/gradient contracts for optional AV capacity, not quality evidence."""

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from deepretina.model import UNet
from deepretina.unified import CHANNELS, UnifiedConfig, build_unified, load_unified, unified_loss


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def original_forward(model, x):
    # Frozen user-owned d83 model.py forward, without a runtime Git dependency.
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w) % model.divisor, 0, (-h) % model.divisor), mode="replicate")
    skips = []
    for i, encoder in enumerate(model.encoders):
        if i:
            x = F.max_pool2d(x, 2)
        x = encoder(x)
        skips.append(x)
    for decoder, skip in zip(model.decoders, reversed(skips[:-1]), strict=True):
        x = decoder(
            torch.cat(
                (
                    F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False),
                    skip,
                ),
                dim=1,
            )
        )
    return model.head(x)[..., :h, :w]


@pytest.mark.parametrize("shape", [(32, 48), (33, 47)])
@pytest.mark.parametrize("norm", ["batch", "group"])
@pytest.mark.parametrize("outputs", [1, 2, 3, 5])
def test_feature_refactor_matches_original_forward_and_buffers(shape, norm, outputs):
    torch.manual_seed(13)
    model = UNet(outputs, base_channels=4, depth=2, norm=norm).train()
    reference = copy.deepcopy(model)
    image = torch.randn(2, 3, *shape)
    actual = model(image)
    expected = original_forward(reference, image)
    assert torch.equal(actual, expected)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], reference.state_dict()[key])


def config(**overrides):
    return replace(
        UnifiedConfig(
            architecture="compact_unet_av_residual",
            base_channels=4,
            depth=2,
            norm="batch",
            lr=0.0003,
            device="cpu",
        ),
        **overrides,
    )


@pytest.mark.parametrize("shape", [(32, 48), (33, 47), (35, 62)])
def test_zero_residual_preserves_backbone_logits_keys_and_rng(shape):
    cfg = config()
    torch.manual_seed(42)
    baseline = build_unified(replace(cfg, architecture="compact_unet"))
    after_baseline = torch.get_rng_state().clone()
    torch.manual_seed(42)
    residual = build_unified(cfg)
    assert torch.equal(after_baseline, torch.get_rng_state())
    base_state, extra_state = baseline.state_dict(), residual.state_dict()
    assert set(extra_state) - set(base_state)
    assert all(key.startswith("av_residual.") for key in set(extra_state) - set(base_state))
    assert set(base_state) <= set(extra_state)
    for key in base_state:
        assert torch.equal(base_state[key], extra_state[key])
    image = torch.randn(2, 3, *shape)
    baseline.train()
    residual.train()
    assert torch.equal(baseline(image), residual(image))
    for key in base_state:
        assert torch.equal(baseline.state_dict()[key], residual.state_dict()[key])
    assert residual.lineage["initialization"] == "scratch"
    assert residual.lineage["historical_exposure"] is False
    assert residual.lineage["independent_model_eligible"] is True


@pytest.mark.parametrize("architecture", ["compact_unet", "compact_unet_av_residual"])
def test_head_stays_before_odd_frame_crop(architecture):
    model = build_unified(config(architecture=architecture)).eval()
    shapes = []
    hook = model.head.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape[-2:]))
    )
    result = model(torch.randn(1, 3, 33, 47))
    hook.remove()
    assert shapes == [(36, 48)] and result.shape == (1, 5, 33, 47)


def test_residual_parameters_cannot_change_direct_vessel_disc_cup_logits():
    model = build_unified(config()).eval()
    image = torch.randn(1, 3, 33, 47)
    original = model(image)
    with torch.no_grad():
        for parameter in model.av_residual.parameters():
            parameter.normal_(0, 0.2)
    modified = model(image)
    assert torch.equal(original[:, [0, 3, 4]], modified[:, [0, 3, 4]])
    assert not torch.equal(original[:, 1:3], modified[:, 1:3])


@pytest.mark.parametrize("support", ["none", "partial", "ambiguous", "known"])
def test_residual_gradients_only_follow_valid_av_and_vessel_condition_is_detached(support):
    model = build_unified(config()).train()
    # Nonzero final weights exercise the whole residual branch, not only its bias.
    with torch.no_grad():
        model.av_residual[-1].weight.normal_(0, 0.05)
    image = torch.randn(2, 3, 33, 47)
    target = torch.zeros(2, 5, 33, 47)
    target[:, 1] = 1
    valid = torch.zeros_like(target, dtype=torch.bool)
    if support == "known":
        valid[:, 1:3] = True
    else:
        valid[:, [0, 3, 4]] = True
        if support == "partial":
            valid[:, 1] = True
        elif support == "ambiguous":
            valid[:, 1:3] = True
            target[:, 2] = 1
    loss, components = unified_loss(model(image), target, valid)
    loss.backward()
    gradients = [p.grad for p in model.av_residual.parameters()]
    if support == "known":
        assert set(components) == {"av"}
        assert any(g is not None and torch.count_nonzero(g) > 0 for g in gradients)
        assert torch.count_nonzero(model.head.weight.grad[[0, 3, 4]]) == 0
        assert torch.count_nonzero(model.head.bias.grad[[0, 3, 4]]) == 0
    else:
        assert "av" not in components
        assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients)


def test_zero_initial_last_layer_receives_first_av_gradient():
    model = build_unified(config()).train()
    image = torch.randn(2, 3, 32, 48)
    target = torch.zeros(2, 5, 32, 48)
    target[:, 1] = 1
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 1:3] = True
    loss, _ = unified_loss(model(image), target, valid)
    loss.backward()
    assert torch.count_nonzero(model.av_residual[-1].bias.grad) > 0
    assert torch.count_nonzero(model.av_residual[0].weight.grad) == 0


def test_safe_checkpoint_roundtrip_keeps_odd_geometry_and_learned_residual(tmp_path):
    cfg = config()
    model = build_unified(cfg).eval()
    with torch.no_grad():
        model.av_residual[-1].weight.normal_(0, 0.05)
    checkpoint = tmp_path / "functionality.pt"
    payload = {
        "format_version": 1,
        "family": "unified",
        "channels": list(CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "lineage": model.lineage,
        "supervised_channels": [True] * 5,
        "purpose": "functionality",
    }
    torch.save(payload, checkpoint)
    restored, metadata, device = load_unified(checkpoint, "cpu")
    image = torch.randn(1, 3, 33, 47)
    assert torch.equal(model(image), restored(image))
    assert metadata["config"]["architecture"] == "compact_unet_av_residual"
    assert restored.architecture_spec["av_residual"]["hidden_channels"] == 8
    assert str(device) == "cpu" and restored.lineage["independent_model_eligible"]


def test_legacy_initialization_is_not_an_eligible_residual_option():
    with pytest.raises(ValueError, match="Legacy initialization requires legacy_universal"):
        build_unified(config(legacy_initialization="teacher.pt", allow_historical_exposure=True))


def test_new_module_changes_training_source_fingerprint(monkeypatch):
    from deepretina import unified

    before = unified._training_source_sha256()
    read = Path.read_bytes

    def changed(self):
        data = read(self)
        return data + b"\n# hypothetical variant\n" if self.name == "unified_av.py" else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert unified._training_source_sha256() != before


def test_parameter_overhead_is_bounded_for_trial_baseline():
    cfg = config(base_channels=12, depth=4)
    baseline = build_unified(replace(cfg, architecture="compact_unet"))
    residual = build_unified(cfg)
    n = sum(p.numel() for p in baseline.parameters())
    extra = sum(p.numel() for p in residual.av_residual.parameters())
    assert extra == 3122 and extra / n < 0.01
    assert sum(p.numel() for p in residual.parameters()) == n + extra


def test_public_training_config_accepts_residual_architecture(tmp_path, monkeypatch, capsys):
    import json

    import deepretina.unified_cli as public
    from deepretina.cli import main

    settings = tmp_path / "residual.json"
    settings.write_text(json.dumps(asdict(config())))
    calls = []

    def capture(manifest, output, cfg, **kwargs):
        calls.append(build_unified(cfg).architecture_spec["name"])
        return output / "best.pt"

    monkeypatch.setattr(public, "train_unified", capture)
    main(
        [
            "train-unified",
            "--manifest",
            "functionality-only.json",
            "--config",
            str(settings),
            "--output",
            str(tmp_path / "mock-output"),
            "--device",
            "cpu",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert calls == ["compact_unet_av_residual"]
    assert result["experimental"] is True


def test_reviewed_hpo_candidate_accepts_residual_without_new_default_plan(tmp_path, monkeypatch):
    import importlib.util
    import json

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "residual_hpo_transport", root / "scripts/unified_hpo.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest = tmp_path / "functionality-only.json"
    manifest.write_text("[]")
    plan = {
        "schema_version": 1,
        "purpose": "full_data_validation_search",
        "execution_code_sha256": helper.code_digest(root),
        "manifest": str(manifest),
        "manifest_sha256": helper.digest(manifest),
        "prepared_content_sha256": "unit-test-only",
        "screen_epochs": 8,
        "total_epochs": 60,
        "candidates": {"residual": asdict(config())},
    }
    path = tmp_path / "mock-plan.json"
    path.write_text(json.dumps(plan))
    calls = []

    def capture(manifest, output, cfg, **kwargs):
        calls.append((build_unified(cfg).architecture_spec["name"], kwargs["stop_after_epoch"]))
        return output / "best.pt"

    monkeypatch.setattr(helper, "train_unified", capture)
    result = helper.run_trial(path, "residual", tmp_path / "mock-output", device="cpu")
    assert calls == [("compact_unet_av_residual", 8)]
    assert result["test_evaluation_performed"] is False
