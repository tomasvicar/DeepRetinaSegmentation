"""CPU contracts; no real CUDA, real data, or speed/quality claims."""

import importlib.util
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_unified import make_manifest

from deepretina import unified as u


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("value", [False, True, None, "fp16", "BF16", 1, []])
def test_only_explicit_precision_modes(value):
    with pytest.raises(ValueError, match="mixed_precision"):
        u.UnifiedConfig(mixed_precision=value).validate()


def test_unsupported_cpu_bf16_fails_before_manifest_or_outputs(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="native BF16 CUDA"):
        u.train_unified(
            tmp_path / "missing.json", output, u.UnifiedConfig(device="cpu", mixed_precision="bf16")
        )
    assert not output.exists()


def test_native_support_checked_on_selected_device(monkeypatch):
    selected = Mock(return_value=nullcontext())
    support = Mock(return_value=False)
    monkeypatch.setattr(torch.cuda, "device", selected)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", support)
    with pytest.raises(ValueError, match="native BF16"):
        u._validate_training_precision(
            u.UnifiedConfig(mixed_precision="bf16"), torch.device("cuda:1")
        )
    selected.assert_called_once_with(torch.device("cuda:1"))
    support.assert_called_once_with(including_emulation=False)
    support.return_value = True
    u._validate_training_precision(u.UnifiedConfig(mixed_precision="bf16"), torch.device("cuda:1"))


def test_none_context_never_calls_autocast_or_cuda(monkeypatch):
    monkeypatch.setattr(torch, "autocast", Mock(side_effect=AssertionError("no autocast")))
    monkeypatch.setattr(torch.cuda, "device", Mock(side_effect=AssertionError("no CUDA")))
    cfg = u.UnifiedConfig()
    u._validate_training_precision(cfg, torch.device("cpu"))
    before = torch.get_rng_state().clone()
    with u._precision_context(cfg, "cpu", training=True):
        pass
    assert torch.equal(before, torch.get_rng_state())


def test_bf16_cpu_operator_smoke_keeps_master_weights_gradients_float32():
    # Operator sanity only: production BF16 CPU training is explicitly forbidden.
    torch.manual_seed(42)
    model = torch.nn.Conv2d(3, 5, 3, padding=1)
    x = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 5, 32, 32)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 0] = True
    with u._precision_context(u.UnifiedConfig(mixed_precision="bf16"), "cpu", training=True):
        logits = model(x - 0.5)
        loss, _ = u.unified_loss(logits.float(), target, valid)
    assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss) and model.weight.dtype == torch.float32
    assert model.weight.grad.dtype == torch.float32 and torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.weight.grad[1:]) == 0


def test_bf16_checkpoint_config_loads_on_cpu_and_ignores_outer_autocast(tmp_path):
    cfg = u.UnifiedConfig(mixed_precision="bf16", base_channels=2, depth=1, tile_size=32, overlap=8)
    model = u.build_unified(cfg).eval()
    payload = {
        "config": asdict(cfg),
        "format_version": 1,
        "family": "unified",
        "channels": list(u.CHANNELS),
        "normalization": "RGB/255-0.5",
        "state_dict": model.state_dict(),
    }
    checkpoint = tmp_path / "bf16-metadata-float32-weights.pt"
    torch.save(payload, checkpoint)
    restored, metadata, device = u.load_unified(checkpoint, "cpu")
    assert metadata["config"]["mixed_precision"] == "bf16" and str(device) == "cpu"
    assert all(v.dtype == torch.float32 for v in restored.parameters())
    assert u.inference_config(payload).mixed_precision == "bf16"
    image = torch.rand(3, 32, 32)
    expected = u.predict_probabilities(model, image, "cpu", cfg, 32, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = u.predict_probabilities(model, image, "cpu", cfg, 32, 8)
    np.testing.assert_array_equal(actual, expected)
    with torch.autocast("cpu", dtype=torch.bfloat16):  # noqa: SIM117 - exercise nested contexts
        with u._precision_context(cfg, "cpu", training=False):
            assert not torch.is_autocast_enabled("cpu")
    assert u._normalized_config(cfg) != u._normalized_config(replace(cfg, mixed_precision="none"))


def test_default_functional_training_matches_pinned_original_weights_optimizer_rng(tmp_path):
    code = subprocess.check_output(
        ["git", "show", "0b8762e:src/deepretina/unified.py"],
        cwd=Path(__file__).parents[1],
        text=True,
    )
    name = "deepretina._precision_parity_baseline"
    spec = importlib.util.spec_from_loader(name, loader=None)
    baseline = importlib.util.module_from_spec(spec)
    baseline.__file__ = u.__file__
    sys.modules[name] = baseline
    try:
        exec(compile(code, u.__file__, "exec"), baseline.__dict__)  # noqa: S102 - pinned repository baseline
        manifest = make_manifest(tmp_path)
        settings = dict(  # noqa: C408 - shared keyword fixture
            epochs=1,
            epoch_samples=4,
            batch_size=2,
            patch_size=32,
            base_channels=2,
            depth=1,
            workers=0,
            device="cpu",
            augmentation="basic",
            foreground_probability=0,
            tile_size=32,
            overlap=8,
            max_steps=2,
        )
        for module, destination in [(baseline, "old"), (u, "new")]:
            module.train_unified(manifest, tmp_path / destination, module.UnifiedConfig(**settings))
        old = torch.load(tmp_path / "old/last.pt", weights_only=True)
        new = torch.load(tmp_path / "new/last.pt", weights_only=True)

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                assert a == b

        equal(old["state_dict"], new["state_dict"])
        equal(old["training_state"], new["training_state"])
        assert old["steps"] == new["steps"] == 2
    finally:
        sys.modules.pop(name)


def test_precision_change_cannot_resume_existing_epoch(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path)
    cfg = u.UnifiedConfig(
        epochs=2,
        epoch_samples=4,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        workers=0,
        device="cpu",
        augmentation="basic",
        tile_size=32,
        overlap=8,
    )
    run = tmp_path / "run"
    u.train_unified(manifest, run, cfg, stop_after_epoch=1)
    before = (run / "last.pt").read_bytes()
    # Bypass only device admission to reach the actual saved-config resume guard.
    monkeypatch.setattr(u, "_validate_training_precision", lambda *args: None)
    with pytest.raises(ValueError, match="Resume training configuration changed"):
        u.train_unified(manifest, run, replace(cfg, mixed_precision="bf16"), resume=run / "last.pt")
    assert (run / "last.pt").read_bytes() == before


def test_bf16_loop_cpu_emulation_preserves_fp32_val_and_checkpoint_states(tmp_path, monkeypatch):
    # Only admission is mocked: actual BF16 train/float32 VAL operators run on CPU.
    monkeypatch.setattr(u, "_validate_training_precision", lambda *args: None)
    manifest = make_manifest(tmp_path)
    cfg = u.UnifiedConfig(epochs=1, epoch_samples=4, batch_size=2, patch_size=32,
                          base_channels=2, depth=1, workers=0, device="cpu",
                          augmentation="basic", tile_size=32, overlap=8, mixed_precision="bf16")
    def validate(*args, **kwargs):
        assert not torch.is_autocast_enabled("cpu")
        return u.evaluate_unified_model(*args, **kwargs)
    run = tmp_path / "bf16"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        u.train_unified(manifest, run, cfg, validation_fn=validate)
    saved = torch.load(run / "last.pt", weights_only=True)
    assert saved["config"]["mixed_precision"] == "bf16" and saved["steps"] == 2
    for value in saved["state_dict"].values():
        if value.is_floating_point():
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
    for state in saved["training_state"]["optimizer"]["state"].values():
        for value in state.values():
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
