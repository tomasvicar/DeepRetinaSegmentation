"""High-level TTA wiring with generated RGB and tiny CPU forwards; no real weights/GT."""

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina import cli
from deepretina import unified as u
from deepretina.legacy.preprocessing import ImageGeometry


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA"))
    yield
    torch.set_num_threads(before)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    raw = (np.arange(13 * 19 * 3) % 251).astype(np.uint8).reshape(13, 19, 3)
    fov = np.ones(raw.shape[:2], bool)
    fov[:2] = False
    image = tmp_path / "image.png"
    Image.fromarray(raw).save(image)
    config = u.UnifiedConfig(base_channels=2, depth=1, norm="batch", tile_size=32, overlap=8)
    model = u.build_unified(config).eval()
    geometry = ImageGeometry(raw.shape[:2], (0, 12, 0, 18), 1.0, raw.shape[:2])
    prepared = SimpleNamespace(
        image=raw, fov=fov, geometry=geometry, field_of_view_degrees=45, fov_source="synthetic"
    )
    calls = []
    from deepretina.legacy import preprocessing

    def prepare(*args, **kwargs):
        calls.append(kwargs)
        return prepared

    monkeypatch.setattr(preprocessing, "prepare_single_net", prepare)
    payload = {
        "config": asdict(config),
        "supervised_channels": [True] * 5,
        "checkpoint_sha256": "synthetic",
        "purpose": "functionality",
    }
    monkeypatch.setattr(u, "load_unified", lambda *args: (model, payload, "cpu"))
    return image, model, fov, calls


@pytest.mark.parametrize("mode", list(u.TTA_OPERATORS))
def test_high_level_actual_tiny_forward_and_resolved_provenance(
    synthetic, tmp_path, monkeypatch, mode
):
    image, model, fov, prepare_calls = synthetic
    original = u.predict_probabilities
    seen, views = [], []

    def low_level(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "predict_probabilities", low_level)
    handle = model.register_forward_hook(lambda *args: views.append(1))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    output = tmp_path / mode
    result = u.predict_unified("synthetic.pt", image, output, tta=mode, threshold=0.35)
    handle.remove()
    assert seen == ([{}] if mode == "none" else [{"tta": mode}])
    assert len(views) == len(u.TTA_OPERATORS[mode])
    assert len(prepare_calls) == 1
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
    with np.load(result["probabilities"]) as saved:
        probability = saved["probability"]
        assert probability.shape == (5, 13, 19) and probability.dtype == np.float32
        assert np.isfinite(probability).all()
        assert np.all(probability[[0, 3, 4]][:, ~fov] == 0)
        assert np.all(probability[1:3, ~fov] == 0.5)
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["threshold"] == 0.35
    assert metadata.get("tta") == u.tta_protocol(mode).get("tta")


def test_default_none_exact_prior_call_arrays_masks_and_rng(synthetic, tmp_path, monkeypatch):
    image, _, _, _ = synthetic
    original = u.predict_probabilities

    def prior(*args, **kwargs):
        assert not kwargs
        return original(*args)  # Historical high-level arithmetic/call contract.

    monkeypatch.setattr(u, "predict_probabilities", prior)
    rng = torch.get_rng_state().clone()
    default = u.predict_unified("synthetic.pt", image, tmp_path / "default")
    explicit = u.predict_unified("synthetic.pt", image, tmp_path / "explicit", tta="none")
    assert torch.equal(rng, torch.get_rng_state())
    with np.load(default["probabilities"]) as a, np.load(explicit["probabilities"]) as b:
        assert all(np.array_equal(a[key], b[key]) for key in a.files)
    for channel in u.CHANNELS:
        assert Path(default[channel]).read_bytes() == Path(explicit[channel]).read_bytes()
    assert "tta" not in json.loads(Path(default["metadata"]).read_text())


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("mode", list(u.TTA_OPERATORS))
def test_public_cli_alias_and_custom_use_same_highlevel_route(
    tmp_path, monkeypatch, capsys, alias, mode
):
    from deepretina import bundled, unified_cli

    checkpoint = tmp_path / "synthetic.pt"
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"family": "unified"})
    seen = []

    def predict(*args, **kwargs):
        seen.append((args, kwargs))
        return {"metadata": "synthetic.json", **u.tta_protocol(kwargs.get("tta", "none"))}

    monkeypatch.setattr(unified_cli, "predict_unified", predict)
    monkeypatch.setattr(
        bundled, "model_catalog", lambda: {"models": {}, "aliases": {"fixture": "toy"}}
    )

    @contextmanager
    def resolve(name):
        assert name == "fixture"
        yield checkpoint

    monkeypatch.setattr(bundled, "bundled_checkpoint", resolve)
    selection = ["--model", "fixture"] if alias else ["--checkpoint", str(checkpoint)]
    cli.main(
        [
            "predict",
            *selection,
            "--input",
            "generated.png",
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--threshold",
            ".35",
            "--tta",
            mode,
        ]
    )
    assert seen[0][0][0] == checkpoint
    assert seen[0][1]["threshold"] == 0.35
    assert seen[0][1].get("tta", "none") == mode
    assert json.loads(capsys.readouterr().out).get("tta") == u.tta_protocol(mode).get("tta")


@pytest.mark.parametrize("family", ["legacy_single_net", "legacy_av", "pilot_unet"])
def test_nonunified_tta_rejected_without_entering_inference(monkeypatch, family):
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"family": family})
    monkeypatch.setattr(cli, "_apply_legacy", lambda *a: pytest.fail("No legacy prediction"))
    with pytest.raises(ValueError, match="requires a unified"):
        cli._apply_model(SimpleNamespace(tta="rot4"), "synthetic.pt")


def test_invalid_highlevel_mode_fails_before_loading(monkeypatch):
    monkeypatch.setattr(u, "load_unified", lambda *a: pytest.fail("No model load"))
    with pytest.raises(ValueError, match="TTA mode"):
        u.predict_unified("unused", "unused", "unused", tta="bad")


def test_module_predict_cli_forwards_tta(monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setattr(u, "predict_unified", lambda *a, **k: seen.append(k) or {"synthetic": True})
    u.main(
        [
            "predict",
            "--checkpoint",
            "synthetic.pt",
            "--input",
            "synthetic.png",
            "--output",
            str(tmp_path / "out"),
            "--tta",
            "rot4",
            "--threshold",
            ".35",
        ]
    )
    assert seen[0]["tta"] == "rot4" and seen[0]["threshold"] == 0.35
    assert json.loads(capsys.readouterr().out)["synthetic"] is True


def test_public_cli_actual_tiny_prediction_writes_selected_metadata(
    synthetic, tmp_path, monkeypatch, capsys
):
    image, _, _, _ = synthetic
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"family": "unified"})
    cli.main(
        [
            "predict",
            "--checkpoint",
            "synthetic.pt",
            "--input",
            str(image),
            "--output",
            str(tmp_path / "cli-out"),
            "--device",
            "cpu",
            "--tta",
            "rot4",
            "--threshold",
            ".35",
        ]
    )
    paths = json.loads(capsys.readouterr().out)
    metadata = json.loads(Path(paths["metadata"]).read_text())
    assert metadata["tta"] == u.tta_protocol("rot4")["tta"]
    assert metadata["threshold"] == 0.35
    with np.load(paths["probabilities"]) as saved:
        assert saved["probability"].dtype == np.float32
        assert saved["probability"].shape == (5, 13, 19)
