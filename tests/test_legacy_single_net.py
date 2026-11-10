"""Numerical compatibility with the original architecture and checkpoint format."""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

from deepretina.legacy.checkpoints import convert_single_net, export_training_index, load_single_net
from deepretina.legacy.single_net import Unet, blend_window, tile_corners, tiled_logits


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def original(monkeypatch):
    reference = Path(__file__).parent / "reference/single_net"
    provenance = json.loads((reference / "provenance.json").read_text())
    source_path = reference / "Unet.py"
    content = source_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == provenance["files"]["Unet.py"]["sha256"]
    source = content.decode("utf-8")
    module = types.ModuleType("Unet")
    exec(source, module.__dict__)  # noqa: S102 - pinned original source, golden comparison
    monkeypatch.setitem(sys.modules, "Unet", module)
    return module


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
@pytest.mark.parametrize("dropout", [0, 0.15])
def test_original_module_names_initialization_and_forward_are_identical(original, depth, dropout):
    torch.manual_seed(19)
    reference = original.Unet(filters=4, in_size=3, out_size=3, do=dropout, depth=depth).eval()
    torch.manual_seed(19)
    ported = Unet(filters=4, in_size=3, out_size=3, do=dropout, depth=depth).eval()
    expected, actual = reference.state_dict(), ported.state_dict()
    assert list(expected) == list(actual)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    image = torch.linspace(-0.5, 0.5, 2 * 3 * 32 * 48).reshape(2, 3, 32, 48)
    assert torch.equal(reference(image), ported(image))
    if dropout:
        reference.train()
        ported.train()
        torch.manual_seed(81)
        a = reference(image)
        torch.manual_seed(81)
        b = ported(image)
        assert torch.equal(a, b)


def test_blending_window_and_traversal_match_original_math():
    crop, border = 32, 4
    kernel = np.ones((2 * border + 1, 2 * border + 1))
    kernel /= kernel.sum()
    reference = np.maximum(
        ndimage.convolve(2 * np.ones((crop, crop)), kernel, mode="constant") - 1, 0.01
    )
    np.testing.assert_allclose(blend_window(crop, border), reference, rtol=0, atol=1e-13)
    assert tile_corners(50, 60, 32, 4) == [(0, 0), (0, 28), (18, 0), (18, 28)]


def test_tiled_logits_blends_before_sigmoid_and_covers_edges():
    class Point(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] * 7

    image = torch.linspace(-0.5, 0.5, 50 * 60).reshape(1, 50, 60).repeat(3, 1, 1)
    result = tiled_logits(Point(), image, crop_size=32, border=4)
    np.testing.assert_allclose(result.numpy(), image[:1].numpy() * 7, atol=1e-12)
    assert result.dtype == torch.float64 and result.shape == (1, 50, 60)
    small = tiled_logits(Point(), image[:, :3, :5], crop_size=32, border=4)
    np.testing.assert_allclose(small.numpy(), image[:1, :3, :5].numpy() * 7, atol=1e-12)


def test_trusted_conversion_keeps_exact_outputs_and_exports_original_splits(
    original, tmp_path, monkeypatch
):
    config_module = types.ModuleType("Config")
    Config = type("Config", (), {"__module__": "Config"})
    config_module.Config = Config
    monkeypatch.setitem(sys.modules, "Config", config_module)
    torch.manual_seed(5)
    reference = original.Unet(filters=4, in_size=3, out_size=3, do=0, depth=2).eval()
    reference.config = Config()
    reference.config.method = "test_13"
    reference.config.patch_size = 32
    reference.dataset_dict = {
        "drive_train_na_21": {"database_name": "drive", "split": "train", "masks": ["vessels"]},
        "refuge_test_na_1": {"database_name": "refuge", "split": "test", "masks": ["disk", "cup"]},
    }
    source = tmp_path / "original.pt"
    torch.save(reference, source)
    previous = hashlib.sha256(source.read_bytes()).hexdigest()
    converted = tmp_path / "portable.pt"
    payload = convert_single_net(source, converted)
    model, safe = load_single_net(converted, "cpu")
    assert safe["family"] == "legacy_single_net" and safe["channels"] == ["vessel", "disc", "cup"]
    assert safe["tiling"]["crop_size"] == 32 and safe["tiling"]["border"] == 4
    assert safe["provenance"]["source_sha256"] == previous
    assert safe["provenance"]["normalization_verified"] is False
    assert torch.load(converted, weights_only=True)["training_index"] == reference.dataset_dict
    image = torch.rand(1, 3, 32, 48) - 0.5
    assert torch.equal(reference(image), model(image))
    report = export_training_index(payload, tmp_path / "index.json")
    assert json.loads(report.read_text())["images"] == reference.dataset_dict
    assert previous == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        convert_single_net(source, converted)
    with pytest.raises(FileExistsError):
        export_training_index(payload, report)


def test_converter_requires_unambiguous_single_output_channel(original, tmp_path, monkeypatch):
    config_module = types.ModuleType("Config")
    Config = type("Config", (), {"__module__": "Config"})
    config_module.Config = Config
    monkeypatch.setitem(sys.modules, "Config", config_module)
    model = original.Unet(filters=4, in_size=3, out_size=1, depth=2)
    model.config = Config()
    model.config.method = "unknown"
    model.dataset_dict = {}
    source = tmp_path / "unknown.pt"
    torch.save(model, source)
    with pytest.raises(ValueError, match="Ambiguous"):
        convert_single_net(source, tmp_path / "ambiguous.pt")
    payload = convert_single_net(source, tmp_path / "explicit.pt", channels=["cup"])
    assert payload["channels"] == ["cup"]


def test_vendored_original_tiler_matches_cleaned_implementation(original):
    path = Path(__file__).parent / "reference/single_net/predict_by_parts.py"
    provenance = json.loads(path.with_name("provenance.json").read_text())
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == provenance["files"][path.name]["sha256"]
    original_tiler = types.ModuleType("reference_tiler")
    exec(content.decode("utf-8"), original_tiler.__dict__)  # noqa: S102 - pinned original reference fixture
    torch.manual_seed(17)
    model = original.Unet(filters=4, in_size=3, out_size=3, depth=2).eval()
    image = torch.rand(3, 50, 60) - 0.5
    with torch.inference_mode():
        expected = original_tiler.predict_by_parts(
            model, image, crop_size=32, out_layers=3, border=4, W=None
        )
        actual = tiled_logits(model, image, crop_size=32, border=4)
    assert torch.equal(expected, actual)
