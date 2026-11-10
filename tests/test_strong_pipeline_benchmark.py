"""Ensure latency comparison uses the selected dedicated reference components."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def benchmark_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "strong_pipeline_benchmark", scripts / "benchmark_strong_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference_files(tmp_path, benchmark_module):
    paths = {}
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    report = {
        "family": "original_reference",
        "split": "val",
        "preview_only": False,
        "single_net_normalized": False,
        "av_preprocessing_profile": "vesclass_matlab_provided_fov_calibrated",
        "disc_reference_variant": "disc0/raw",
    }
    for name, key in keys.items():
        path = tmp_path / (name + ".pt")
        path.write_bytes(name.encode())
        paths[name] = path
        report[key] = benchmark_module.sha256(path)
    return paths, report


def test_exact_selected_components_required(benchmark_module, reference_files):
    paths, report = reference_files
    benchmark_module.validate_reference(report, paths)
    paths["disc"].write_bytes(b"a different disc checkpoint")
    with pytest.raises(ValueError, match="disc weights differ"):
        benchmark_module.validate_reference(report, paths)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"preview_only": True},
        {"av_preprocessing_profile": "proxy"},
        {"disc_reference_variant": "universal/raw"},
        {"single_net_normalized": True},
    ],
)
def test_weaker_or_different_reference_is_rejected(benchmark_module, reference_files, change):
    paths, report = reference_files
    with pytest.raises(ValueError, match="full calibrated"):
        benchmark_module.validate_reference({**report, **change}, paths)


def test_four_network_pipeline_uses_dedicated_heads_native_tiles_and_calibrated_av(
    benchmark_module, monkeypatch
):
    m = benchmark_module
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    fov = np.ones((2, 3), dtype=bool)

    class Geometry:
        def restore(self, values, order):
            assert order == 0
            return values

    prepared = SimpleNamespace(tensor=lambda: torch.zeros(3, 2, 3), geometry=Geometry())
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    vessel = torch.tensor([[[0.0, 1.0, -1.0], [1.0, -1.0, 0.0]]])
    disc = torch.ones(1, 2, 3)
    universal = -torch.ones(3, 2, 3)
    universal[2, 1, 2] = 1
    specs = {
        "vessel": (vessel, 320, 40),
        "disc": (disc, 192, 24),
        "universal": (universal, 288, 36),
    }
    calls = []

    def tiled(model, image, device, crop_size, border):
        name = model
        assert (crop_size, border) == specs[name][1:]
        calls.append(name)
        return specs[name][0]

    monkeypatch.setattr(m, "tiled_logits", tiled)

    def calibrated(raw, predicted_vessel, *, fov, **kwargs):
        assert raw is rgb
        np.testing.assert_array_equal(predicted_vessel, vessel[0].numpy() > 0)
        return SimpleNamespace(inputs=lambda: "calibrated inputs")

    monkeypatch.setattr(m, "prepare_vesclass_matlab_calibrated", calibrated)
    classes = np.array([[0, 1, 0], [2, 0, 0]], np.uint8)

    def av(model, inputs, *, device, original_size):
        assert inputs == "calibrated inputs" and original_size == (2, 3)
        return {"classes": classes}

    monkeypatch.setattr(m, "predict_av_input", av)
    models = {
        name: (name, {"tiling": {"crop_size": spec[1], "border": spec[2]}})
        for name, spec in specs.items()
    }
    result = m.original_operation(rgb, fov, models, "av", torch.device("cpu"), {})
    assert calls == ["vessel", "disc", "universal"]
    np.testing.assert_array_equal(result[0], vessel[0] > 0)
    np.testing.assert_array_equal(result[1], classes == 1)
    np.testing.assert_array_equal(result[2], classes == 2)
    np.testing.assert_array_equal(result[3], disc[0] > 0)
    np.testing.assert_array_equal(result[4], universal[2] > 0)


def test_nonfinite_original_predictions_fail(benchmark_module, monkeypatch):
    m = benchmark_module
    prepared = SimpleNamespace(tensor=lambda: torch.zeros(3, 2, 3))
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    monkeypatch.setattr(m, "tiled_logits", lambda *a, **kw: torch.full((1, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="Nonfinite"):
        m.original_operation(None, None, {"vessel": ("vessel", {"tiling": {}})}, None, "cpu", {})


def test_unified_uses_checkpoint_geometry_and_masks_original_frame(benchmark_module, monkeypatch):
    m = benchmark_module
    fov = np.array([[False, True, True], [True, True, False]])

    class Geometry:
        def restore(self, value, order):
            return value

    prepared = SimpleNamespace(
        image=np.zeros((2, 3, 3), dtype=np.uint8), fov=fov, geometry=Geometry()
    )
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    cfg = SimpleNamespace(tile_size=96, overlap=16)

    def predict(model, image, device, config, tile, overlap):
        assert config is cfg and (tile, overlap) == (96, 16)
        probability = np.full((5, 2, 3), 0.9, np.float32)
        probability[2] = 0.1
        probability[3, 0, 1] = 0.1
        return probability

    monkeypatch.setattr(m, "predict_probabilities", predict)
    monkeypatch.setattr(m, "restore_probabilities", lambda p, g: p)
    model = SimpleNamespace(supervised_channels=[True] * 5)
    result = m.unified_operation(None, fov, model, cfg, "cpu", {})
    assert not result[:, ~fov].any()
    assert result[1, 0, 1] and not result[2, 0, 1]
    assert not result[4, 0, 1]  # Cup containment retained.
    assert result[4, 1, 0]
