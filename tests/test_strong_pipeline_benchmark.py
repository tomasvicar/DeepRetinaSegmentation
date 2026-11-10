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


def test_palette_fov_uses_displayed_intensity_not_indices(tmp_path, benchmark_module):
    from PIL import Image

    m = benchmark_module
    image = tmp_path / "image.png"
    fov_path = tmp_path / "fov.png"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    indices = np.array([[0, 1, 1], [1, 0, 1]], np.uint8)
    fov = Image.frombytes("P", (3, 2), indices.tobytes())
    fov.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
    fov.save(fov_path)
    rgb, mask, identity = m.input_snapshots(image, fov_path)
    np.testing.assert_array_equal(mask, indices == 0)
    assert rgb.shape == (2, 3, 3) and identity["fov_sha256"] == m.sha256(fov_path)


def test_input_identity_matches_decoded_snapshot_after_path_replacement(
    tmp_path, benchmark_module, monkeypatch
):
    import hashlib
    import io

    from PIL import Image

    m = benchmark_module
    image = tmp_path / "image.png"
    fov = tmp_path / "fov.png"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    Image.fromarray(np.array([[0, 255, 255], [255, 0, 255]], np.uint8)).save(fov)
    image_bytes, fov_bytes = image.read_bytes(), fov.read_bytes()
    replacement = io.BytesIO()
    Image.fromarray(np.full((2, 3, 3), 190, np.uint8)).save(replacement, format="PNG")
    read = Path.read_bytes

    def swap(self):
        data = read(self)
        if self == image:
            self.write_bytes(replacement.getvalue())
        if self == fov:
            self.write_bytes(image_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", swap)
    rgb, mask, identity = m.input_snapshots(image, fov)
    assert (rgb == 80).all()
    np.testing.assert_array_equal(mask, np.array([[False, True, True], [True, False, True]]))
    assert identity == {
        "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "fov_sha256": hashlib.sha256(fov_bytes).hexdigest(),
    }
    assert hashlib.sha256(read(image)).hexdigest() != identity["input_sha256"]


def test_benchmark_report_retains_timed_input_snapshot_hashes(
    tmp_path, benchmark_module, reference_files, monkeypatch
):
    import hashlib
    import json

    from PIL import Image

    m = benchmark_module
    paths, reference = reference_files
    image = tmp_path / "input.png"
    fov = tmp_path / "fov.png"
    unified = tmp_path / "unified.pt"
    ref = tmp_path / "ref.json"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    Image.fromarray(np.ones((2, 3), np.uint8) * 255).save(fov)
    unified.write_bytes(b"unified")
    ref.write_text(json.dumps(reference))
    snapshots = {
        "input_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "fov_sha256": hashlib.sha256(fov.read_bytes()).hexdigest(),
    }
    channels = {"vessel": ["vessel"], "disc": ["disc"], "universal": ["vessel", "disc", "cup"]}
    monkeypatch.setattr(
        m,
        "load_single_net",
        lambda path, device: (
            torch.nn.Linear(1, 1),
            {"channels": channels[path.stem], "tiling": {"crop_size": 288, "border": 36}},
        ),
    )
    monkeypatch.setattr(m, "load_av", lambda *args: (torch.nn.Linear(1, 1), {}))
    model = torch.nn.Linear(1, 1)
    model.supervised_channels = [True] * 5
    monkeypatch.setattr(
        m,
        "load_unified",
        lambda *args: (model, {"config": {"purpose": "full_data"}}, torch.device("cpu")),
    )
    calls = []

    def inspect_original(rgb, mask, *args):
        assert (rgb == 80).all() and mask.all()
        calls.append("original")
        return np.zeros((5, 2, 3), bool)

    def inspect_unified(rgb, mask, *args):
        assert (rgb == 80).all() and mask.all()
        calls.append("unified")
        return np.zeros((5, 2, 3), bool)

    monkeypatch.setattr(m, "original_operation", inspect_original)
    monkeypatch.setattr(m, "unified_operation", inspect_unified)

    def measured(operation, *args, **kwargs):
        operation()
        image.write_bytes(b"path replaced after decoding")
        fov.write_bytes(b"path replaced after decoding")
        return {"median_seconds": 1}

    monkeypatch.setattr(m, "measure", measured)
    args = SimpleNamespace(
        **paths,
        unified=unified,
        reference_report=ref,
        input=image,
        fov=fov,
        dataset="drive",
        field_of_view_degrees=None,
        device="cpu",
        warmup=0,
        repeats=1,
    )
    report = m.benchmark(args)
    assert calls == ["original", "unified"]
    assert {key: report[key] for key in snapshots} == snapshots
