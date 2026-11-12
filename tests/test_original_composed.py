"""Toy forwards exercise the literal benchmark operation without bundled weights."""

import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from deepretina import original as api
from deepretina.legacy.av import predict_av_input
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated
from deepretina.legacy.preprocessing import ImageGeometry, prepare_single_net
from deepretina.legacy.single_net import tiled_logits


class ToySingle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return torch.cat(
            [x[:, i % 3 : i % 3 + 1] + (i - 0.5) * 0.1 for i in range(self.channels)], 1
        )


class ToyAV(nn.Module):
    def forward(self, x):
        return torch.cat((x[:, :1] * 0 + 100, x[:, :1] - x[:, 1:2], x[:, 2:3] - x[:, :1]), 1)


@pytest.fixture
def inputs():
    rgb = np.random.default_rng(18).integers(15, 240, (31, 35, 3), dtype=np.uint8)
    yy, xx = np.indices(rgb.shape[:2])
    fov = ((yy - 15) / 13) ** 2 + ((xx - 17) / 15) ** 2 < 1
    return rgb, fov


def components():
    models = {
        k: (ToySingle(c), {"tiling": {"crop_size": n, "border": 4}})
        for k, c, n in (("vessel", 1, 32), ("disc", 1, 40), ("universal", 3, 48))
    }
    provenance = {k: {"model": name, "sha256": sha} for k, (name, sha) in api.COMPONENTS.items()}
    return models, ToyAV(), provenance


def literal_operation(rgb, fov, models, av, kwargs):
    prepared = prepare_single_net(rgb, fov=fov, normalized=False, **kwargs)
    native = {}
    for name, (model, payload) in models.items():
        logits = tiled_logits(model, prepared.tensor(), device="cpu", **payload["tiling"])
        native[name] = logits.numpy() > 0
    native["universal"][2] &= native["disc"][0]
    restored = {name: prepared.geometry.restore(mask, order=0) for name, mask in native.items()}
    vessel = restored["vessel"][0]
    calibrated = prepare_vesclass_matlab_calibrated(rgb, vessel, fov=fov, **kwargs)
    classes = predict_av_input(av, calibrated.inputs(), device="cpu", original_size=rgb.shape[:2])[
        "classes"
    ]
    return np.stack(
        (vessel, classes == 1, classes == 2, restored["disc"][0], restored["universal"][2])
    )


def test_actual_toy_forward_matches_literal_benchmark_and_exports(inputs, monkeypatch, tmp_path):
    rgb, fov = inputs
    before = rgb.copy(), fov.copy()
    monkeypatch.setattr(api, "_load_components", lambda device: components())
    models, av, _ = components()
    kwargs = {"field_of_view_degrees": 8, "pixels_per_degree": 25}
    expected = literal_operation(rgb, fov, models, av, kwargs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = api.predict_original(
            rgb, fov=fov, field_of_view_degrees=8, device="cpu", output=tmp_path / "out"
        )
    np.testing.assert_array_equal(result.masks, expected)
    np.testing.assert_array_equal(rgb, before[0])
    np.testing.assert_array_equal(fov, before[1])
    assert result.masks.dtype == bool and result.masks.shape == (5, 31, 35)
    assert result.probabilities.dtype == np.float32
    assert np.isfinite(result.probabilities).all()
    assert result.av_conditional_probabilities.shape == (2, 31, 35)
    np.testing.assert_allclose(
        result.probabilities[1:3].sum(0), result.av_vessel_support, atol=2e-7
    )
    meta = json.loads((tmp_path / "out/prediction.json").read_text())
    assert meta["policy"] == api.POLICY
    assert {k: v["sha256"] for k, v in meta["components"].items()} == {
        k: v[1] for k, v in api.COMPONENTS.items()
    }
    assert all((tmp_path / f"out/{name}.png").is_file() for name in api.CHANNELS)
    with np.load(tmp_path / "out/probabilities.npz", allow_pickle=False) as saved:
        assert saved["channels"].tolist() == list(api.CHANNELS)


def test_binary_native_intersection_is_not_interpolated_score_threshold():
    geometry = ImageGeometry((5, 7), (0, 4, 0, 6), 1.0, (2, 2))
    cup = np.array([[10.0, -0.01], [10.0, -0.01]])
    disc = np.array([[0.1, 0.1], [-10.0, -10.0]])
    expected = geometry.restore((cup > 0) & (disc > 0), order=0)
    bad = (geometry.restore(cup, order=1) > 0) & (geometry.restore(disc, order=1) > 0)
    assert np.any(expected != bad)


@pytest.mark.parametrize("case", ["unknown", "nan_fov", "color_fov", "special", "tiny", "diameter"])
def test_invalid_contract_before_weights(inputs, monkeypatch, case):
    rgb, fov = inputs
    kwargs = {"field_of_view_degrees": 8}
    if case == "unknown":
        kwargs = {"dataset": "unlisted-camera"}
    if case == "nan_fov":
        fov = fov.astype(float)
        fov[1, 1] = np.nan
    if case == "color_fov":
        fov = np.stack([fov] * 3, -1)
    if case == "special":
        kwargs["dataset"] = "riga"
    if case == "tiny":
        kwargs["field_of_view_degrees"] = 0.01
    if case == "diameter":
        kwargs["fov_diameter_pixels"] = True
    monkeypatch.setattr(api, "_load_components", lambda _: pytest.fail("must not load weights"))
    with pytest.raises(ValueError):
        api.predict_original(rgb, fov=fov, device="cpu", **kwargs)


def test_output_guard_precedes_input_and_weights(tmp_path, monkeypatch):
    (tmp_path / "existing").write_text("preserve")
    monkeypatch.setattr(api, "_load_components", lambda _: pytest.fail("must not load weights"))
    with pytest.raises(FileExistsError):
        api.predict_original(None, fov=None, output=tmp_path)
    assert (tmp_path / "existing").read_text() == "preserve"


def test_bad_component_bytes_never_load(tmp_path, monkeypatch):
    path = tmp_path / "fake.pt"
    path.write_bytes(b"not the reviewed weights")

    @contextmanager
    def bundle(name):
        yield path

    monkeypatch.setattr(api, "bundled_checkpoint", bundle)
    monkeypatch.setattr(api, "load_single_net", lambda *a, **k: pytest.fail("must not deserialize"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        api._load_components("cpu")


def test_cli_dispatch_and_parser():
    from deepretina.cli import parser

    args = parser().parse_args(
        [
            "predict-original",
            "--input",
            "image.png",
            "--fov",
            "fov.png",
            "--output",
            "out",
            "--dataset",
            "drive",
        ]
    )
    assert args.command == "predict-original" and args.device == "auto"
