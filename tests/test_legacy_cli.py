"""Public command-line routes distinguish verified originals from pilot models."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.cli import main


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cli_original_vessel_alias_outputs_native_geometry(tmp_path, capsys):
    rgb = np.zeros((47, 59, 3), np.uint8)
    rgb[6:-6, 6:-6] = np.random.default_rng(8).integers(30, 220, (35, 47, 3), np.uint8)
    image = tmp_path / "fundus.png"
    Image.fromarray(rgb).save(image)
    main(
        [
            "predict",
            "--model",
            "vessels",
            "--input",
            str(image),
            "--output",
            str(tmp_path / "out"),
            "--field-of-view-degrees",
            "2",
            "--device",
            "cpu",
        ]
    )
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    assert info["family"] == "legacy_single_net"
    assert info["shape"] == [47, 59]
    assert info["checkpoint_preprocessing"]["normalization"] == "RGB/255-0.5"
    assert Image.open(outputs["vessel"]).size == (59, 47)


def test_cli_av_prepared_array_keeps_original_input_values(tmp_path, capsys):
    rng = np.random.default_rng(15)
    planes = rng.uniform(0, 4096, (4, 31, 37)).astype(np.float32)
    planes[3] = 0
    planes[3, 5:25, 10:13] = 1
    path = tmp_path / "prepared.npz"
    np.savez_compressed(path, input=planes)
    main(
        [
            "predict",
            "--model",
            "legacy-av",
            "--prepared-array",
            str(path),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ]
    )
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    assert info["family"] == "legacy_av"
    assert info["preprocessing"]["profile"] == "caller_supplied_prepared_four_channel_tensor"
    assert not info["threshold_used"]
    assert Image.open(outputs["classes"]).size == (37, 31)


def test_cli_av_raw_proxy_is_explicit_and_marked(tmp_path, capsys):
    rng = np.random.default_rng(19)
    rgb = rng.integers(30, 210, (48, 64, 3), np.uint8)
    image = tmp_path / "fundus.png"
    Image.fromarray(rgb).save(image)
    vessel = np.zeros((48, 64), np.uint8)
    vessel[8:38, 30:34] = 255
    mask = tmp_path / "vessel.png"
    Image.fromarray(vessel).save(mask)
    main(
        [
            "predict",
            "--model",
            "legacy-av",
            "--av-raw-proxy",
            "--input",
            str(image),
            "--vessel-mask",
            str(mask),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--field-of-view-degrees",
            "2",
            "--fov-diameter-pixels",
            "64",
        ]
    )
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    assert not info["raw_av_proxy"]["matlab_parity_certified"]
    assert info["shape"] == [48, 64]
    assert Image.open(outputs["artery"]).size == (64, 48)


def test_cli_refuses_to_treat_raw_rgb_as_certified_av_input(tmp_path):
    image = tmp_path / "fundus.png"
    Image.fromarray(np.full((32, 32, 3), 100, np.uint8)).save(image)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "predict",
                "--model",
                "legacy-av",
                "--input",
                str(image),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
