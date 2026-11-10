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


def matlab_cli_args(tmp_path):
    rng = np.random.default_rng(81)
    image = tmp_path / "fundus.png"
    Image.fromarray(rng.integers(35, 220, (96, 112, 3), np.uint8)).save(image)
    fov = tmp_path / "fov.png"
    mask = np.zeros((96, 112), np.uint8)
    mask[4:-4, 4:-4] = 255
    Image.fromarray(mask).save(fov)
    vessel = tmp_path / "vessel.png"
    mask[:] = 0
    mask[12:80, 25:29] = 255
    mask[38:42, 15:90] = 255
    Image.fromarray(mask).save(vessel)
    return [
        "predict",
        "--model",
        "legacy-av",
        "--av-raw-matlab",
        "--input",
        str(image),
        "--fov",
        str(fov),
        "--field-of-view-degrees",
        "8",
        "--fov-diameter-pixels",
        "100",
        "--vessel-mask",
        str(vessel),
        "--output",
        str(tmp_path / "out"),
        "--device",
        "cpu",
    ]


def test_cli_matlab_profile_bundled_av_canary(tmp_path, capsys):
    main(matlab_cli_args(tmp_path))
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    assert info["family"] == "legacy_av"
    assert info["shape"] == [96, 112]
    assert info["inference_shape"] == [192, 224]
    profile = info["raw_av_preprocessing"]
    assert profile["name"] == "vesclass_matlab_provided_fov_calibrated"
    assert not profile["matlab_parity_certified"]
    assert not profile["historical_training_file_provenance_verified"]
    assert not profile["original_single_net_python_profile_changed"]
    assert info["preprocessing"]["raw_profile"] == profile["name"]
    assert info["raw_av_geometry"]["original_shape"] == [96, 112]
    assert info["raw_av_geometry"]["prepared_shape"] == [192, 224]
    assert "raw_av_proxy" not in info
    assert info["vessel_prediction"] is None
    assert profile["fov_calibration"]["mode"] == "explicit_diameter"
    for name in ["artery", "vein", "classes", "vessel", "overlay"]:
        assert Image.open(outputs[name]).size == (112, 96)


@pytest.mark.parametrize(
    ("remove", "extra", "message"),
    [
        ("--fov", [], "provided --fov"),
        ("--field-of-view-degrees", [], "known --dataset"),
        ("--fov-diameter-pixels", ["--fov-diameter-pixels", "-1"], "positive finite"),
        ("--fov-diameter-pixels", ["--fov-diameter-pixels", "nan"], "positive finite"),
        ("--field-of-view-degrees", ["--field-of-view-degrees", "0"], "known --dataset"),
        (None, ["--prepared"], "prepared input modes"),
        (None, ["--av-raw-proxy"], "not allowed with argument"),
        (None, ["--prepared-array", "missing.npy"], "single prepared"),
        (None, ["--dicoms", "r", "g", "b", "v"], "prepared input modes"),
    ],
)
def test_cli_matlab_rejects_invalid_modes_before_prediction(
    tmp_path, monkeypatch, capsys, remove, extra, message
):
    import deepretina.legacy.raw_proxy as raw

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid calibration must be rejected before any model inference")

    monkeypatch.setattr(raw, "predict_legacy", forbidden)
    args = matlab_cli_args(tmp_path)
    if remove:
        i = args.index(remove)
        del args[i : i + 2]
    with pytest.raises(SystemExit) as exc:
        main(args + extra)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_matlab_rejects_wrong_model(tmp_path, capsys):
    args = matlab_cli_args(tmp_path)
    args[args.index("legacy-av")] = "pilot-vessels"
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert "original A/V model" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_matlab_auto_vessel_path_preserves_original_model(tmp_path, monkeypatch, capsys):
    import deepretina.legacy.raw_proxy as raw

    actual = raw.predict_legacy
    calls = []
    args = matlab_cli_args(tmp_path)
    args.extend(["--dataset", "drive"])
    i = args.index("--vessel-mask")
    vessel = args[i + 1]
    del args[i : i + 2]

    def segmentation_then_actual_av(checkpoint, image, output, **kwargs):
        calls.append(kwargs)
        if "prepared_inputs" in kwargs:
            return actual(checkpoint, image, output, **kwargs)
        # Segmentation result contract only; the A/V forward still uses bundled weights.
        metadata = output / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "family": "legacy_single_net",
                    "outputs": {"metadata": str(metadata), "vessel": vessel},
                }
            )
        )
        return {"vessel": vessel, "metadata": str(metadata)}

    monkeypatch.setattr(raw, "predict_legacy", segmentation_then_actual_av)
    main(args)
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    assert len(calls) == 2
    assert calls[0]["normalized"] is False
    assert calls[0]["fov_path"] == tmp_path / "fov.png"
    assert info["vessel_prediction"]["family"] == "legacy_single_net"
    assert info["dataset"] == "drive"
    assert info["vessel_prediction"]["outputs"] == {"vessel": outputs["vessel"]}
    assert all(Path(path).is_file() for path in info["vessel_prediction"]["outputs"].values())
    assert info["raw_av_preprocessing"]["name"] == "vesclass_matlab_provided_fov_calibrated"


@pytest.mark.parametrize("invalid", ["shape", "empty", "tiny"])
def test_cli_matlab_geometry_guard_before_vessel_model(tmp_path, monkeypatch, capsys, invalid):
    import deepretina.legacy.raw_proxy as raw

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid FOV/calibration reached vessel model")

    monkeypatch.setattr(raw, "predict_legacy", forbidden)
    args = matlab_cli_args(tmp_path)
    i = args.index("--vessel-mask")
    del args[i : i + 2]
    if invalid == "shape":
        Image.fromarray(np.ones((40, 50), np.uint8)).save(tmp_path / "fov.png")
    elif invalid == "empty":
        Image.fromarray(np.zeros((96, 112), np.uint8)).save(tmp_path / "fov.png")
    else:
        args[args.index("--fov-diameter-pixels") + 1] = "10000"
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
    assert {"shape": "dimensions differ", "empty": "foreground", "tiny": "too small"}[
        invalid
    ] in capsys.readouterr().err


def test_cli_matlab_bundled_av_automatic_fov_calibration(tmp_path, capsys):
    args = matlab_cli_args(tmp_path)
    i = args.index("--fov-diameter-pixels")
    del args[i : i + 2]
    main(args)
    outputs = json.loads(capsys.readouterr().out)
    info = json.loads(Path(outputs["metadata"]).read_text())
    profile = info["raw_av_preprocessing"]
    calibration = profile["fov_calibration"]
    assert calibration["mode"] == "estimated_from_provided_fov"
    assert calibration["method"] == "souradnice_generic_binary_sobel"
    assert calibration["center_row_one_based"] > 0
    assert calibration["center_col_one_based"] > 0
    assert profile["fov_diameter_pixels"] == 2 * calibration["radius_pixels"]
    assert info["shape"] == [96, 112]
    assert Image.open(outputs["classes"]).size == (112, 96)
    assert not profile["matlab_parity_certified"]
    assert not profile["historical_training_file_provenance_verified"]


@pytest.mark.parametrize("invalid", ["empty", "full", "special"])
def test_cli_matlab_auto_calibration_fails_closed_before_vessel_model(
    tmp_path, monkeypatch, capsys, invalid
):
    import deepretina.legacy.raw_proxy as raw

    def forbidden(*args, **kwargs):
        pytest.fail("Unusable FOV or unsupported source reached model inference")

    monkeypatch.setattr(raw, "predict_legacy", forbidden)
    args = matlab_cli_args(tmp_path)
    for flag in ["--fov-diameter-pixels", "--vessel-mask"]:
        i = args.index(flag)
        del args[i : i + 2]
    if invalid == "special":
        args += ["--dataset", "riga"]
    else:
        Image.fromarray(np.full((96, 112), 255 if invalid == "full" else 0, np.uint8)).save(
            tmp_path / "fov.png"
        )
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    expected = {"empty": "foreground", "full": "paired edge", "special": "special souradnice"}
    assert expected[invalid] in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("dtype", "tiny"), [(torch.float32, 1e-8), (torch.float64, 1e-16)])
def test_raw_av_automatic_vessel_pipeline_preserves_tiny_positive_default_logits(
    tmp_path, monkeypatch, capsys, dtype, tiny
):
    from deepretina.legacy import inference

    def tiny_vessel_logits(model, image, *args, **kwargs):
        h, w = image.shape[-2:]
        logits = torch.full((1, h, w), -tiny, dtype=dtype)
        logits[:, h // 4 : 3 * h // 4, w // 2 : w // 2 + 4] = tiny
        logits[:, 0] = 0
        assert (logits.sigmoid() == 0.5).all()
        return logits

    monkeypatch.setattr(inference, "tiled_logits", tiny_vessel_logits)
    args = matlab_cli_args(tmp_path)
    i = args.index("--vessel-mask")
    del args[i : i + 2]
    main(args)
    paths = json.loads(capsys.readouterr().out)
    info = json.loads(Path(paths["metadata"]).read_text())
    vessel = np.asarray(Image.open(paths["vessel"])) > 0
    assert 0 < vessel.sum() < vessel.size
    assert info["vessel_prediction"]["mask_decision"] == "strict_logit_gt_zero"
    assert info["vessel_prediction"]["threshold"] == 0.5
    assert Image.open(paths["classes"]).size == (112, 96)
