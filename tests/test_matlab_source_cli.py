"""Source profile opt-in routing and paired preprocessing, without model forwards."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepretina.cli import main, parser
from deepretina.legacy import matlab_source_auto as source

ROOT = Path(__file__).parent / "reference" / "matlab_fov_source"


def example(tmp_path):
    meta = json.loads((ROOT / "source-auto-synthetic-metadata.json").read_text())
    case = meta["chains"][0]
    with np.load(ROOT / "source-auto-synthetic.npz", allow_pickle=False) as arrays:
        image = arrays[case["image"]].copy()
        vessel = arrays[case["vessel"]].copy()
    image_path, vessel_path = tmp_path / "rgb.png", tmp_path / "vessel.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(vessel.astype(np.uint8) * 255).save(vessel_path)
    return case, image, vessel, image_path, vessel_path


@pytest.mark.parametrize("flag", ["--av-raw-proxy", "--av-raw-matlab"])
def test_source_cli_is_mutually_exclusive(flag):
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["predict", "--model", "legacy-av", "--output", "x", "--av-raw-matlab-source", flag]
        )


def test_source_cli_forwards_existing_options_without_changing_default(
    tmp_path, monkeypatch, capsys
):
    import deepretina.legacy.raw_proxy as raw

    called = []
    monkeypatch.setattr(
        raw, "predict_raw_av", lambda *a, **kw: called.append(kw) or {"routed": True}
    )
    args = [
        "predict",
        "--model",
        "legacy-av",
        "--input",
        "raw.png",
        "--output",
        str(tmp_path),
        "--av-raw-matlab-source",
        "--dataset",
        "drive",
        "--fov",
        "fov.png",
        "--vessel-mask",
        "v.png",
        "--field-of-view-degrees",
        "45",
        "--fov-diameter-pixels",
        "540",
        "--device",
        "cpu",
    ]
    main(args)
    assert json.loads(capsys.readouterr().out) == {"routed": True}
    assert called[0]["preprocessing_profile"] == "matlab_source"
    assert called[0]["fov_path"] == Path("fov.png")
    assert called[0]["vessel_mask_path"] == Path("v.png")
    assert called[0]["fov_diameter_pixels"] == 540
    assert called[0]["threshold"] == 0.5
    args[args.index("--av-raw-matlab-source")] = "--av-raw-matlab"
    main(args)
    assert called[1]["preprocessing_profile"] == "matlab"


def test_raw_source_preparation_reuses_rgb_with_exact_vessel_sampling(tmp_path, monkeypatch):
    import deepretina.legacy.raw_proxy as raw

    case, image, vessel, image_path, vessel_path = example(tmp_path)
    expected = source.prepare_vesclass_matlab_source_auto(
        image, vessel, dataset=case["dataset"], field_of_view_degrees=case["degree"]
    )
    calls = []

    def no_forward(checkpoint, image, output, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(kwargs["prepared_inputs"], expected.inputs())
        output = Path(output)
        output.mkdir()
        metadata = output / "prediction.json"
        metadata.write_text(json.dumps({"preprocessing": {}}))
        return {"metadata": str(metadata)}

    monkeypatch.setattr(raw, "predict_legacy", no_forward)
    paths = raw.predict_raw_av(
        "unused",
        image_path,
        tmp_path / "out",
        dataset=case["dataset"],
        field_of_view_degrees=case["degree"],
        vessel_mask_path=vessel_path,
        preprocessing_profile="matlab_source",
        device="cpu",
    )
    assert len(calls) == 1
    metadata = json.loads(Path(paths["metadata"]).read_text())
    assert metadata["raw_av_preprocessing"]["name"] == "vesclass_matlab_source_auto"
    assert metadata["raw_av_preprocessing"]["matlab_parity_certified"] is False
    assert metadata["raw_av_geometry"]["original_shape"] == list(image.shape[:2])
    np.testing.assert_array_equal(np.asarray(Image.open(paths["vessel"])) > 0, vessel)


def test_provided_source_uses_processing_mask_and_explicit_diameter(tmp_path, monkeypatch):
    case, image, vessel, *_ = example(tmp_path)
    fov = np.ones(image.shape[:2], bool)
    original = source.prepare_vesclass_matlab_calibrated
    called = []

    def spy(*args, **kwargs):
        called.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "prepare_vesclass_matlab_calibrated", spy)
    prepared = source.prepare_vesclass_matlab_source(
        image,
        vessel,
        dataset="erlangen",
        fov=fov,
        field_of_view_degrees=case["degree"],
        fov_diameter_pixels=200,
    )
    assert prepared.profile["name"] == "vesclass_matlab_source_provided"
    assert prepared.source_fov.source_branch == "provided_erlangen_rails"
    assert called[0]["fov_diameter_pixels"] == 200
    assert called[0]["fov"] is prepared.source_fov.processing_fov
    assert prepared.profile["diameter_source"] == "explicit supplied override"
    assert prepared.profile["fov_calibration"] == {
        "mode": "explicit_override",
        "method": "caller_supplied_diameter",
        "center_row_one_based": None,
        "center_col_one_based": None,
        "radius_pixels": 100,
    }
    assert prepared.profile["used_diameter_pixels"] == 200
    assert prepared.profile["source_fov_calibration"] == {
        "mode": "estimated_from_source_provided_fov",
        "method": "provided_erlangen_rails",
        **asdict(prepared.source_fov.calibration),
    }
    assert prepared.profile["matlab_parity_certified"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": None},
        {"pixels_per_degree": 24},
        {"fov_diameter_pixels": float("nan")},
        {"dataset": "rimone"},
    ],
)
def test_invalid_source_request_never_reaches_model(tmp_path, monkeypatch, changes):
    import deepretina.legacy.raw_proxy as raw

    case, _, _, image, _ = example(tmp_path)
    monkeypatch.setattr(
        raw, "predict_legacy", lambda *a, **k: pytest.fail("invalid source request reached model")
    )
    kwargs = {"dataset": case["dataset"], "field_of_view_degrees": case["degree"], **changes}
    with pytest.raises(ValueError):
        raw.predict_raw_av(
            "unused", image, tmp_path / "out", preprocessing_profile="matlab_source", **kwargs
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("model", ["pilot-vessels", "unified-full-data-baseline"])
def test_source_cli_rejects_non_av_family(tmp_path, capsys, model):
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--model",
                model,
                "--input",
                "raw.png",
                "--output",
                str(tmp_path),
                "--av-raw-matlab-source",
            ]
        )
    assert "raw" in capsys.readouterr().err.lower()


def test_source_cli_refuses_prepared_mode(tmp_path, monkeypatch):
    import deepretina.legacy.raw_proxy as raw

    monkeypatch.setattr(
        raw, "predict_raw_av", lambda *a, **k: pytest.fail("mixed input modes reached preparation")
    )
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--model",
                "legacy-av",
                "--input",
                "raw.png",
                "--output",
                str(tmp_path),
                "--av-raw-matlab-source",
                "--prepared",
            ]
        )
