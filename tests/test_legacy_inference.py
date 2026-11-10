"""Integration checks using the user's converted originals when locally available."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.legacy.av import load_av, predict_av_input
from deepretina.legacy.inference import predict_legacy
from deepretina.legacy.preprocessing import stack_vesclass_inputs

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def checkpoint(name):
    path = MODELS / name
    if not path.is_file():
        pytest.skip(f"Optional local converted model absent: {path.name}")
    return path


def photograph(tmp_path, shape=(32, 40)):
    rng = np.random.default_rng(2026)
    rgb = rng.integers(40, 240, (*shape, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    return path, rgb


def test_single_net_original_geometry_channels_and_zero_outside_crop(tmp_path):
    model = checkpoint("legacy-single-net-universal.pt")
    image_path, rgb = photograph(tmp_path)
    fov = np.zeros(rgb.shape[:2], dtype=np.uint8)
    fov[5:27, 7:34] = 255
    fov_path = tmp_path / "fov.png"
    Image.fromarray(fov).save(fov_path)
    paths = predict_legacy(
        model,
        image_path,
        tmp_path / "prediction",
        device="cpu",
        fov_path=fov_path,
        # Small explicit field of view keeps this integration check to one trained tile.
        field_of_view_degrees=1.6,
    )
    with np.load(paths["probabilities"]) as saved:
        assert saved["channels"].tolist() == ["vessel", "disc", "cup"]
        probability = saved["probability"]
    assert probability.shape == (3, 32, 40)
    assert np.isfinite(probability).all()
    assert (probability[:, fov == 0] == 0).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    for channel in ("vessel", "disc", "cup"):
        mask = np.array(Image.open(paths[channel]))
        assert mask.shape == (32, 40)
        assert set(np.unique(mask)) <= {0, 255}
        assert (mask[fov == 0] == 0).all()
    metadata = json.loads(Path(paths["metadata"]).read_text())
    assert metadata["preprocessing"]["normalization_verified"] is False
    assert metadata["family"] == "legacy_single_net"
    assert metadata["geometry"]["crop_position"] == [5, 26, 7, 33]
    assert metadata["tiling"]["crop_size"] == 288
    assert Image.open(paths["overlay"]).size == (40, 32)


def test_av_prepared_outputs_match_original_decoder_and_restore_full_frame(tmp_path):
    model_path = checkpoint("legacy-av-v9.pt")
    image_path, rgb = photograph(tmp_path)
    vessel = np.zeros(rgb.shape[:2], dtype=np.uint8)
    vessel[::2, 3:35] = 255
    vessel_path = tmp_path / "vessels.png"
    Image.fromarray(vessel).save(vessel_path)
    paths = predict_legacy(
        model_path,
        image_path,
        tmp_path / "av",
        device="cpu",
        prepared=True,
        vessel_mask_path=vessel_path,
        original_size=(64, 80),
    )
    model, _ = load_av(model_path)
    expected = predict_av_input(model, stack_vesclass_inputs(rgb, vessel), original_size=(64, 80))
    classes = np.array(Image.open(paths["classes"]))
    np.testing.assert_array_equal(classes, expected["classes"])
    np.testing.assert_array_equal(np.array(Image.open(paths["artery"])) > 0, classes == 1)
    np.testing.assert_array_equal(np.array(Image.open(paths["vein"])) > 0, classes == 2)
    with np.load(paths["probabilities"]) as saved:
        assert saved["channels"].tolist() == ["artery", "vein"]
        assert saved["probability"].shape == (2, 64, 80)
        np.testing.assert_allclose(saved["conditional_probability"].sum(axis=0), 1, atol=1e-6)
        assert (saved["probability"][:, ~saved["vessel_support"]] == 0).all()
        np.testing.assert_allclose(
            saved["conditional_probability"][:, ::2, ::2],
            expected["prepared_conditional_probabilities"],
            atol=0,
            rtol=0,
        )
    metadata = json.loads(Path(paths["metadata"]).read_text())
    assert metadata["preprocessing"]["raw_matlab_pipeline_certified"] is False
    assert metadata["threshold_used"] is False
    assert Image.open(paths["overlay"]).size == (80, 64)


def test_av_raw_input_is_rejected_before_outputs_are_created(tmp_path):
    model = checkpoint("legacy-av-v9.pt")
    image_path, _ = photograph(tmp_path)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="already-preprocessed RGB"):
        predict_legacy(model, image_path, output, device="cpu")
    assert not output.exists()


def test_av_empty_mask_and_color_visualization_are_rejected(tmp_path):
    model = checkpoint("legacy-av-v9.pt")
    image_path, rgb = photograph(tmp_path)
    path = tmp_path / "mask.png"
    Image.fromarray(np.zeros(rgb.shape[:2], dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="zero and one positive"):
        predict_legacy(model, image_path, tmp_path / "empty", prepared=True, vessel_mask_path=path)
    with pytest.raises(ValueError, match="single-channel"):
        predict_legacy(
            model, image_path, tmp_path / "color", prepared=True, vessel_mask_path=image_path
        )


def test_av_stored_dicom_planes_preserve_values_and_decoder(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    model_path = checkpoint("legacy-av-v9.pt")
    _, rgb = photograph(tmp_path, (19, 23))
    planes = [channel.astype(np.uint16) * 16 for channel in rgb.transpose(2, 0, 1)]
    vessel = np.zeros((19, 23), dtype=np.uint16)
    vessel[2:15:2, 4:19] = 255
    planes.append(vessel)
    paths = []
    for index, plane in enumerate(planes):
        path = tmp_path / f"plane{index}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Rows, ds.Columns = plane.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 3  # Historical input ignores display rescale metadata.
        ds.RescaleIntercept = -40
        ds.PixelData = plane.tobytes()
        ds.save_as(path)
        paths.append(path)
    outputs = predict_legacy(
        model_path, None, tmp_path / "dicom_result", device="cpu", dicom_paths=paths
    )
    model, _ = load_av(model_path)
    expected = predict_av_input(model, np.stack(planes).astype(np.float32))
    np.testing.assert_array_equal(np.array(Image.open(outputs["classes"])), expected["classes"])
    with np.load(outputs["probabilities"]) as saved:
        np.testing.assert_array_equal(
            saved["conditional_probability"], expected["prepared_conditional_probabilities"]
        )
