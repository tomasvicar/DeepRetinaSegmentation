"""A second model must not leave a mixture of old and new prediction files."""

import numpy as np
import pytest

from deepretina.inference import predict
from deepretina.legacy.inference import _save, predict_legacy
from deepretina.legacy.raw_proxy import predict_raw_av
from deepretina.prediction_output import check_prediction_output
from deepretina.unified import predict_unified


def test_reusing_universal_output_for_vessel_preserves_entire_first_prediction(tmp_path):
    output = tmp_path / "prediction"
    rgb = np.zeros((3, 4, 3), np.uint8)
    probability = np.full((3, 3, 4), 0.75, np.float32)
    _save(
        output, rgb, probability, ("vessel", "disc", "cup"), probability > 0.5, {}
    )
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="new or empty"):
        _save(output, rgb, probability[:1], ("vessel",), probability[:1] > 0.5, {})
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


@pytest.mark.parametrize("predictor", [predict, predict_legacy, predict_raw_av, predict_unified])
def test_populated_output_rejected_before_loading_model_or_image(tmp_path, predictor):
    output = tmp_path / "prediction"
    output.mkdir()
    (output / ".previous-run").write_bytes(b"preserve")
    # Missing inputs prove rejection happens before model loading/preprocessing/inference.
    with pytest.raises(FileExistsError, match="new or empty"):
        predictor(tmp_path / "missing.pt", tmp_path / "missing.png", output)
    assert list(output.iterdir()) == [output / ".previous-run"]
    assert (output / ".previous-run").read_bytes() == b"preserve"


def test_new_and_empty_output_are_accepted_without_creating_files(tmp_path):
    new = tmp_path / "new"
    assert check_prediction_output(new) == new
    assert not new.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_prediction_output(empty) == empty
    assert not list(empty.iterdir())


def test_output_file_is_preserved(tmp_path):
    output = tmp_path / "prediction"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="new or empty"):
        check_prediction_output(output)
    assert output.read_bytes() == b"preserve"
