"""Regression tests for scientific label semantics and source separation."""

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from deepretina.prepare import assign_splits, decode_av, decode_refuge, prepare


def test_unknown_av_pixels_do_not_train_false_negative_classes():
    colors = np.array(
        [[[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8
    )
    target, valid = decode_av(colors, np.array([[1, 1, 1, 1, 0]], dtype=bool))
    np.testing.assert_array_equal(target[:, 0], [[1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(valid[1, 0], [1, 1, 0, 0, 0])
    assert valid[0].all()


def test_refuge_disc_includes_cup():
    target = decode_refuge(np.array([[255, 128, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(target[:, 0], [[0, 1, 1], [0, 0, 1]])


def test_sources_and_eye_pairs_cannot_leak_between_splits():
    records = []
    for i in range(20):
        for task in ("vessels", "av"):
            records.append(
                {"source_id": f"drive:{i}", "split": "test" if i < 5 else "train", "task": task}
            )
    assign_splits(records, 42)
    assert {r["split"] for r in records if int(r["source_id"].split(":")[1]) < 5} == {"test"}
    assert {r["split"] for r in records} == {"train", "val", "test"}
    for i in range(20):
        assert len({r["split"] for r in records if r["source_id"] == f"drive:{i}"}) == 1
    again = [
        dict(r, split="test" if int(r["source_id"].split(":")[1]) < 5 else "train") for r in records
    ]
    assign_splits(again, 42)
    assert records == again


def _png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def test_prepare_drive_deduplicates_export_and_keeps_coordinates(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    image = np.zeros((24, 32, 3), np.uint8)
    image[4:20, 4:28] = 100
    mask = np.zeros((24, 32), np.uint8)
    mask[10:12, 4:28] = 255
    with zipfile.ZipFile(data / "DRIVE.zip", "w") as z:
        for prefix in ("", "content/DRIVE/"):
            z.writestr(prefix + "training/images/21_training.tif", _png(image))
            z.writestr(prefix + "training/1st_manual/21_manual1.gif", _png(mask))
            z.writestr(
                prefix + "training/mask/21_training_mask.gif",
                _png(np.full(mask.shape, 255, np.uint8)),
            )
    manifest = prepare(data, tmp_path / "prepared", ["drive"], max_size=32)
    records = json.loads(manifest.read_text())
    assert len(records) == 1
    row = records[0]
    with np.load(manifest.parent / row["mask"]) as labels:
        np.testing.assert_array_equal(labels["target"][0], mask > 127)
        assert labels["valid"].all()
        assert labels["target"].dtype == np.float32
    assert row["source_id"] == "drive:21"
    assert row["original_size"] == [24, 32]
    assert row["preparation_max_size"] == 32
    provenance = json.loads((manifest.parent / "preparation.json").read_text())
    assert provenance["seed"] == 42
    assert provenance["max_size"] == 32
    assert provenance["archives"][0]["members"][0]["crc32"] > 0


def test_preparation_refuses_to_modify_existing_experiment_data(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    previous = output / "manifest.json"
    previous.write_bytes(b"original experiment metadata")
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare(tmp_path, output, ["drive"], max_size=16)
    assert previous.read_bytes() == b"original experiment metadata"
    assert list(output.iterdir()) == [previous]
