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


def test_train_only_patient_policy_is_honored_without_forging_an_official_split():
    records = [
        {
            "source_id": f"riga:image{i}",
            "split": "unsplit",
            "metadata": {"split_policy": "train_only_without_patient_crosswalk"},
        }
        for i in range(20)
    ]
    assign_splits(records, 42)
    assert {row["split"] for row in records} == {"train"}
    assert all("official_split" not in row["metadata"] for row in records)


@pytest.mark.parametrize("heldout", ["val", "test"])
def test_train_only_policy_cannot_override_a_known_official_alias(heldout):
    records = [
        {
            "source_id": "shared:photo",
            "split": "unsplit",
            "metadata": {"split_policy": "train_only_without_patient_crosswalk"},
        },
        {"source_id": "shared:photo", "split": heldout, "metadata": {}},
    ]
    with pytest.raises(ValueError, match="official held-out"):
        assign_splits(records, 42)


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


def test_photo_hash_uses_original_geometry_and_is_independent_of_memory_layout():
    from deepretina.prepare import decoded_photo_id

    image = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
    same = np.ascontiguousarray(image.transpose(1, 0, 2)).transpose(1, 0, 2)
    assert not same.flags.c_contiguous
    assert decoded_photo_id(image) == decoded_photo_id(same)
    assert decoded_photo_id(image) != decoded_photo_id(image.reshape(32, 16, 3))
    assert decoded_photo_id(image) != decoded_photo_id(255 - image)


def test_exact_photo_union_crosses_dataset_ids_and_transitively_keeps_patient_eyes():
    # Two different photos of patientA; one has another dataset's alias. Neither
    # files nor photo identities are merged, but the connected group shares a split.
    records = [
        {
            "source_id": "a:patientA",
            "photo_id": "sha256:left",
            "image": "left.png",
            "split": "unsplit",
        },
        {
            "source_id": "a:patientA",
            "photo_id": "sha256:right",
            "image": "right.png",
            "split": "unsplit",
        },
        {
            "source_id": "b:otherName",
            "photo_id": "sha256:left",
            "image": "alias.png",
            "split": "test",
        },
        {
            "source_id": "b:otherName",
            "photo_id": "sha256:anotherView",
            "image": "other.png",
            "split": "unsplit",
        },
    ]
    identities = [(r["source_id"], r["photo_id"], r["image"]) for r in records]
    assign_splits(records, 42)
    assert {r["split"] for r in records} == {"test"}
    assert len({r["split_group_id"] for r in records}) == 1
    assert identities == [(r["source_id"], r["photo_id"], r["image"]) for r in records]
    assert len({r["photo_id"] for r in records}) == 3


@pytest.mark.parametrize("heldout", ["val", "test"])
def test_photo_alias_preserves_official_heldout_and_train_only_fails_closed(heldout):
    records = [
        {"source_id": "a:unrelatedID", "photo_id": "sameRGB", "split": "unsplit"},
        {
            "source_id": "b:officialID",
            "photo_id": "sameRGB",
            "split": heldout,
            "metadata": {"official_split": heldout},
        },
    ]
    assign_splits(records, 42)
    assert {r["split"] for r in records} == {heldout}
    records[0].update(
        split="unsplit", metadata={"split_policy": "train_only_without_patient_crosswalk"}
    )
    with pytest.raises(ValueError, match="official held-out"):
        assign_splits(records, 42)


def test_photo_alias_with_official_val_and_test_uses_stricter_test_partition():
    records = [
        {"source_id": "a:1", "photo_id": "same", "split": "val"},
        {"source_id": "b:2", "photo_id": "same", "split": "test"},
    ]
    assign_splits(records, 42)
    assert {r["split"] for r in records} == {"test"}


def test_preparation_hashes_raw_rgb_before_resize_and_caches_observer_resources(
    tmp_path, monkeypatch
):
    from deepretina import prepare as module
    from deepretina.adapters import segmentation

    # Valid empty archive; the synthetic adapter supplies actual pixel arrays.
    with zipfile.ZipFile(tmp_path / "ARIA.zip", "w"):
        pass
    raw = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    second = 255 - raw
    target = np.ones((1, 24, 32), np.float32)

    def adapter(source, emit):
        for stem, patient, image, resource, observer in [
            ("left_first", "patientA", raw, "left.tif", "first"),
            ("left_second", "patientA", raw, "left.tif", "second"),
            ("right", "patientA", second, "right.tif", "first"),
            ("alias", "patientB", raw, "different.tif", "first"),
        ]:
            fov = np.ones(image.shape[:2], bool)
            if stem == "alias":
                fov[:, :16] = False  # Identity is independent of annotation/FOV policy.
            emit(
                stem,
                "unsplit",
                image,
                target,
                fov=fov,
                source_id=patient,
                metadata={"source_image": resource, "annotator": observer},
            )

    monkeypatch.setitem(segmentation.ADAPTERS, "aria", adapter)
    calls = []
    original = module.decoded_photo_id

    def capture(image):
        calls.append(image.shape)
        return original(image)

    monkeypatch.setattr(module, "decoded_photo_id", capture)
    manifest = prepare(tmp_path, tmp_path / "prepared", ["aria"], max_size=16)
    rows = json.loads(manifest.read_text())
    assert calls == [(24, 32, 3)] * 3  # Same resource/second observer uses the cache.
    assert len({row["photo_id"] for row in rows}) == 2
    assert len({row["split_group_id"] for row in rows}) == 1
    assert {row["split"] for row in rows} == {"train"}
    left = [row for row in rows if "left" in row["id"]]
    assert left[0]["image"] == left[1]["image"]
    assert len({row["image"] for row in rows}) == 3  # Distinct original resources remain portable.
    assert all(row["photo_id"] == original(raw) for row in rows if "right" not in row["id"])
    for row in rows:
        with Image.open(manifest.parent / row["image"]) as image:
            assert image.size == (16, 12)
