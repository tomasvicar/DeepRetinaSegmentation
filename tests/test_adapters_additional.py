"""Native annotation semantics, unknown supervision and official split guards."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from deepretina.adapters.additional import (
    decode_gamma_mask,
    disc_from_points,
    drions,
    gamma,
    idrid_disc,
)
from deepretina.prepare import prepare
from deepretina.sources import Source


def encoded(a, fmt="PNG"):
    stream = io.BytesIO()
    Image.fromarray(a).save(stream, format=fmt)
    return stream.getvalue()


def contours():
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    return "\n".join(f"{8 + 4 * np.cos(x)},{7 + 3 * np.sin(x)}" for x in t)


def test_drions_two_observers_binary_disc_and_unknown_cup(tmp_path):
    path = tmp_path / "DRIONS-DB.zip"
    rgb = np.full((16, 20, 3), 100, np.uint8)
    with zipfile.ZipFile(path, "w") as z:
        for prefix in ["", "DRIONS-DB/"]:
            z.writestr(prefix + "images/image_001.jpg", encoded(rgb))
            for observer in [1, 2]:
                z.writestr(prefix + f"experts_anotation/anotExpert{observer}_001.txt", contours())
    rows = []
    with Source(path) as source:
        drions(source, lambda *a, **k: rows.append((a, k)))
    assert len(rows) == 2
    for args, kwargs in rows:
        assert args[1] == "train" and args[3].shape == (2, 16, 20)
        assert args[3][0, 6, 7] == 1 and args[3][0, 0, 0] == 0
        assert not args[3][1].any() and not kwargs["valid"][1].any()
        assert kwargs["valid"][0].all()
        assert kwargs["source_id"] == "drions:001"
    assert rows[0][0][0] != rows[1][0][0]
    manifest = prepare(tmp_path, tmp_path / "out", ["drions"], max_size=32)
    import json

    written = json.loads(manifest.read_text())
    assert len(written) == 2 and {r["split"] for r in written} == {"train"}


def test_polygon_one_based_xy_and_rejects_bad_points():
    mask = disc_from_points(contours(), (16, 20))
    assert mask[6, 7] and not mask[0, 0]
    with pytest.raises(ValueError):
        disc_from_points("1,2\n2,3", (16, 20))
    with pytest.raises(ValueError):
        disc_from_points(contours().replace("12.0", "120.0"), (16, 20))


def gamma_archive(path, bad=False):
    rgb = np.full((16, 20, 3), 100, np.uint8)
    label = np.full((16, 20), 255, np.uint8)
    label[3:13, 4:16] = 128
    label[6:10, 8:12] = 0
    if bad:
        label[1, 1] = 7
    with zipfile.ZipFile(path, "w") as z:
        pre = "GAMMA/GAMMA_training data/training_data/"
        z.writestr(pre + "Disc_Cup_Mask/0001.png", encoded(label))
        z.writestr(pre + "multi-modality_images/0001/0001.jpg", encoded(rgb))
        z.writestr(pre + "multi-modality_images/0001/0001/1.jpg", b"never decode OCT")
        z.writestr(pre + "Disc_Cup_Fovea_check/0001.png", b"never decode overlay")
    return label


def test_gamma_native_class_encoding_and_only_fundus_resources(tmp_path):
    path = tmp_path / "GAMMA.zip"
    label = gamma_archive(path)
    rows = []
    with Source(path) as source:
        gamma(source, lambda *a, **k: rows.append((a, k)))
    assert len(rows) == 1
    args, kw = rows[0]
    target = args[3]
    np.testing.assert_array_equal(target[0], label != 255)
    np.testing.assert_array_equal(target[1], label == 0)
    assert kw["metadata"]["official_split"] == "train"
    assert kw["metadata"]["split_policy"] == "train_only_without_patient_crosswalk"
    assert kw["metadata"]["physical_calibration"]["field_of_view_degrees"] is None
    assert not (target[1].astype(bool) & ~target[0].astype(bool)).any()
    with pytest.raises(ValueError):
        decode_gamma_mask(np.full((3, 4), 7))


def test_idrid_train_only_audit_never_opens_test_labels(tmp_path):
    path = tmp_path / "IDRiD.zip"
    rgb = np.full((16, 20, 3), 100, np.uint8)
    mask = np.zeros((16, 20), np.uint8)
    mask[3:10, 4:12] = 255
    with zipfile.ZipFile(path, "w") as z:
        for folder, stem in [("a. Training Set", "IDRiD_01"), ("b. Testing Set", "IDRiD_55")]:
            root = "IDRiD/A. Segmentation/"
            z.writestr(root + f"1. Original Images/{folder}/{stem}.jpg", encoded(rgb))
            z.writestr(
                root + f"2. All Segmentation Groundtruths/{folder}/5. Optic Disc/{stem}_OD.tif",
                encoded(mask),
            )
    rows = []
    with Source(path) as source:
        original = source.read

        def training_read(name, *a, **kw):
            assert "Testing" not in name
            return original(name, *a, **kw)

        source.read = training_read
        original_bytes = source.bytes

        def training_bytes(name):
            assert "Testing" not in name
            return original_bytes(name)

        source.bytes = training_bytes
        idrid_disc(source, lambda *a, **k: rows.append((a, k)), splits=("train",))
    assert len(rows) == 1
    args, kw = rows[0]
    assert args[1] == "train" and kw["metadata"]["official_split"] == "train"
    np.testing.assert_array_equal(args[3][0], mask > 0)
    assert not kw["valid"][1].any()
    assert kw["metadata"]["physical_calibration"]["field_of_view_degrees"] == 50
    with Source(path) as source:
        all_rows = []
        idrid_disc(source, lambda *a, **k: all_rows.append((a, k)))
    assert {a[1] for a, k in all_rows} == {"train", "test"}


def test_idrid_palette_indices_are_not_thresholded_rendered_red(tmp_path):
    path = tmp_path / "IDRiD.zip"
    image = np.full((16, 20, 3), 100, np.uint8)
    label = np.zeros((16, 20), np.uint8)
    label[3:10, 4:12] = 1
    mask = Image.fromarray(label, mode="P")
    mask.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    stream = io.BytesIO()
    mask.save(stream, format="TIFF")
    with zipfile.ZipFile(path, "w") as z:
        root = "IDRiD/A. Segmentation/"
        z.writestr(root + "1. Original Images/a. Training Set/IDRiD_01.jpg", encoded(image))
        z.writestr(
            root + "2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/IDRiD_01_OD.tif",
            stream.getvalue(),
        )
    rows = []
    with Source(path) as source:
        idrid_disc(source, lambda *a, **k: rows.append((a, k)), splits=("train",))
    np.testing.assert_array_equal(rows[0][0][3][0], label > 0)
    assert rows[0][1]["metadata"]["annotation_native_mode"] == "P"


def test_conflicting_transport_aliases_fail_closed(tmp_path):
    path = tmp_path / "DRIONS-DB.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("images/image_001.jpg", encoded(np.full((16, 20, 3), 100, np.uint8)))
        z.writestr("DRIONS-DB/images/image_001.jpg", encoded(np.full((16, 20, 3), 200, np.uint8)))
    with Source(path) as source, pytest.raises(ValueError, match="Conflicting"):
        drions(source, lambda *a, **kw: pytest.fail("conflicting photo emitted"))
