"""Secondary observers, disc-only annotations and external source overrides."""

import io
import json
import zipfile

import numpy as np
from PIL import Image

from deepretina.prepare import decode_iostar_disc, prepare


def png(array):
    data = io.BytesIO()
    Image.fromarray(array).save(data, format="PNG")
    return data.getvalue()


def test_drive_secondary_observer_is_retained_without_duplicate_photo(tmp_path):
    image = np.full((24, 32, 3), 100, np.uint8)
    first = np.zeros((24, 32), np.uint8)
    first[8:10, 4:28] = 255
    second = np.roll(first, 2, axis=0)
    with zipfile.ZipFile(tmp_path / "DRIVE.zip", "w") as source:
        source.writestr("test/images/01_test.tif", png(image))
        source.writestr("test/1st_manual/01_manual1.gif", png(first))
        source.writestr("test/2nd_manual/01_manual2.gif", png(second))
        source.writestr("test/mask/01_test_mask.gif", png(np.full((24, 32), 255, np.uint8)))
    manifest = prepare(tmp_path, tmp_path / "prepared", ["drive"], max_size=32)
    rows = json.loads(manifest.read_text())
    assert len(rows) == 2
    assert {r["source_id"] for r in rows} == {"drive:01"}
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["image"] for r in rows}) == 1
    assert {r["metadata"]["annotator"] for r in rows} == {"first", "second"}
    targets = [np.load(manifest.parent / r["mask"])["target"][0] for r in rows]
    assert not np.array_equal(*targets)


def test_chase_both_observers_share_eye_pair_identity(tmp_path):
    image = np.full((24, 32, 3), 100, np.uint8)
    mask = np.zeros((24, 32), np.uint8)
    mask[8:10, 4:28] = 255
    with zipfile.ZipFile(tmp_path / "CHASEDB1.zip", "w") as source:
        for eye in ("L", "R"):
            stem = f"Image_01{eye}"
            source.writestr(f"CHASEDB1/{stem}.jpg", png(image))
            source.writestr(f"CHASEDB1/{stem}_1stHO.png", png(mask))
            source.writestr(f"CHASEDB1/{stem}_2ndHO.png", png(np.roll(mask, 1, axis=0)))
    manifest = prepare(tmp_path, tmp_path / "prepared", ["chasedb1"], max_size=32)
    rows = json.loads(manifest.read_text())
    assert len(rows) == 4
    assert len({r["source_id"] for r in rows}) == 1
    assert len({r["image"] for r in rows}) == 2


def test_iostar_disc_ignores_all_border_components_and_masks_missing_cup(tmp_path):
    image = np.full((24, 32, 3), 100, np.uint8)
    fov = np.full((24, 32), 255, np.uint8)
    vessel = np.zeros_like(fov)
    vessel[8:10, 4:28] = 255
    av = np.zeros_like(image)
    av[vessel > 0, 0] = 255
    od = np.ones((24, 32), bool)
    od[0] = False  # One component touching the top border.
    od[-1] = False  # A separate component touching the bottom border.
    od[10:15, 12:18] = False  # The enclosed optic disc.
    disc = decode_iostar_disc(od)
    assert disc.sum() == 30
    assert not disc[[0, -1]].any()
    with zipfile.ZipFile(tmp_path / "IOSTAR.zip", "w") as source:
        for path, array in (
            ("image/STAR01.jpg", image),
            ("GT/STAR01_GT.tif", vessel),
            ("mask/STAR01_Mask.tif", fov),
            ("AV_GT/STAR01_AV.tif", av),
            ("mask_OD/STAR01_ODMask.tif", od),
        ):
            source.writestr(path, png(array))
    manifest = prepare(tmp_path, tmp_path / "prepared", ["iostar"], max_size=32)
    rows = json.loads(manifest.read_text())
    assert {r["task"] for r in rows} == {"vessels", "av", "disc_cup"}
    assert len({r["source_id"] for r in rows}) == 1
    assert len({r["image"] for r in rows}) == 1
    row = next(r for r in rows if r["task"] == "disc_cup")
    with np.load(manifest.parent / row["mask"]) as labels:
        np.testing.assert_array_equal(labels["target"][0] > 0, disc)
        assert not labels["valid"][1].any()
    assert row["supervised_channels"] == [True, False]


def test_ubmi_source_override_copies_prediction_resource_without_fake_ground_truth(tmp_path):
    originals = tmp_path / "private"
    originals.mkdir()
    image = np.full((24, 32, 3), 100, np.uint8)
    Image.fromarray(image).save(originals / "Gacr_01_017_01_L.jpg")
    masks = originals / "ImageAnalysis/VesselsSeg"
    masks.mkdir(parents=True)
    Image.fromarray(np.full((24, 32), 255, np.uint8)).save(masks / "samplewhole.png")
    manifest = prepare(
        tmp_path / "missing-default-root",
        tmp_path / "prepared",
        ["ubmi"],
        max_size=32,
        limit=1,
        source_overrides={"ubmi": originals},
    )
    (row,) = json.loads(manifest.read_text())
    assert row["task"] == "images"
    assert "mask" not in row
    assert not row["usable_for_training"]
    assert row["preparation_preview"]
    refs = row["metadata"]["annotation_resources"]
    assert refs[0]["kind"] == "existing_vessel_segmentation_for_inference"
    assert (manifest.parent / refs[0]["prepared_path"]).is_file()
