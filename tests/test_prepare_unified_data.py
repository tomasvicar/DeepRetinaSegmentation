"""Canonical physical frames, multi-annotator identity and split-safety regressions."""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "unified_prepare_under_test", Path(__file__).parents[1] / "scripts/prepare_unified_data.py"
)
prep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prep
SPEC.loader.exec_module(prep)


def writer(tmp_path):
    out = tmp_path / "prepared"
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    return prep.UnifiedWriter(out, tmp_path, pixels_per_degree=1)


def photo():
    yy, xx = np.indices((40, 60))
    return np.stack([80 + xx, 70 + yy, 90 + (xx + yy) // 2], -1).astype(np.uint8)


def test_exact_drive_rite_alias_uses_one_canonical_grid_and_preserves_variants(tmp_path):
    w = writer(tmp_path)
    rgb = photo()
    fov = np.ones(rgb.shape[:2], bool)
    vessel = np.zeros(rgb.shape[:2], np.float32)
    vessel[8:30, 20:24] = 1
    first = w.emit(
        dataset="drive",
        stem="21_training",
        split="train",
        image=rgb,
        target=vessel[None],
        fov=fov,
        source_id="drive:21",
        metadata={"source_image": "training/images/21_training.tif", "annotator": "first"},
    )
    av = np.stack([vessel, vessel, np.zeros_like(vessel)])
    valid = np.stack([fov, vessel > 0, vessel > 0])
    second = w.emit(
        dataset="rite",
        stem="21_training",
        split="train",
        image=rgb.copy(),
        target=av,
        valid=valid,
        fov=None,
        source_id="drive:21",
        task="av",
        metadata={"source_image": "training/images/21_training.tif"},
    )
    assert first["photo_id"] == second["photo_id"] and first["image"] == second["image"]
    assert first["metadata"]["original_geometry"] == second["metadata"]["original_geometry"]
    assert "preparation_max_size" not in first
    w.finalize()
    photos = json.loads((w.output / "photos.json").read_text())
    assert (
        len(photos) == 1 and len(photos[0]["annotations"]) == 2 and len(photos[0]["sources"]) == 2
    )
    assert {r["metadata"]["official_split"] for r in w.rows} == {"train"}
    assert (w.output / "_SUCCESS").is_file()
    audit = prep.audit_prepared(w.output)
    assert audit["photos_checked"] == 1 and audit["annotations_checked"] == 2
    artery = next(r for r in audit["head_counts"] if r["channel"] == "artery")
    assert artery["photos"] == 1 and artery["positive_photos"] == 1
    data = np.load(w.output / second["mask"])
    target, valid, fov = data["target"], data["valid"], data["fov"]
    data.close()
    target[1][valid[1]] = 0
    np.savez_compressed(w.output / second["mask"], target=target, valid=valid, fov=fov)
    with pytest.raises(ValueError, match="conditional AV support"):
        prep.audit_prepared(w.output)


def test_missing_cup_and_positive_only_vessels_stay_unknown(tmp_path):
    w = writer(tmp_path)
    rgb = photo()
    mask = np.zeros(rgb.shape[:2], np.float32)
    mask[10:30, 20:40] = 1
    target = np.stack([mask, np.zeros_like(mask)])
    valid = np.ones_like(target, bool)
    valid[1] = False
    row = w.emit(
        dataset="g1020",
        stem="x",
        split="unsplit",
        image=rgb,
        target=target,
        valid=valid,
        source_id="g1020:x",
        task="disc_cup",
        metadata={"source_image": "x.jpg"},
    )
    with np.load(w.output / row["mask"]) as data:
        assert not data["valid"][1].any() and not data["target"][1].any()
    vessel = np.zeros_like(mask)
    vessel[:, 30:32] = 1
    row = w.emit(
        dataset="avrdb",
        stem="x",
        split="unsplit",
        image=rgb,
        target=vessel[None],
        valid=(vessel > 0)[None],
        source_id="avrdb:x",
        metadata={"source_image": "IM/x.jpg"},
    )
    with np.load(w.output / row["mask"]) as data:
        assert data["valid"].any()
        assert (data["target"][data["valid"]] == 1).all()
    rejected = w.emit(
        dataset="riga",
        stem="bad",
        split="unsplit",
        image=rgb,
        target=np.zeros_like(target),
        valid=np.zeros_like(valid),
        source_id="riga:bad",
        task="disc_cup",
        metadata={"source_image": "badprime.tif", "quality": {"requires_review": True}},
    )
    assert rejected is None and "quarantined" in w.excluded[-1]["reason"]


def test_patient_and_photo_alias_union_cannot_leak_official_test(tmp_path):
    del tmp_path
    rows = [
        {
            "id": "a",
            "dataset": "drive",
            "source_id": "drive:01",
            "photo_id": "p",
            "split": "test",
            "metadata": {"official_split": "test"},
        },
        {
            "id": "b",
            "dataset": "hrf",
            "source_id": "hrf:01",
            "photo_id": "p",
            "split": "train",
            "metadata": {"official_split": "train"},
        },
        {
            "id": "c",
            "dataset": "hrf",
            "source_id": "hrf:01",
            "photo_id": "other_eye",
            "split": "unsplit",
            "metadata": {},
        },
    ]
    conflicts = prep.safe_splits(rows, 42)
    assert {r["split"] for r in rows} == {"test"}
    assert len({r["split_group_id"] for r in rows}) == 1
    assert conflicts and conflicts[0]["official_splits"] == ["test", "train"]


def encode(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def test_raw_reader_hook_prepares_portable_paired_core_manifest_without_cap(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    rgb = photo()
    fov = np.full(rgb.shape[:2], 255, np.uint8)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[8:30, 20:24] = 255
    av = np.zeros_like(rgb)
    av[mask > 0] = [255, 0, 0]
    with zipfile.ZipFile(data / "DRIVE.zip", "w") as z:
        z.writestr("DRIVE/training/images/21_training.tif", encode(rgb))
        z.writestr("DRIVE/training/1st_manual/21_manual1.gif", encode(mask))
        z.writestr("DRIVE/training/mask/21_training_mask.gif", encode(fov))
    with zipfile.ZipFile(data / "RITE.zip", "w") as z:
        z.writestr("RITE/AV_groundTruth/training/images/21_training.tif", encode(rgb))
        z.writestr("RITE/AV_groundTruth/training/vessel/21_training.png", encode(mask))
        z.writestr("RITE/AV_groundTruth/training/av/21_training.png", encode(av))
    out = tmp_path / "canonical"
    report = prep.prepare_unified_data(data, out, ["drive", "rite"], pixels_per_degree=1)
    rows = json.loads((out / "manifest.json").read_text())
    assert report["photos"] == 1 and report["annotations"] == 2
    assert rows[0]["photo_id"] == rows[1]["photo_id"]
    assert all(
        r["pixels_per_degree"] == 1 and r["preprocessing_profile"] == "legacy1_raw" for r in rows
    )
    assert all("preparation_max_size" not in r for r in rows)
    assert json.loads((out / "preparation.json").read_text())["max_size"] is None
    for row in rows:
        with Image.open(out / row["image"]) as im, np.load(out / row["mask"]) as labels:
            assert labels["target"].shape[1:] == (im.height, im.width)
            assert labels["valid"].shape == labels["target"].shape
            assert labels["target"].dtype == np.float32 and labels["valid"].dtype == bool
        assert row["metadata"]["official_split"] == "train"
