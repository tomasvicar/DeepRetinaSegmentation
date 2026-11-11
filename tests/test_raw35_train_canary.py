"""Synthetic encoded resources and probabilities only; no real-data/model tests."""

import importlib.util
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from deepretina.legacy.preprocessing import ImageGeometry
from deepretina.sources import Source

spec = importlib.util.spec_from_file_location(
    "raw35_canary", Path(__file__).parents[1] / "scripts/prepare_raw35_train_canary.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_resource_filter_is_applied_before_core_enumeration_or_pixel_decode(tmp_path):
    path = tmp_path / "source.zip"
    image = np.ones((6, 8, 3), np.uint8) * 100
    entries = {}
    for name, array, fmt in [
        ("training/images/37_training.tif", image, "TIFF"),
        ("training/1st_manual/37_manual1.gif", image[:, :, 0], "GIF"),
        ("training/mask/37_training_mask.gif", np.full((6, 8), 255, np.uint8), "GIF"),
    ]:
        blob = io.BytesIO()
        Image.fromarray(array).save(blob, format=fmt)
        entries[name] = blob.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in entries.items():
            archive.writestr(name, blob)
        archive.writestr("test/images/01_test.tif", b"Invalid image: decoding this is forbidden")
    with Source(path) as source:
        limited = m.FilteredSource(source, entries)
        calls = []
        m.decode_core("drive", limited, lambda *a, **k: calls.append(a))
        assert len(calls) == 1 and calls[0][1] == "train"
        assert {x["resource"] for x in limited.reads} == set(entries)
        assert limited.images("test/images", ".tif") == []
        with pytest.raises(ValueError, match="allowlist"):
            limited.read("test/images/01_test.tif", "RGB")
        assert len(limited.reads) == 3


def test_every_emission_hashes_RGB_even_if_resource_was_previously_memoized(tmp_path):
    pid = next(iter(m.PHOTO_IDS))
    rows = [{"id": "drive_37_training_vessels", "photo_id": pid, "split": "train"}]
    rows += [{"id": str(i), "photo_id": pid, "split": "train"} for i in range(6)]
    replay = m.Replay(tmp_path, rows, [{"photo_id": p} for p in m.PHOTO_IDS], tmp_path)
    replay.resource_rgb[("drive", "training/images/37_training.tif")] = pid.split(":")[-1]
    with pytest.raises(AssertionError, match="emission RGB"):
        replay.emit(
            "drive",
            "37_training",
            "train",
            np.zeros((2, 3, 3), np.uint8),
            np.ones((1, 2, 3), np.float32),
            metadata={"source_image": "training/images/37_training.tif"},
        )
    assert not replay.seen and not replay.prepared


def test_noncanonical_alias_cannot_take_first_geometry_ownership(tmp_path, monkeypatch):
    image = np.ones((2, 3, 3), np.uint8)
    pid = "sha256:" + m.rgb_hash(image)
    ids = {pid, "sha256:other1", "sha256:other2"}
    monkeypatch.setattr(m, "PHOTO_IDS", ids)
    metadata = {"source_image": "RITE/training/images/37_training.tif", "source_split": "train"}
    row = {
        "id": "rite_37_training_av",
        "photo_id": pid,
        "split": "train",
        "source_id": "drive:37",
        "metadata": metadata,
    }
    rows = [row] + [{"id": str(i), "photo_id": pid, "split": "train"} for i in range(6)]
    photos = [
        {
            "photo_id": p,
            "canonical_dataset": "drive",
            "sources": [{"dataset": "drive", "resource": "training/images/37_training.tif"}],
        }
        for p in ids
    ]
    replay = m.Replay(tmp_path, rows, photos, tmp_path)

    def forbidden(*a, **k):
        raise RuntimeError("An alias must not prepare geometry")

    monkeypatch.setattr(m, "prepare_single_net", forbidden)
    with pytest.raises(AssertionError):
        replay.emit(
            "rite",
            "37_training",
            "train",
            image,
            np.ones((3, 2, 3), np.float32),
            source_id="drive:37",
            task="av",
            metadata=metadata,
        )
    assert not replay.prepared


@pytest.mark.parametrize("scale,shape", [(0.5, (3, 4)), (0.75, (4, 6))])
def test_nearest_projection_preserves_actual_soft_values_and_missing_channels(scale, shape):
    target = np.full((2, 6, 8), 0.37, np.float32)
    valid = np.ones_like(target, bool)
    valid[1] = False
    geometry = ImageGeometry((6, 8), (0, 5, 0, 7), scale, shape)
    prepared = SimpleNamespace(geometry=geometry, fov=np.ones(shape, bool))
    actual, allowed = m.project(prepared, target, valid, None)
    assert actual.dtype == np.float32 and np.all(actual == np.float32(0.37))
    assert allowed[0].all() and not allowed[1].any()
    assert m.channels(actual, allowed)[1]["valid_positive_pixels"] == 0


def prepared(shape, ppd, crop=(0, 0, 0, 1)):
    return SimpleNamespace(
        geometry=ImageGeometry((1, 2), crop, 1, shape),
        fov=np.ones(shape, bool),
        field_of_view_degrees=45,
        pixels_per_degree=ppd,
    )


def test_common_grid_regrids_joint_AV_not_conditional_class_probabilities():
    values = np.zeros((5, 1, 2), np.float32)
    values[0] = [0.1, 0.9]
    values[1] = [0.9, 0.1]
    values[2] = [0.1, 0.9]
    mapped = m.regrid35_to25(values, prepared((1, 2), 35), prepared((1, 1), 25))
    assert mapped.shape == (5, 1, 1) and mapped[0, 0, 0] == pytest.approx(0.5)
    assert mapped[1, 0, 0] == pytest.approx(0.18, abs=1e-6)
    assert mapped[2, 0, 0] == pytest.approx(0.82, abs=1e-6)
    with pytest.raises(ValueError, match="geometry"):
        m.regrid35_to25(values, prepared((1, 2), 35), prepared((1, 1), 25, crop=(0, 0, 0, 0)))


def test_AV_conditional_support_excludes_background_unknown_and_crossings():
    target = np.array([[[0, 1, 1, 1]], [[0, 1, 0, 1]], [[0, 0, 0, 1]]], dtype=np.float32)
    valid = np.ones_like(target, bool)
    valid[1:, 0, 2] = False
    actual = m.av_supports(target, valid)
    assert actual == {
        "AV_class_validity_pixels_including_background": 3,
        "AV_conditional_scoring_pixels": 1,
        "AV_unknown_vessel_pixels": 1,
    }
