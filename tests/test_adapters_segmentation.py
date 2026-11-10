"""Annotation semantics and geometry regressions for legacy adapters."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from deepretina.adapters.segmentation import ADAPTERS, decode_g1020, decode_onhsd, recover_riga


def test_g1020_missing_cup_is_not_background_supervision():
    target, valid, _notes = decode_g1020(
        {"shapes": [{"label": "disc", "points": [[2, 2], [8, 2], [8, 8], [2, 8]]}]}, (12, 12)
    )
    assert target[0, 2, 2] == 1  # LabelMe coordinates stay zero based.
    assert valid[0].all() and not valid[1].any()
    assert not target[1].any()


def test_g1020_duplicate_disc_labels_resolve_by_nested_geometry():
    shapes = [
        {"label": "disc", "points": [[4, 4], [6, 4], [6, 6], [4, 6]]},
        {"label": "disc", "points": [[2, 2], [8, 2], [8, 8], [2, 8]]},
    ]
    target, valid, notes = decode_g1020({"shapes": shapes}, (12, 12))
    assert target[0].sum() > target[1].sum() > 0
    assert valid.all()
    assert any("relabelled" in note for note in notes)
    shapes[0]["points"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    target, valid, notes = decode_g1020({"shapes": shapes}, (12, 12))
    assert not valid.any()
    assert any("requires_review" in note for note in notes)


def test_onhsd_matlab_center_becomes_zero_based():
    mask = decode_onhsd([11, 11], np.full(24, 4), (24, 24))
    assert mask[10, 10]
    assert mask[10, 14] and mask[10, 6]
    assert not mask[10, 15]
    with pytest.raises(ValueError, match="positive"):
        decode_onhsd([11, 11], np.zeros(24), (24, 24))


def test_riga_recovers_nested_outlines_and_refuses_open_curves():
    prime = np.full((100, 100, 3), 150, dtype=np.uint8)
    marked = Image.fromarray(prime)
    draw = ImageDraw.Draw(marked)
    draw.ellipse((15, 15, 85, 85), outline=(255, 0, 0), width=2)
    draw.ellipse((35, 35, 65, 65), outline=(255, 0, 0), width=2)
    disc, cup, metadata = recover_riga(prime, np.asarray(marked))
    assert disc.sum() > cup.sum() > 0
    assert not (cup & ~disc).any()
    assert cup[50, 50] and disc[20, 50]
    assert "closed" in metadata["method"]
    with pytest.raises(ValueError, match="closed contours"):
        recover_riga(prime, prime)


class MemorySource:
    def __init__(self, images):
        self.images = images
        self.names = list(images)

    def find(self, suffix, optional=False):
        if suffix in self.images:
            return suffix
        matches = [p for p in self.images if p.lower().endswith("/" + suffix.lower())]
        if matches:
            return matches[0]
        if optional:
            return None
        raise FileNotFoundError(suffix)

    def read(self, path, mode="L"):
        return np.asarray(Image.fromarray(self.images[self.find(path)]).convert(mode))


def test_uoadr72_swap_and_timestamp_ignored():
    rgb = np.full((100, 100, 3), 100, dtype=np.uint8)
    disc = np.zeros_like(rgb)
    disc[20:40, 20:40] = 255
    vessel = np.zeros_like(rgb)
    vessel[10:80, 50:52] = 255
    center_marker = np.zeros_like(rgb)
    center_marker[25:35, 25:35] = 255
    source = MemorySource(
        {"72/72.jpg": rgb, "72/72.1.jpg": disc, "72/72.2.jpg": vessel, "72/72.3.jpg": center_marker}
    )
    records = []
    ADAPTERS["uoadr"](source, lambda *a, **kw: records.append((a, kw)))
    assert len(records) == 2
    np.testing.assert_array_equal(records[0][0][3][0], vessel[..., 0] > 127)
    np.testing.assert_array_equal(records[1][0][3][0], disc[..., 0] > 127)
    assert not records[0][1]["fov"][97:, :25].any()
    assert records[0][1]["source_id"] == records[1][1]["source_id"]
    assert not records[1][1]["valid"][1].any()
    assert not records[1][0][3][1].any()
    assert records[1][1]["metadata"]["optic_disc_center_marker_centroid_xy"] == [29.5, 29.5]
    center_ref = records[1][1]["metadata"]["annotation_resources"][1]
    assert center_ref["kind"] == "optic_disc_center" and center_ref["preserve"]


def test_avrdb_sparse_lines_have_no_unmarked_negative_supervision():
    image = np.full((24, 24, 3), 100, np.uint8)
    vessel = np.full_like(image, 255)
    vessel[4:20, 12] = 0
    artery = np.full_like(image, 255)
    artery[4:12, 12] = [237, 27, 36]
    vein = np.full_like(image, 255)
    vein[12:20, 12] = [45, 49, 146]
    source = MemorySource(
        {
            "IM000001/IM000001.JPG": image,
            "IM000001/IM000001--vessels.jpg": vessel,
            "IM000001/IM000001--atertries.jpg": artery,
            "IM000001/IM000001--veinds.jpg": vein,
        }
    )
    records = []
    ADAPTERS["avrdb"](source, lambda *a, **kw: records.append((a, kw)))
    assert len(records) == 2
    binary, av = records
    assert binary[1]["valid"].sum() == 16
    assert not av[1]["valid"][:, 0, 0].any()
    assert av[1]["valid"][0].sum() == 16
    assert av[1]["valid"][1].sum() == 16
    assert av[0][3][1].sum() == av[0][3][2].sum() == 8
    assert len(av[1]["metadata"]["annotation_resources"]) == 2


def test_rimone_keeps_stereo_right_half_unsupervised_and_observers_grouped():
    image = np.full((32, 64, 3), 100, np.uint8)
    disc = np.zeros((32, 64), np.uint8)
    disc[8:24, 8:24] = 255
    cup = np.zeros_like(disc)
    cup[12:20, 12:20] = 255
    files = {"Healthy/Stereo Images/N-1-L.jpg": image}
    for folder, prefix, suffix in [
        ("Average_masks", "N-1-L", "-Avg"),
        ("Expert1_masks", "N-1-L-1", "-exp1"),
        ("Expert2_masks", "N-1-L-1", "-exp2"),
    ]:
        files[f"Healthy/{folder}/{prefix}-Disc{suffix}.png"] = disc
        files[f"Healthy/{folder}/{prefix}-Cup{suffix}.png"] = cup
    records = []
    ADAPTERS["rimone"](MemorySource(files), lambda *a, **kw: records.append((a, kw)))
    assert len(records) == 3
    assert {kw["source_id"] for _, kw in records} == {"N-1"}
    for args, kw in records:
        assert kw["metadata"]["source_image"] == "Healthy/Stereo Images/N-1-L.jpg"
        assert not kw["fov"][:, 32:].any()
        assert len(kw["metadata"]["annotation_resources"]) == 2
        np.testing.assert_array_equal(args[3][0], disc > 127)


def test_riga_bounded_gap_repair_preserves_closed_outline_area():
    prime = np.full((120, 120, 3), 150, dtype=np.uint8)
    marked = Image.fromarray(prime)
    draw = ImageDraw.Draw(marked)
    draw.ellipse((15, 15, 105, 105), outline=(255, 0, 0), width=1)
    draw.ellipse((40, 40, 80, 80), outline=(255, 0, 0), width=1)
    clean = np.asarray(marked).copy()
    reference_disc, reference_cup, _ = recover_riga(prime, clean)
    broken = clean.copy()
    broken[22, 36] = prime[22, 36]
    disc, cup, metadata = recover_riga(prime, broken)
    assert metadata["closing_diamond_radius"] > 0
    assert (disc & reference_disc).sum() / (disc | reference_disc).sum() > 0.98
    assert (cup & reference_cup).sum() / (cup | reference_cup).sum() > 0.98
    # A large missing arc is not repaired by the bounded morphology fallback.
    broken[12:24, 40:80] = prime[12:24, 40:80]
    with pytest.raises(ValueError, match="closed contours"):
        recover_riga(prime, broken)
