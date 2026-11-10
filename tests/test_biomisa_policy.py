"""Fixed source filename roles must never become per-image area sorting."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from deepretina.adapters.biomisa import FIXED_POLICY, biomisa
from deepretina.sources import Source


def encoded(a, mode=None):
    f = io.BytesIO()
    Image.fromarray(a, mode=mode).save(f, format="PNG")
    return f.getvalue()


def fixture(path, missing_inner=False, wrong_nesting=False):
    raw = np.full((16, 20, 3), 100, np.uint8)
    outer = np.zeros((16, 20), np.uint8)
    outer[2:14, 3:17] = 255
    inner = np.zeros_like(outer)
    inner[5:10, 7:12] = 255
    if wrong_nesting:
        inner[0, 0] = 255
    root = "BIOMISA/OD/Healthy/P_1/Left Eye/"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(root + "123_Color_L_001.jpg", encoded(raw))
        z.writestr(root + "Fundus_Left_Healthy_Cup.jpg", encoded(outer))
        if not missing_inner:
            z.writestr(root + "Fundus_Left_Healthy_Disc.jpg", encoded(inner))
    return outer > 0, inner > 0


def test_quarantine_is_default_and_explicit_mapping_preserves_pixels(tmp_path):
    path = tmp_path / "BIOMISA.zip"
    outer, inner = fixture(path)
    rows = []
    with Source(path) as source:
        biomisa(source, lambda *a, **k: rows.append((a, k)))
    assert not rows[0][0][3].any() and not rows[0][1]["valid"].any()
    assert len(rows[0][1]["metadata"]["annotation_resources"]) == 2
    fixed = []
    with Source(path) as source:
        biomisa(source, lambda *a, **k: fixed.append((a, k)), label_policy=FIXED_POLICY)
    args, kw = fixed[0]
    np.testing.assert_array_equal(args[3][0], outer)
    np.testing.assert_array_equal(args[3][1], inner)
    assert kw["valid"].all() and args[1] == "train"
    assert kw["metadata"]["role_sorting_by_area"] is False
    assert kw["metadata"]["mask_values_or_geometry_modified"] is False
    assert kw["source_id"] == "biomisa:Healthy:P_1"


def test_missing_named_disc_becomes_disc_only_not_negative_cup(tmp_path):
    path = tmp_path / "BIOMISA.zip"
    outer, _ = fixture(path, missing_inner=True)
    rows = []
    with Source(path) as source:
        biomisa(source, lambda *a, **k: rows.append((a, k)), label_policy=FIXED_POLICY)
    np.testing.assert_array_equal(rows[0][0][3][0], outer)
    assert rows[0][1]["valid"][0].all() and not rows[0][1]["valid"][1].any()


def test_policy_violation_fails_without_sorting_or_clipping(tmp_path):
    path = tmp_path / "BIOMISA.zip"
    fixture(path, wrong_nesting=True)
    with Source(path) as source, pytest.raises(ValueError, match="no per-image sorting"):
        biomisa(
            source,
            lambda *a, **k: pytest.fail("contradictory source labels emitted"),
            label_policy=FIXED_POLICY,
        )
