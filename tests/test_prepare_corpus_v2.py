"""Parent immutability, alias holdouts and nominal camera provenance for v2."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_v2_under_test", Path(__file__).parents[1] / "scripts/prepare_corpus_v2.py"
)
v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v2
SPEC.loader.exec_module(v2)
POLICY = Path(__file__).parents[1] / "reports/data/corpus-v2-geometry-policy/policy.json"


def row(ident, source, photo, split, official=None):
    return {
        "id": ident,
        "source_id": source,
        "photo_id": photo,
        "split": split,
        "mask": ident + ".npz",
        "metadata": {"source_split": split, "official_split": official},
    }


def test_parent_heldout_alias_is_quarantined_without_row_mutation():
    parent = [row("p", "old:patient1", "photo1", "val")]
    pristine = copy.deepcopy(parent)
    new = row("n", "new:one", "photo1", "train")
    kept, excluded = v2.resolve_partitions(parent, copy.deepcopy(parent) + [new])
    assert kept == pristine == parent
    assert excluded[0]["id"] == "n" and excluded[0]["protected_split"] == "val"


def test_new_alias_cannot_join_two_parent_partitions():
    parent = [row("a", "old:a", "pa", "train"), row("b", "old:b", "pb", "test")]
    # Two captures in one new patient group bridge two immutable parent photos.
    new = [row("n1", "new:patient", "pa", "train"), row("n2", "new:patient", "pb", "train")]
    with pytest.raises(ValueError, match="multiple frozen"):
        v2.resolve_partitions(parent, copy.deepcopy(parent) + new)


def test_new_official_test_cannot_move_parent_training_photo():
    parent = [row("p", "old:one", "photo", "train")]
    with pytest.raises(ValueError, match="official heldout"):
        v2.resolve_partitions(
            parent, copy.deepcopy(parent) + [row("n", "new:one", "photo", "test", "test")]
        )


def test_new_train_alias_of_official_new_test_is_quarantined():
    train = row("a", "new:a", "same", "train")
    test = row("b", "new:b", "same", "test", "test")
    kept, excluded = v2.resolve_partitions([], [train, test])
    assert [r["id"] for r in kept] == ["b"] and kept[0]["split"] == "test"
    assert excluded[0]["id"] == "a"


def test_cloned_parent_has_no_shared_writable_inodes(tmp_path, monkeypatch):
    parent = tmp_path / "v1"
    parent.mkdir()
    for name, data in [
        ("manifest.json", "[]"),
        ("photos.json", "[]"),
        ("integrity.json", "{}"),
        ("_SUCCESS", "proof"),
    ]:
        (parent / name).write_text(data)
    (parent / "image").write_bytes(b"original")
    os.link(parent / "image", parent / "alias")
    verified = []
    monkeypatch.setattr(
        v2,
        "verify_prepared_content",
        lambda p, expected: verified.append((p, expected)) or expected,
    )
    child = tmp_path / "v2"
    v2.clone_parent(parent, child, "fixture")
    assert len(verified) == 2
    assert (parent / "image").stat().st_ino != (child / "image").stat().st_ino
    (child / "image").write_bytes(b"child edits")
    assert (parent / "image").read_bytes() == b"original"
    assert (parent / "_SUCCESS").exists() and not (child / "_SUCCESS").exists()
    assert (child / "parent-v1-metadata/manifest.json").read_bytes() == (
        parent / "manifest.json"
    ).read_bytes()


def test_policy_matches_exact_native_shape_and_remains_unmeasured():
    policy = json.loads(POLICY.read_text())
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == v2.GEOMETRY_POLICY_SHA256
    for dataset, shape, angle in [
        ("gamma", (1934, 1956, 3), 30),
        ("gamma", (2000, 2992, 3), 45),
        ("drions", (400, 600, 3), 30),
        ("biomisa", (1934, 2032, 3), 45),
        ("idrid_disc", (2848, 4288, 3), 50),
    ]:
        selected = v2.angle_for(policy, dataset, shape)
        assert selected["field_of_view_degrees"] == angle
        assert selected["per_image_angle_measured"] is False
        assert selected["native_shape_hw"] == list(shape[:2])
    fallback = v2.angle_for(policy, "gamma", (1956, 1934, 3))
    assert fallback["field_of_view_degrees"] == 45
    assert fallback["camera_attribution_method"] == "explicit_unseen_shape_nominal_fallback"
    with pytest.raises(ValueError, match="explicit reviewed"):
        v2.angle_for(policy, "unlisted", (1, 2, 3))


def test_unsplit_train_only_alias_is_quarantined():
    parent = [row("p", "old:one", "photo", "test")]
    new = row("n", "new:one", "photo", "train")
    new["metadata"].update(
        source_split="unsplit", split_policy="train_only_without_patient_crosswalk"
    )
    kept, excluded = v2.resolve_partitions(parent, copy.deepcopy(parent) + [new])
    assert kept == parent and excluded[0]["id"] == "n"


def test_tiny_full_append_preserves_parent_bytes_and_prepares_official_test(tmp_path, monkeypatch):
    import numpy as np

    parent = tmp_path / "parent"
    (parent / "images").mkdir(parents=True)
    (parent / "masks").mkdir()
    monkeypatch.setitem(v2.base.FIELDS, "drive", 0.8)
    writer = v2.base.UnifiedWriter(parent, tmp_path)
    rgb = np.full((16, 20, 3), 100, np.uint8)
    fov = np.ones((16, 20), bool)
    target = np.zeros((1, 16, 20), np.float32)
    target[:, 4:12, 6:10] = 1
    writer.emit(
        dataset="drive",
        stem="fixture",
        split="train",
        image=rgb,
        target=target,
        fov=fov,
        source_id="drive:fixture",
        task="vessels",
        metadata={"source_image": "fixture.png", "source_split": "train"},
    )
    writer.finalize()
    parent_audit = v2.base.audit_prepared(parent)
    original = json.loads((parent / "manifest.json").read_text())
    before = (parent / "manifest.json").read_bytes()
    policy = {
        "schema_version": 1,
        "policy_id": "synthetic_fixture_only",
        "policies": {
            name: {
                "field_of_view_degrees": 0.8,
                "nominal_assumption": True,
                "status": "synthetic_fixture_only",
            }
            for name in ["drions", "gamma", "idrid_disc", "biomisa"]
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    class Source:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def bytes(self, name):
            return (self.path.name + ":" + name).encode()

    monkeypatch.setattr(v2, "Source", Source)

    def adapter(source, emit, **kwargs):
        name = {
            "DRIONS-DB.zip": "drions",
            "GAMMA.zip": "gamma",
            "IDRiD.zip": "idrid_disc",
            "BIOMISA.zip": "biomisa",
        }[source.path.name]
        if name == "biomisa":
            assert kwargs["label_policy"] == v2.FIXED_POLICY
        color = {"drions": 110, "gamma": 120, "idrid_disc": 130, "biomisa": 140}[name]
        for split in ["train", "test"] if name == "idrid_disc" else ["train"]:
            image = np.full((16, 20, 3), color + (split == "test"), np.uint8)
            t = np.zeros((2, 16, 20), np.float32)
            t[0, 3:13, 4:16] = 1
            valid = np.zeros_like(t, dtype=bool)
            valid[0] = True
            stem = name + "_" + split
            emit(
                stem,
                split,
                image,
                t,
                valid=valid,
                source_id=name + ":" + split,
                task="disc_cup",
                metadata={
                    "source_image": stem + ".png",
                    "source_annotation": stem + "_mask.png",
                    "annotation_resources": [
                        {"path": stem + "_mask.png", "kind": "source_segmentation"}
                    ],
                    "source_split": split,
                    "official_split": split if name == "idrid_disc" else None,
                },
            )

    for name in ["drions", "gamma", "idrid_disc", "biomisa"]:
        monkeypatch.setattr(v2, name, adapter)
    data = tmp_path / "raw"
    data.mkdir()
    for name in ["DRIONS-DB.zip", "GAMMA.zip", "IDRiD.zip", "BIOMISA.zip"]:
        (data / name).write_bytes(b"fixture")
    out = tmp_path / "v2"
    result = v2.append(
        parent,
        out,
        data,
        policy_path,
        biomisa_policy=v2.FIXED_POLICY,
        expected_parent=parent_audit["prepared_content_sha256"],
        expected_policy=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )
    rows = json.loads((out / "manifest.json").read_text())
    assert rows[: len(original)] == original
    assert (parent / "manifest.json").read_bytes() == before
    assert len(rows) == 6 and [r["split"] for r in rows].count("test") == 1
    assert result["annotations_checked"] == 6
    assert (out / "geometry-policy.json").read_bytes() == policy_path.read_bytes()
    for r in rows[1:]:
        assert r["metadata"]["physical_calibration"]["nominal_assumption"] is True
        assert (
            r["metadata"]["policy_sha256"] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
        )
        with np.load(out / r["mask"]) as arrays:
            assert arrays["target"].shape == (2, 16, 20)
            assert arrays["valid"][0].all() and not arrays["valid"][1].any()
    with pytest.raises(ValueError, match="new independent"):
        v2.clone_parent(parent, out, expected=parent_audit["prepared_content_sha256"])
