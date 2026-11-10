"""Synthetic encoded assets only: no GT decoding, NumPy, Torch or model forward."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from deepretina import val_bridge as v

PARENT = "a" * 64
TARGET = "b" * 64


def save(path, value):
    path.write_bytes(v.encoded(value))


def seal(root, canonical):
    raw = (root / "manifest.json").read_bytes()
    save(
        root / "integrity.json",
        {
            "fingerprint_schema": 2,
            "manifest_sha256": v.digest(raw),
            "prepared_content_sha256": canonical,
        },
    )
    (root / "_SUCCESS").write_text(v.digest(raw) + "\n")


def add(root, rows, photos, ident, split):
    row = {
        "id": ident + "-expert1",
        "photo_id": ident,
        "source_id": "source:" + ident,
        "split_group_id": "group:" + ident,
        "split": split,
        "dataset": "fixture",
        "task": "disc_cup",
        "image": "images/" + ident + ".png",
        "mask": "masks/" + ident + ".npz",
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25.0,
        "metadata": {
            "annotator": "expert1",
            "original_geometry": {
                "original_shape": [2, 3],
                "prepared_shape": [2, 3],
                "crop_position": [0, 1, 0, 2],
                "rescale_factor": 1.0,
            },
            "source_image": ident + ".jpg",
        },
    }
    photo = {
        "photo_id": ident,
        "split": split,
        "image": row["image"],
        "fov": "fovs/" + ident + ".png",
        "geometry": copy.deepcopy(row["metadata"]["original_geometry"]),
        "raw_rgb_sha256": ident,
        "annotations": [{"id": row["id"], "annotator": "expert1"}],
    }
    for rel in (row["image"], row["mask"], photo["fov"]):
        p = root / rel
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(("synthetic:" + rel).encode())
    rows.append(row)
    photos.append(photo)


@pytest.fixture
def case(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    rows = []
    photos = []
    for ident, split in [("p0", "val"), ("p1", "val"), ("tr", "train"), ("te", "test")]:
        add(old, rows, photos, ident, split)
    save(old / "manifest.json", rows)
    save(old / "photos.json", photos)
    seal(old, PARENT)
    new = tmp_path / "new"
    shutil.copytree(old, new)
    add(new, rows, photos, "new-train", "train")
    add(new, rows, photos, "new-official-test", "test")
    save(new / "manifest.json", rows)
    save(new / "photos.json", photos)
    seal(new, TARGET)
    ref = {
        "family": "original_reference",
        "split": "val",
        "preview_only": False,
        "photos": 2,
        "manifest_sha256": v.digest((old / "manifest.json").read_bytes()),
        "prepared_content_sha256": PARENT,
        "per_photo": [
            {
                "photo_id": pid,
                "record_ids": [pid + "-expert1"],
                "dataset": "fixture",
                "channel_datasets": {"disc": "fixture"},
                "channels": {"disc": {"tp": 1, "fp": 0, "fn": 0, "valid_pixels": 6}},
            }
            for pid in ("p0", "p1")
        ],
        "historical_policy": "untouched",
    }
    path = tmp_path / "measured.json"
    save(path, ref)
    return old, new, path, tmp_path / "bridge"


def build(case):
    old, new, ref, _ = case
    return v.build_bridge(
        old,
        new,
        ref,
        expected_parent_content=PARENT,
        expected_target_content=TARGET,
        expected_photos=2,
    )


def test_bridge_keeps_original_body_and_binds_explicit_val_context(case):
    _old, new, ref, out = case
    before = ref.read_bytes()
    bridge, wrapper = build(case)
    receipt = v.publish(out, bridge, wrapper)
    assert wrapper["original_measured_reference"] == json.loads(before)
    assert wrapper["family"] == v.WRAPPER and "per_photo" not in wrapper
    assert bridge["photo_count"] == 2 and bridge["test_bridge_allowed"] is False
    assert all(
        "new-" not in row["path"] and "/te." not in row["path"] for row in bridge["val_assets"]
    )
    resolved, proof = v.resolve_reference(
        out / "reference-for-target.json",
        out / "bridge.json",
        receipt["bridge_sha256"],
        ref,
        new,
        expected_photos=2,
    )
    assert resolved["manifest_sha256"] == v.digest((new / "manifest.json").read_bytes())
    assert (
        resolved["prepared_content_sha256"] == TARGET
        and resolved["per_photo"] == json.loads(before)["per_photo"]
    )
    assert proof["new_forward"] is False and ref.read_bytes() == before
    with pytest.raises(ValueError, match="new bridge output"):
        v.publish(out, bridge, wrapper)


@pytest.mark.parametrize(
    "change",
    [
        "GT",
        "RGB",
        "FOV",
        "geometry",
        "observer",
        "group",
        "photo",
        "membership",
        "additionalVAL",
        "photoindex",
        "assetpath",
    ],
)
def test_changed_domain_cannot_bridge(case, change):
    _old, new, _ref, _ = case
    rows = json.loads((new / "manifest.json").read_bytes())
    photos = json.loads((new / "photos.json").read_bytes())
    if change in {"GT", "RGB", "FOV"}:
        rel = {"GT": "masks/p0.npz", "RGB": "images/p0.png", "FOV": "fovs/p0.png"}[change]
        (new / rel).write_bytes(b"changed-encoded-asset")
    if change == "geometry":
        rows[0]["metadata"]["original_geometry"]["rescale_factor"] = 1.1
    if change == "observer":
        rows[0]["metadata"]["annotator"] = "expert2"
    if change == "group":
        rows[0]["split_group_id"] = "new-group"
    if change == "photo":
        rows[0]["photo_id"] = "changed-photo"
    if change == "membership":
        rows[0]["split"] = "test"
    if change == "additionalVAL":
        add(new, rows, photos, "extra-val", "val")
    if change == "photoindex":
        photos[0]["geometry"]["prepared_shape"] = [3, 3]
    if change == "assetpath":
        rows[0]["mask"] = "masks/te.npz"
    save(new / "manifest.json", rows)
    save(new / "photos.json", photos)
    seal(new, TARGET)
    with pytest.raises(ValueError):
        build(case)


@pytest.mark.parametrize("resource", ["asset", "metadata"])
def test_scope_rejects_symlinks(case, resource):
    _, new, _, out = case
    path = new / ("masks/p0.npz" if resource == "asset" else "photos.json")
    external = out.parent / "external"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="escaped|Symlink"):
        build(case)


def test_crosssplit_alias_and_test_measurement_are_rejected(case):
    _old, new, ref, _ = case
    rows = json.loads((new / "manifest.json").read_bytes())
    rows[-1]["source_id"] = rows[0]["source_id"]
    save(new / "manifest.json", rows)
    seal(new, TARGET)
    with pytest.raises(ValueError, match="crosses"):
        build(case)
    rows[-1]["source_id"] = "new-source"
    save(new / "manifest.json", rows)
    seal(new, TARGET)
    r = json.loads(ref.read_bytes())
    r["split"] = "test"
    save(ref, r)
    with pytest.raises(ValueError, match="VAL"):
        build(case)


def test_changed_observer_inventory_in_measured_reference_is_rejected(case):
    _, _, ref, _ = case
    r = json.loads(ref.read_bytes())
    r["per_photo"][0]["record_ids"] = ["wrong-expert"]
    save(ref, r)
    with pytest.raises(ValueError, match="observer"):
        build(case)


@pytest.mark.parametrize(
    "change",
    ["wrappercounts", "original", "bridge", "trusted_filename_hash", "targetGT", "canonical"],
)
def test_optin_requires_exact_external_trust_and_current_target_bytes(case, change):
    _, new, ref, out = case
    bridge, wrapper = build(case)
    receipt = v.publish(out, bridge, wrapper)
    trusted = receipt["bridge_sha256"]
    if change == "wrappercounts":
        wrapper["original_measured_reference"]["per_photo"][0]["channels"]["disc"]["tp"] = 0
        save(out / "reference-for-target.json", wrapper)
    if change == "original":
        ref.write_bytes(ref.read_bytes() + b" ")
    if change == "bridge":
        bridge["scope"] = "test"
        save(out / "bridge.json", bridge)
    if change == "trusted_filename_hash":
        trusted = hashlib.sha256(str(out / "bridge.json").encode()).hexdigest()
    if change == "targetGT":
        (new / "masks/p0.npz").write_bytes(b"changedafterbridge")
    if change == "canonical":
        seal(new, "c" * 64)
    with pytest.raises(ValueError):
        v.resolve_reference(
            out / "reference-for-target.json",
            out / "bridge.json",
            trusted,
            ref,
            new,
            expected_photos=2,
        )


def test_target_only_hashes_val_and_rejects_non770_production_context(case, monkeypatch):
    _, new, ref, out = case
    bridge, wrapper = build(case)
    receipt = v.publish(out, bridge, wrapper)
    hashed = []
    real = v._file_sha

    def scoped(path, *a):
        assert (
            "/te." not in str(path)
            and "new-official-test" not in str(path)
            and "new-train" not in str(path)
        )
        hashed.append(str(path))
        return real(path, *a)

    monkeypatch.setattr(v, "_file_sha", scoped)
    v.resolve_reference(
        out / "reference-for-target.json",
        out / "bridge.json",
        receipt["bridge_sha256"],
        ref,
        new,
        expected_photos=2,
    )
    assert hashed
    with pytest.raises(ValueError, match="complete VAL"):
        v.resolve_reference(
            out / "reference-for-target.json",
            out / "bridge.json",
            receipt["bridge_sha256"],
            ref,
            new,
        )


def test_metadata_number_type_change_is_not_exact_identity(case):
    _, new, _, _ = case
    rows = json.loads((new / "manifest.json").read_bytes())
    rows[0]["metadata"]["original_geometry"]["rescale_factor"] = 1
    save(new / "manifest.json", rows)
    seal(new, TARGET)
    with pytest.raises(ValueError, match="metadata changed"):
        build(case)


def test_whole_operation_detects_rewrite_of_already_hashed_asset(case, monkeypatch):
    old, _, _, _ = case
    real = v._file_sha
    changed = False

    def mutate(path, *args, **kwargs):
        nonlocal changed
        result = real(path, *args, **kwargs)
        if not changed and path == old / "images/p0.png":
            changed = True
            s = path.stat()
            path.write_bytes(path.read_bytes()[::-1])
            __import__("os").utime(path, ns=(s.st_atime_ns, s.st_mtime_ns))
        return result

    monkeypatch.setattr(v, "_file_sha", mutate)
    with pytest.raises(ValueError, match="changed"):
        build(case)


def test_strong_comparison_wrapper_requires_external_optin(case):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "completed_bridge_optin", Path(__file__).parents[1] / "scripts/compare_completed_val.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, new, _, out = case
    bridge, wrapper = build(case)
    v.publish(out, bridge, wrapper)
    dummy = out / "dummy.json"
    save(dummy, {})
    with pytest.raises(ValueError, match="external trusted bridge"):
        module.compare(
            out / "reference-for-target.json",
            dummy,
            new / "manifest.json",
            completion_qa_path=dummy,
            repeats=100,
        )


def test_generic_comparison_does_not_implicitly_unwrap_bridge(case):
    from deepretina.comparison import compare_reports

    _, new, ref, out = case
    bridge, wrapper = build(case)
    v.publish(out, bridge, wrapper)
    with pytest.raises((KeyError, ValueError)):
        compare_reports(out / "reference-for-target.json", ref, new / "manifest.json", repeats=100)
