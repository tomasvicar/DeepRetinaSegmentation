"""Synthetic only: never open a real dataset, model or ownCloud resource."""

import copy
import io
import json

import numpy as np
import pytest
from PIL import Image

from deepretina.adapters import segmentation
from deepretina.integrity import verify_prepared_content
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.prepare import decoded_photo_id
from deepretina.raw35_producer import (
    FilteredSource,
    FrozenPlan,
    Replay,
    dependencies,
    normalized,
    preserve_support,
    project,
    resolve_requests,
    row_id,
    safe,
    validate_profile,
)


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "canonical25"
    root.mkdir()
    for folder in ("images", "fovs", "masks"):
        (root / folder).mkdir()
    image = np.full((10, 10, 3), 140, np.uint8)
    image[2:8, 2:8, 1] = 180
    native_fov = np.ones((10, 10), bool)
    native_fov[:, 8:] = False
    target = np.zeros((2, 10, 10), np.float32)
    target[0, 2:7, 2:7] = 0.25
    target[1, 3:5, 3:5] = 0.75
    valid = np.ones_like(target, bool)
    valid[1, :, :3] = False
    p25 = prepare_single_net(
        image,
        dataset="rimone",
        fov=native_fov,
        field_of_view_degrees=0.4,
        pixels_per_degree=25,
        normalized=False,
    )
    pid = decoded_photo_id(image)
    photo = {
        "photo_id": pid,
        "image": "images/native.png",
        "fov": "fovs/native.png",
        "raw_rgb_sha256": pid[7:],
        "raw_shape": list(image.shape),
        "geometry": normalized(p25.geometry.as_dict()),
        "prepared_shape": list(p25.image.shape),
        "canonical_dataset": "rimone",
        "field_of_view_degrees": 0.4,
        "pixels_per_degree": 25.0,
        "fov_source": p25.fov_source,
        "split": "train",
        "sources": [{"dataset": "rimone", "resource": "rgb.jpg"}],
        "annotations": [],
    }
    ident = row_id("rimone", "sample_average", "disc_cup")
    m = {"source_image": "rgb.jpg", "source_split": "unsplit", "annotator": "average"}
    row = {
        "id": ident,
        "dataset": "rimone",
        "split": "train",
        "source_id": "rimone:sample",
        "photo_id": pid,
        "image": "images/alias.png",
        "mask": "masks/row.npz",
        "task": "disc_cup",
        "pixels_per_degree": 25.0,
        "preprocessing_profile": "legacy25_raw",
        "split_group_id": "patient:frozen",
        "metadata": m,
    }
    photo["annotations"] = [{"id": ident, "mask": row["mask"]}]
    Image.fromarray(p25.image).save(root / photo["image"])
    Image.fromarray(p25.image).save(root / row["image"])
    Image.fromarray(p25.fov.astype(np.uint8) * 255).save(root / photo["fov"])
    t, v = project(p25, target, valid, native_fov)
    np.savez_compressed(root / row["mask"], target=t, valid=v, fov=p25.fov)
    output = tmp_path / "new35"
    output.mkdir()
    return root, output, image, target, valid, native_fov, row, photo


def emit(replay, image, target, valid, fov, **kwargs):
    replay.emit(
        "rimone",
        "sample_average",
        "unsplit",
        image,
        target,
        valid=valid,
        fov=fov,
        source_id="sample",
        task="disc_cup",
        metadata={"source_image": "rgb.jpg", "annotator": "average"},
        **kwargs,
    )


def test_replay_soft_partial_identity_and_truthful_fingerprints(frozen):
    root, out, image, target, valid, fov, row, photo = frozen
    replay = Replay(root, FrozenPlan([row], [photo]), out)
    emit(replay, image, target, valid, fov)
    fingerprint = replay.finish({"pixels_per_degree": 35})
    assert verify_prepared_content(out / "manifest.json") == fingerprint
    actual = json.loads((out / "manifest.json").read_text())[0]
    for key in ("id", "source_id", "photo_id", "split", "split_group_id", "image", "mask"):
        assert actual[key] == row[key]
    assert actual["pixels_per_degree"] == 35
    p = json.loads((out / "photos.json").read_text())[0]
    assert p["geometry"] != photo["geometry"]
    assert p["sources"] == photo["sources"]
    with np.load(out / row["mask"]) as labels:
        assert labels["target"].dtype == np.float32
        assert set(np.unique(labels["target"])) == {0.0, 0.25, 0.75}
        assert not labels["valid"][1].all()
        assert np.array_equal(
            labels["valid"] & ~labels["fov"][None], np.zeros_like(labels["valid"])
        )


@pytest.mark.parametrize("field", ["RGB", "target", "valid", "fov"])
def test_every25_component_must_replay_before35_write(frozen, field):
    root, out, image, target, valid, fov, row, photo = frozen
    if field == "RGB":
        Image.fromarray(np.zeros((10, 8, 3), np.uint8)).save(root / row["image"])
    else:
        with np.load(root / row["mask"]) as old:
            arrays = {k: old[k].copy() for k in old.files}
        arrays[field].flat[0] = 1 - arrays[field].flat[0]
        np.savez_compressed(root / row["mask"], **arrays)
    replay = Replay(root, FrozenPlan([row], [photo]), out)
    with pytest.raises(ValueError, match="RAW25"):
        emit(replay, image, target, valid, fov)
    assert list(out.iterdir()) == []


def test_every_emission_rgb_hashed_including_repeated_resource(frozen):
    root, out, image, target, valid, fov, row, photo = frozen
    second = copy.deepcopy(row)
    second["id"] = row_id("rimone", "sample_expert1", "disc_cup")
    second["metadata"]["annotator"] = "expert1"
    replay = Replay(root, FrozenPlan([row, second], [photo]), out)
    emit(replay, image, target, valid, fov)
    altered = image.copy()
    altered[0, 0, 0] += 1
    with pytest.raises(ValueError, match="emission RGB"):
        replay.emit(
            "rimone",
            "sample_expert1",
            "unsplit",
            altered,
            target,
            valid=valid,
            fov=fov,
            source_id="sample",
            task="disc_cup",
            metadata={"source_image": "rgb.jpg", "annotator": "expert1"},
        )


def test_owner_order_split_and_identity_cannot_change(frozen):
    root, out, image, target, _valid, _fov, row, photo = frozen
    alias = copy.deepcopy(row)
    alias.update(id="erlangen_alias_vessels", dataset="erlangen", source_id="erlangen:sample")
    alias["metadata"]["source_image"] = "alias.jpg"
    photo["sources"].append({"dataset": "erlangen", "resource": "alias.jpg"})
    plan = FrozenPlan([alias, row], [photo])
    assert plan.decode_order() == ["rimone", "erlangen"]
    replay = Replay(root, plan, out)
    with pytest.raises(ValueError, match="owner must emit first"):
        replay.emit(
            "erlangen",
            "alias",
            "unsplit",
            image,
            target,
            metadata={"source_image": "alias.jpg", "annotator": "average"},
            source_id="sample",
        )
    bad = copy.deepcopy(row)
    bad["split"] = "val"
    with pytest.raises(ValueError, match="split conflict"):
        FrozenPlan([bad], [photo])
    with pytest.raises(ValueError, match="Source/patient identity"):
        replay.emit(
            "rimone",
            "sample_average",
            "unsplit",
            image,
            target,
            metadata={"source_image": "rgb.jpg", "annotator": "average"},
            source_id="other",
            task="disc_cup",
        )


def test_test_rows_separate_unmaterialized_and_finish_requires_all(frozen):
    root, out, _image, _target, _valid, _fov, row, photo = frozen
    heldout = copy.deepcopy(row)
    heldout.update(id="test_row", split="test", photo_id="TEST_NEVER_DECODE")
    plan = FrozenPlan([row, heldout], [photo])
    assert set(plan.rows) == {row["id"]}
    assert plan.test_inventory[0]["materialized35"] is False
    assert "TEST_NEVER_DECODE" not in plan.photos
    with pytest.raises(ValueError, match="incomplete"):
        Replay(root, plan, out).finish({})
    assert not (out / "_SUCCESS").exists()


class BytesOnly:
    names = ("selected/rgb.png", "selected/label.png", "TEST/secret.png")

    def __init__(self):
        self.calls = []

    def bytes(self, name):
        self.calls.append(name)
        buf = io.BytesIO()
        Image.fromarray(np.full((3, 3, 3), 120, np.uint8)).save(buf, format="PNG")
        return buf.getvalue()


def test_resource_filter_blocks_before_underlying_bytes_and_images():
    raw = BytesOnly()
    scope = FilteredSource(raw, raw.names[:2], "rimone", {})
    assert scope.images("TEST", ".png") == []
    with pytest.raises(ValueError):
        scope.read("TEST/secret.png", "RGB")
    assert raw.calls == []
    assert scope.read("rgb.png", "RGB").shape == (3, 3, 3)
    assert raw.calls == ["selected/rgb.png"]
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_requests(["a/rgb.png", "b/rgb.png"], ["rgb.png"])


def test_resource_digest_drift_and_sidecar_are_guarded():
    raw = BytesOnly()
    scope = FilteredSource(
        raw,
        raw.names[:2],
        "rimone",
        {},
        expected={"selected/rgb.png": {"size": 1, "sha256": "0" * 64}},
    )
    with pytest.raises(ValueError, match="byte identity drift"):
        scope.read("rgb.png")
    r = {
        "dataset": "g1020",
        "task": "disc_cup",
        "metadata": {"source_image": "image_1.jpg", "patient_crosswalk": {"resource": "G1020.csv"}},
    }
    assert ("G1020.zip", "G1020.csv", "patient_sidecar") in dependencies(r)


def test_predecode_observer_hook_preserves_names_and_skips_excluded(monkeypatch):
    class ObserverSource:
        names = ("BinRushed/BinRushed1-Corrected/image10prime.jpg",)

        def retains(self, stem, task):
            return stem.endswith("observer4")

        def find(self, path, optional=False):
            return path

        def read(self, path, mode="RGB"):
            assert "prime" in path or "-4." in path, "Unretained observer decoded"
            return np.ones((10, 10, 3), np.uint8)

    monkeypatch.setattr(
        segmentation,
        "recover_riga",
        lambda a, b: (np.ones((10, 10), bool), np.zeros((10, 10), bool), {}),
    )
    emitted = []
    segmentation.riga(ObserverSource(), lambda *a, **kw: emitted.append((a, kw)))
    assert [a[0] for a, _ in emitted] == ["BinRushed1-Corrected_image10_observer4"]
    assert emitted[0][1]["metadata"]["annotator"] == 4


@pytest.mark.parametrize(
    "old,new",
    [
        ({"valid": [1]}, {"valid": [0]}),
        ({"positive": [1]}, {"positive": [0]}),
        ({"AV_scoring": 1}, {"AV_scoring": 0}),
    ],
)
def test_lost_support_requires_review(old, new):
    with pytest.raises(ValueError, match="support disappeared"):
        preserve_support(old, new)


def test_safe_paths_profile_disabled_and_symlink_escape(frozen, tmp_path):
    root, out, _, _, _, _, row, photo = frozen
    for relative in ("../escape", "/absolute"):
        with pytest.raises(ValueError, match="escapes"):
            safe(out, relative)
    (out / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        safe(out, "link/outside")
    with pytest.raises(ValueError, match="disabled"):
        validate_profile(
            {"execution_authorized": False}, FrozenPlan([row], [photo]), root, root, out
        )


def test_full_synthetic_execution_profile_and_native_resource_gate(frozen, tmp_path, monkeypatch):
    """Exercise runner/closure/output lifecycle; decoder uses fixture arrays only."""
    import platform
    import zipfile
    from importlib.metadata import version

    from deepretina import raw35_producer as producer

    root, out, image, target, valid, fov, row, photo = frozen
    out.rmdir()  # run requires exclusive creation
    packages = {p: version(p) for p in ("numpy", "pillow")}
    for name, value in (
        ("manifest.json", [row]),
        ("photos.json", [photo]),
        ("preparation.json", {"runtime_packages": packages}),
        ("integrity.json", {"prepared_content_sha256": "synthetic"}),
    ):
        (root / name).write_text(json.dumps(value))
    for constant, name in (
        ("MANIFEST_SHA", "manifest.json"),
        ("PHOTOS_SHA", "photos.json"),
        ("PREPARATION_SHA", "preparation.json"),
    ):
        monkeypatch.setattr(producer, constant, producer.sha((root / name).read_bytes()))
    monkeypatch.setattr(producer, "CONTENT_SHA", "synthetic")
    plan = FrozenPlan([row], [photo])
    data = tmp_path / "raw"
    data.mkdir()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    raw = buffer.getvalue()
    with zipfile.ZipFile(data / "RIM-ONE.zip", "w") as archive:
        archive.writestr("rgb.jpg", raw)
        archive.writestr("TEST/DO_NOT_READ.jpg", b"THIS IS NOT AN IMAGE")
    profile = {
        "execution_authorized": True,
        "membership_sha256": plan.signature,
        "max_seconds": 10,
        "max_read_bytes": 10000,
        "max_output_bytes": 2000000,
        "min_free_bytes": 1,
        "max_alias_cached_photos": 1,
        "max_rss_bytes": 10**12,
        "Python": platform.python_version(),
        "runtime_packages": packages,
        "exclusive_output_parent": str(tmp_path),
        "resources": {
            "RIM-ONE.zip": {
                "request_to_member": {"rgb.jpg": "rgb.jpg"},
                "members": {"rgb.jpg": {"size": len(raw), "sha256": producer.sha(raw)}},
            }
        },
    }

    def fixture_decoder(source, callback):
        decoded = source.read("rgb.jpg", "RGB")
        callback(
            "sample_average",
            "unsplit",
            decoded,
            target,
            valid=valid,
            fov=fov,
            source_id="sample",
            task="disc_cup",
            metadata={"source_image": "rgb.jpg", "annotator": "average"},
        )

    monkeypatch.setitem(producer.ADAPTERS, "rimone", fixture_decoder)
    policy_text = (
        producer.Path(__file__).parent / "reference/geometry_policy/policy.json"
    ).read_text()
    policy = {"format_version": 1, "sha256": producer.POLICY_SHA, "document_json": policy_text}
    fingerprint = producer.run(root, data, out, profile, plan, policy)
    assert verify_prepared_content(out / "manifest.json") == fingerprint
    preparation = json.loads((out / "preparation.json").read_text())
    assert preparation["geometry_policy_sha256"] != producer.POLICY_SHA
    assert preparation["parent25_geometry_policy_sha256"] == producer.POLICY_SHA
    assert preparation["parent_geometry_policy_sha256"] == producer.POLICY_SHA
    policy35 = json.loads((out / "geometry-policy35.json").read_text())
    assert policy35["parent_geometry_policy_sha256"] == producer.POLICY_SHA
    assert policy35["source_rule_document"] == json.loads(policy_text)
    assert policy35["policy_id"] != json.loads(policy_text)["policy_id"]
    assert preparation["resources"]["RIM-ONE.zip"].keys() == {"rgb.jpg"}
    assert (
        json.loads((out / "geometry-policy35.json").read_text())["target_pixels_per_degree"] == 35
    )
    assert preparation["TEST_materialized"] is False


def test_native_source_normalization_keeps_direct_v2_and_parent_ids():
    from deepretina.raw35_producer import native_source_id

    assert native_source_id("idrid_disc", "IDRiD_01", "idrid:idrid_01") == "idrid:idrid_01"
    assert native_source_id("rite", "21_training", "drive:21") == "drive:21"
    assert native_source_id("onhsd", "D01187-L_Bob", "D01187") == "onhsd:D01187"
    assert native_source_id("aria", "(0001)aria_d_26", "aria:d:26") == "aria:d:26"


def test_native_read_budget_rejects_before_bytes():
    from deepretina.raw35_producer import Budget

    raw = BytesOnly()

    class RejectBudget:
        def __init__(self):
            self.profile = {"max_read_bytes": 1}
            self.read_bytes = 0

        def check(self, additional_read=0):
            Budget.check(self, additional_read)

    scope = FilteredSource(
        raw,
        raw.names[:1],
        "rimone",
        {},
        budget=RejectBudget(),
        expected={"selected/rgb.png": {"size": 100, "sha256": "0" * 64}},
    )
    # Actual Budget checks the time bound first; supply that state before invoking it.
    import time

    scope.budget.started = time.monotonic()
    scope.budget.profile["max_seconds"] = 10
    with pytest.raises(ValueError, match="Read budget exceeded"):
        scope.read("rgb.png")
    assert raw.calls == []


def test_zip_metadata_alias_resolution_rejects_conflicts_without_bytes():
    names = ["long/a/rgb.png", "a/rgb.png"]
    assert resolve_requests(names, ["rgb.png"], {n: (10, 123) for n in names}) == {
        "rgb.png": "a/rgb.png"
    }
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_requests(names, ["rgb.png"], {names[0]: (10, 123), names[1]: (10, 456)})
