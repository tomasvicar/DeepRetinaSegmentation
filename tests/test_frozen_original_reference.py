"""Synthetic only: no actual image, annotation, original model or CUDA access."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def api(monkeypatch):
    folder = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "frozen_original_reference", folder / "evaluate_original_frozen.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(api, tmp_path, monkeypatch):
    rows = []
    photos = []
    for split in ("train", "val", "test"):
        for ds, task in (("rite", "av"), ("idrid_disc", "disc_cup")):
            pid = f"{split}-{ds}"
            rows.append(
                {
                    "id": pid,
                    "dataset": ds,
                    "task": task,
                    "split": split,
                    "source_id": pid,
                    "split_group_id": pid,
                    "photo_id": pid,
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25,
                    "field_of_view_degrees": 50 if ds == "idrid_disc" else 45,
                }
            )
            photos.append(
                {
                    "photo_id": pid,
                    "split": split,
                    "canonical_dataset": ds,
                    "field_of_view_degrees": 50 if ds == "idrid_disc" else 45,
                    "fov": "not-opened.png",
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    (tmp_path / "photos.json").write_text(json.dumps(photos))
    (tmp_path / "integrity.json").write_text(
        json.dumps(
            {
                "fingerprint_schema": 2,
                "manifest_sha256": api.sha_bytes(manifest.read_bytes()),
                "prepared_content_sha256": "c" * 64,
            }
        )
    )
    paths = {}
    for name in api.MODEL_SHA:
        paths[name] = tmp_path / (name + ".pt")
        paths[name].write_bytes(name.encode())
    pins = {name: api.sha_bytes(p.read_bytes()) for name, p in paths.items()}
    monkeypatch.setattr(api, "MODEL_SHA", pins)
    recipe = copy.deepcopy(api.POLICY)
    recipe["model_sha256"] = pins
    monkeypatch.setattr(api, "POLICY", recipe)
    monkeypatch.setattr(api, "source_identity", lambda: {"synthetic": "source"})
    monkeypatch.setattr(api, "capture_geometry_policy", lambda *a: None)
    return manifest, rows, paths


def test_metadata_dynamic_inventory_and_idrid_geometry(api, inputs):
    manifest, rows, _ = inputs
    identity, groups = api.metadata_identity(manifest)
    assert identity["eligible_photo_counts"] == {"train": 2, "val": 2, "test": 2}
    assert set(groups["test"]) == {"test-rite", "test-idrid_disc"}
    rows[-1]["field_of_view_degrees"] = 45
    manifest.write_text(json.dumps(rows))
    audit = json.loads((manifest.parent / "integrity.json").read_text())
    audit["manifest_sha256"] = api.sha_bytes(manifest.read_bytes())
    (manifest.parent / "integrity.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="50 degrees"):
        api.metadata_identity(manifest)


def policy(api, manifest):
    identity, _ = api.metadata_identity(manifest)
    return {
        "kind": "frozen_original_reference_policy",
        "schema_version": 1,
        "policy": api.POLICY,
        "source": api.source_identity(),
        "data": identity,
        "validation_report_sha256": "a" * 64,
        "selected_unified_selection_sha256": "b" * 64,
        "rationale": "VAL selected",
        "frozen_at_utc": "2026-11-10T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    "mutation", ["review-sha", "source", "policy", "data", "model", "val-digest"]
)
def test_bad_review_fails_before_device_loader_or_gt(api, inputs, monkeypatch, mutation):
    manifest, _, paths = inputs
    selection = policy(api, manifest)
    if mutation == "source":
        selection["source"] = {}
    if mutation == "policy":
        selection["policy"] = {}
    if mutation == "data":
        selection["data"] = {}
    if mutation == "val-digest":
        selection["validation_report_sha256"] = None
    path = manifest.parent / "policy.json"
    path.write_text(json.dumps(selection))
    digest = api.sha_bytes(path.read_bytes())
    if mutation == "review-sha":
        digest = "0" * 64
    if mutation == "model":
        paths["disc"].write_bytes(b"changed")

    def forbidden(*a, **k):
        pytest.fail("No GT/model/CUDA allowed before failed review")

    for name in (
        "resolve_device",
        "verify_prepared_content",
        "UnifiedDataset",
        "load_single_net",
        "load_av",
    ):
        monkeypatch.setattr(api, name, forbidden)
    with pytest.raises(ValueError):
        api.evaluate(
            manifest,
            paths,
            manifest.parent,
            split="test",
            device="cuda:0",
            policy_path=path,
            expected_policy_sha256=digest,
        )


def test_test_without_explicit_policy_fails_before_metadata(api, monkeypatch):
    monkeypatch.setattr(api, "metadata_identity", lambda *a: pytest.fail("metadata not needed"))
    with pytest.raises(ValueError, match="reviewed"):
        api.evaluate("absent", {}, ".", split="test")


def test_native_forward_is_raw_strict_same_disc_and_clip_before_restore(api, monkeypatch):
    outputs = {
        "universal": torch.tensor(
            [[[1.0, 0.0], [-1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
        ),
        "vessel": torch.tensor([[[0.0, -1.0], [1.0, 1.0]]]),
        "disc": torch.tensor([[[0.0, 1.0], [-1.0, 1.0]]]),
    }
    calls = []

    def forward(model, image, **kwargs):
        calls.append(model)
        assert torch.equal(image, torch.zeros(3, 2, 2))
        return outputs[model]

    monkeypatch.setattr(api, "tiled_logits", forward)
    loaded = {name: (name, {"tiling": {"crop_size": 2, "border": 0}}) for name in outputs}
    masks = api.native_masks(torch.full((3, 2, 2), 0.5), loaded, "cpu")
    assert calls == ["universal", "vessel", "disc"]
    assert np.array_equal(masks[4], np.array([[False, True], [False, True]]))
    assert not masks[0, 0, 0]  # strict sign, zero stays negative
    outputs["disc"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        api.native_masks(torch.full((3, 2, 2), 0.5), loaded, "cpu")


def test_av_calls_calibrated_provided_fov_and_keeps_tie_zero_nn(api, monkeypatch):
    calls = []

    class Calibration:
        diameter_pixels = 17

    # asdict contract is a dataclass in production.
    monkeypatch.setattr(api, "asdict", lambda x: {"diameter": 17})

    def calibrate(fov, **kw):
        calls.append(("calibrate", kw))
        return Calibration()

    def prepare(rgb, vessel, **kw):
        calls.append(("prepare", kw))
        return SimpleNamespace(inputs=lambda: "calibrated-input")

    def predict(model, inputs, **kw):
        assert inputs == "calibrated-input"
        return {
            "classes": np.array([[1, 2], [0, 0]], np.uint8),
            "prepared_logits": np.array([[[0, 0], [0, 0]], [[2, 1], [3, 0]], [[1, 2], [3, 0]]]),
        }

    restores = []
    monkeypatch.setattr(api, "calibrate_provided_fov", calibrate)
    monkeypatch.setattr(api, "prepare_vesclass_matlab_calibrated", prepare)
    monkeypatch.setattr(api, "predict_av_input", predict)
    monkeypatch.setattr(
        api, "restore_classes", lambda x, shape: restores.append((x.copy(), shape)) or x
    )
    fov = np.ones((2, 2), bool)
    _classes, conditional, meta = api.calibrated_av(
        np.zeros((2, 2, 3)),
        fov,
        fov,
        {"canonical_dataset": "rite", "field_of_view_degrees": 45},
        None,
        "cpu",
    )
    assert calls[0][0] == "calibrate" and calls[1][1]["fov_diameter_pixels"] == 17
    assert calls[1][1]["fov"] is fov
    assert np.array_equal(conditional, [[1, 2], [0, 0]]) and len(restores) == 1
    assert meta["raw_fov_sha256"] == api.sha_bytes(fov.tobytes())


def fake_evaluation(api, inputs, monkeypatch):
    manifest, rows, paths = inputs
    monkeypatch.setattr(api, "verify_prepared_content", lambda *a: "c" * 64)
    monkeypatch.setattr(api.gate, "_canonical_metadata", lambda *a: "stable")
    monkeypatch.setattr(api, "resolve_device", lambda x: "cpu")

    class Model:
        def state_dict(self):
            return {"x": torch.ones(1)}

    def load(handle, device):
        name = handle.read().decode()
        return Model(), {
            "channels": {
                "universal": ["vessel", "disc", "cup"],
                "vessel": ["vessel"],
                "disc": ["disc"],
            }[name]
        }

    monkeypatch.setattr(api, "load_single_net", load)
    monkeypatch.setattr(api, "load_av", lambda *a: (Model(), {}))

    class Dataset:
        def __init__(self, manifest, split, config):
            self.rows = [r for r in rows if r["split"] == split]
            self.groups = [[r] for r in sorted(self.rows, key=lambda r: r["photo_id"])]
            self.photo_ids = [r[0]["photo_id"] for r in self.groups]

        def sample(self, index, **kwargs):
            row = self.groups[index][0]
            target = torch.zeros(5, 2, 2)
            valid = torch.zeros(5, 2, 2, dtype=torch.bool)
            if row["task"] == "av":
                target[0, 0] = 1
                target[1, 0, 0] = 1
                target[2, 0, 1] = 1
                valid[0] = True
                valid[1:3, 0] = True
            else:
                target[3, 0, 0] = 1
                valid[3] = True  # CUP UNKNOWN, never negative GT
            return torch.zeros(3, 2, 2), target, valid, [(row["id"], row["task"])]

    monkeypatch.setattr(api, "UnifiedDataset", Dataset)
    masks = np.zeros((5, 2, 2), bool)
    masks[0, 0] = True
    masks[3, 0, 0] = True
    masks[4, 0, 0] = True
    monkeypatch.setattr(api, "native_masks", lambda *a: masks.copy())
    orders = []
    monkeypatch.setattr(
        api,
        "geometry",
        lambda p: SimpleNamespace(
            restore=lambda x, order: orders.append(order) or x, prepare_mask=lambda x: x
        ),
    )
    monkeypatch.setattr(api, "raw_photo", lambda *a: np.zeros((2, 2, 3), np.uint8))

    class FakeImage:
        def __enter__(self):
            return np.ones((2, 2), np.uint8)

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(api.Image, "open", lambda *a: FakeImage())
    calls = []

    def av(*args):
        calls.append(args)
        return np.array([[1, 2], [0, 0]]), np.array([[1, 2], [0, 0]]), {"calibrated": True}

    monkeypatch.setattr(api, "calibrated_av", av)
    return manifest, paths, orders, calls


def test_full_val_unknown_cup_support_and_dynamic_attribution(api, inputs, monkeypatch):
    manifest, paths, orders, calls = fake_evaluation(api, inputs, monkeypatch)
    report = api.evaluate(manifest, paths, manifest.parent)
    assert report["photos"] == 2 and len(report["per_photo"]) == 2
    assert report["channels"]["cup"]["valid_pixels"] == 0 and report["channels"]["cup"]["fp"] == 0
    assert report["channels"]["disc"]["tp"] == 1
    assert report["channels"]["artery"]["tp"] == report["channels"]["vein"]["tp"] == 1
    assert report["av_conditional_accuracy"] == report["av_end_to_end_accuracy"] == 1
    assert orders == [0, 0] and len(calls) == 1
    assert report["by_dataset"]["idrid_disc"]["cup"]["dice"] is None
    assert report["by_dataset"]["rite"]["artery"]["tp"] == 1


def test_freeze_then_reviewed_test_and_tampered_inventory(api, inputs, monkeypatch):
    manifest, paths, _, _ = fake_evaluation(api, inputs, monkeypatch)
    val = api.evaluate(manifest, paths, manifest.parent)
    val_path = manifest.parent / "val.json"
    api.write_new(val_path, val)
    policy_path = manifest.parent / "policy.json"
    digest = api.freeze(val_path, manifest, policy_path, "b" * 64, "VAL chosen")
    report = api.evaluate(
        manifest,
        paths,
        manifest.parent,
        split="test",
        policy_path=policy_path,
        expected_policy_sha256=digest,
    )
    assert (
        report["split"] == "test"
        and report["photos"] == 2
        and report["reviewed_policy_sha256"] == digest
    )
    assert {r["photo_id"] for r in report["per_photo"]} == {"test-rite", "test-idrid_disc"}
    val["per_photo"].append(val["per_photo"][0])
    val_path.write_text(json.dumps(val))
    with pytest.raises(ValueError, match="complete"):
        api.freeze(val_path, manifest, manifest.parent / "other.json", "b" * 64, "selected")
    with pytest.raises(FileExistsError):
        api.write_new(policy_path, {})


def test_bundled_four_models_really_load_from_cpu_bytesio(api):
    """Real safe API compatibility, no dataset inputs or inference."""
    import io

    folder = Path(__file__).parents[1] / "src/deepretina/pretrained"
    names = {
        "universal": "legacy-single-net-universal.pt",
        "vessel": "legacy-single-net-vessels.pt",
        "disc": "legacy-single-net-disc-0.pt",
        "av": "legacy-av-v9.pt",
    }
    snapshots = api.model_snapshots({name: folder / filename for name, filename in names.items()})
    for name, blob in snapshots.items():
        loader = api.load_av if name == "av" else api.load_single_net
        model, payload = loader(io.BytesIO(blob), "cpu")
        assert not model.training
        assert all(
            v.device.type == "cpu" and torch.isfinite(v).all() for v in model.state_dict().values()
        )
        if name != "av":
            assert (
                payload["channels"]
                == {"universal": ["vessel", "disc", "cup"], "vessel": ["vessel"], "disc": ["disc"]}[
                    name
                ]
            )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", ["counts", "source", "aggregate"])
def test_freeze_rejects_inconsistent_saved_val_without_gt(api, inputs, monkeypatch, change):
    manifest, paths, _, _ = fake_evaluation(api, inputs, monkeypatch)
    val = api.evaluate(manifest, paths, manifest.parent)
    if change == "counts":
        val["per_photo"][0]["channels"]["disc"]["tp"] = 0.5
    if change == "source":
        val["per_photo"][0]["channel_datasets"]["disc"] = "wrong"
    if change == "aggregate":
        val["macro_dataset_mean_dice"] = -1
    path = manifest.parent / "changed-val.json"
    api.write_new(path, val)
    with pytest.raises(ValueError):
        api.freeze(path, manifest, manifest.parent / "rejected-policy.json", "b" * 64, "selected")
