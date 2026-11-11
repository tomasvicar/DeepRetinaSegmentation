"""Opt-in full-image intensity contracts; synthetic fixtures, no scientific scores."""

import copy
import json
import random
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina import photometry as p
from deepretina import unified as u
from deepretina.legacy.preprocessing import ImageGeometry, local_contrast_and_clahe


@pytest.fixture
def corpus(tmp_path):
    raw = np.random.default_rng(3).integers(15, 220, (72, 96, 3), dtype=np.uint8)
    fov = np.zeros(raw.shape[:2], bool)
    fov[5:-5, 7:-7] = True
    Image.fromarray(raw).save(tmp_path / "image.png")
    Image.fromarray(fov.astype(np.uint8) * 255).save(tmp_path / "fov.png")
    target = np.zeros((3, 72, 96), np.float32)
    target[0, 10:50, 12:80] = 1
    target[1, 10:30, 12:80] = 1
    target[2, 30:50, 12:80] = 1
    np.savez(tmp_path / "labels.npz", target=target, valid=np.repeat(fov[None], 3, axis=0), fov=fov)
    row = {
        "id": "sample",
        "photo_id": "photo1",
        "source_id": "source1",
        "dataset": "synthetic",
        "split": "train",
        "task": "av",
        "image": "image.png",
        "mask": "labels.npz",
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25.0,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([row]))
    (tmp_path / "photos.json").write_text(
        json.dumps(
            [
                {
                    "photo_id": "photo1",
                    "fov": "fov.png",
                    "image": "image.png",
                    "preprocessing_profile": "legacy25_raw",
                    "normalized": False,
                    "pixels_per_degree": 25.0,
                    "prepared_shape": list(raw.shape),
                }
            ]
        )
    )
    return manifest, raw, fov, row


@pytest.mark.parametrize("profile", p.PROFILES)
def test_full_image_deterministic_geometry_range_outside_and_rng(corpus, profile):
    _, raw, fov, _ = corpus
    original = raw.copy()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    first = p.apply(raw, profile, fov=fov)
    second = p.apply(raw, profile, fov=fov)
    assert first.shape == raw.shape and first.dtype == np.uint8 and np.isfinite(first).all()
    assert np.array_equal(first, second) and np.array_equal(raw, original)
    assert np.array_equal(first[~fov], raw[~fov])
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(before[2], np.random.get_state()))
    if profile == "raw":
        assert first is raw
    else:
        assert not np.array_equal(first[fov], raw[fov])


def test_background_contract_reuses_actual_legacy_operator(corpus):
    _, raw, fov, _ = corpus
    expected = local_contrast_and_clahe(raw, fov, 25)
    result = p.apply(raw, "legacy_lc_clahe_v1", fov=fov)
    assert np.array_equal(result[fov], expected[fov])
    assert np.array_equal(result[~fov], raw[~fov])
    descriptor = p.contract("legacy_lc_clahe_v1")
    assert descriptor["kernel_size_effective"] == [113, 113]
    assert descriptor["background"]["sigma"] == 80 / (55 / 25)
    assert descriptor["clip_limit"] == 0.005 and descriptor["nbins"] == 256
    assert descriptor["matlab_parity_claim"] is False


def test_missing_fov_invalid_profile_and_degenerate_background_fail(corpus):
    _, raw, fov, _ = corpus
    for domain in (None, np.zeros_like(fov), fov[:2], fov.astype(np.uint8)):
        with pytest.raises(ValueError):
            p.apply(raw, "clahe_hsv_v1", fov=domain)
    with pytest.raises(ValueError):
        p.contract("unknown")
    with pytest.raises(ValueError):
        p.contract("clahe_hsv_v1", 0.01)
    with pytest.raises(ValueError, match="background"):
        p.apply(np.zeros_like(raw), "legacy_lc_clahe_v1", fov=fov)


def test_default_raw_does_not_read_fov_and_retains_crop_rng(corpus, monkeypatch):
    manifest, raw, _, _ = corpus

    def forbidden(*args):
        raise AssertionError("raw must not read canonical FOV")

    monkeypatch.setattr(u, "fov_index", forbidden)
    monkeypatch.setattr(u, "image_fov", forbidden)
    cfg = u.UnifiedConfig(augmentation="basic", foreground_probability=0, patch_size=32)
    a = u.UnifiedDataset(manifest, config=cfg)
    b = u.UnifiedDataset(manifest, config=replace(cfg, photometric_profile="raw"))
    assert torch.equal(a._image(0), torch.from_numpy(raw.transpose(2, 0, 1).copy()).float() / 255)
    torch.manual_seed(42)
    first = a[0]
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    second = b[0]
    assert all(torch.equal(x, y) for x, y in zip(first, second)) and torch.equal(
        state, torch.get_rng_state()
    )


@pytest.mark.parametrize("profile", p.PROFILES[1:])
def test_full_photo_precedes_crop_targets_unchanged_and_cache_separates_profiles(
    corpus, profile, monkeypatch
):
    manifest, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(
        augmentation="basic", foreground_probability=0, patch_size=32, photometric_profile=profile
    )
    calls = []
    original = u.apply_photometry

    def checked(image, *args, **kwargs):
        calls.append(image.shape)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(u, "apply_photometry", checked)
    dataset = u.UnifiedDataset(manifest, config=cfg)
    full = dataset._image(0)
    assert dataset._image(0) is full
    assert calls == [raw.shape]
    assert torch.equal(
        full,
        torch.from_numpy(p.apply(raw, profile, fov=fov).transpose(2, 0, 1).copy()).float() / 255,
    )
    baseline = u.UnifiedDataset(manifest, config=replace(cfg, photometric_profile="raw"))
    torch.manual_seed(13)
    a = dataset[0]
    state = torch.get_rng_state().clone()
    torch.manual_seed(13)
    b = baseline[0]
    assert a[0].shape == (3, 32, 32) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert torch.equal(state, torch.get_rng_state())
    cfg.photometric_profile = "raw"
    assert torch.equal(dataset._image(0), baseline._image(0)) and calls == [raw.shape]


@pytest.mark.parametrize("profile", p.PROFILES)
def test_high_level_inference_and_dataset_receive_identical_prepared_pixels(
    corpus, profile, tmp_path, monkeypatch
):
    manifest, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(photometric_profile=profile)
    expected = u.UnifiedDataset(manifest, config=cfg)._image(0)
    geometry = ImageGeometry(
        raw.shape[:2], (0, raw.shape[0] - 1, 0, raw.shape[1] - 1), 1.0, raw.shape[:2]
    )
    prepared = SimpleNamespace(
        image=raw.copy(),
        fov=fov,
        geometry=geometry,
        field_of_view_degrees=45,
        fov_source="synthetic image-only FOV",
    )
    from deepretina.legacy import preprocessing

    monkeypatch.setattr(preprocessing, "prepare_single_net", lambda *a, **kw: prepared)
    payload = {
        "config": asdict(cfg),
        "supervised_channels": [True] * 5,
        "checkpoint_sha256": "synthetic",
        "purpose": "functionality",
    }
    model = SimpleNamespace(lineage={}, architecture_spec={})
    monkeypatch.setattr(u, "load_unified", lambda *a: (model, payload, torch.device("cpu")))
    seen = []

    def capture(model, image, *a, **kw):
        assert torch.equal(image, expected)
        seen.append(image)
        return np.zeros((5, *raw.shape[:2]), np.float32)

    monkeypatch.setattr(u, "predict_probabilities", capture)
    u.predict_unified(
        tmp_path / "synthetic.pt",
        tmp_path / "image.png",
        tmp_path / "output",
        device="cpu",
        dataset="drive",
    )
    assert len(seen) == 1


@pytest.mark.parametrize("profile", p.PROFILES[1:])
def test_safe_checkpoint_records_and_enforces_photometric_parameters(tmp_path, profile):
    cfg = u.UnifiedConfig(photometric_profile=profile, base_channels=2, depth=1)
    model = u.build_unified(cfg)
    payload = {
        "family": "unified",
        "format_version": 1,
        "channels": list(u.CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "preprocessing": {"photometry": p.contract(profile)},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    restored, metadata, _ = u.load_unified(path, "cpu")
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), restored.state_dict().values())
    )
    assert metadata["preprocessing"]["photometry"] == p.contract(profile)
    bad = copy.deepcopy(payload)
    bad["preprocessing"]["photometry"]["clip_limit"] = 0.2
    torch.save(bad, path)
    with pytest.raises(ValueError, match="photometry parameters"):
        u.load_unified(path, "cpu")
    bad = copy.deepcopy(payload)
    del bad["preprocessing"]
    torch.save(bad, path)
    with pytest.raises(ValueError, match="photometry parameters"):
        u.load_unified(path, "cpu")


def test_optin_missing_image_fov_has_no_target_mask_fallback(corpus):
    manifest, _, _, _ = corpus
    (manifest.parent / "photos.json").unlink()
    with pytest.raises(FileNotFoundError):
        u.UnifiedDataset(
            manifest, config=u.UnifiedConfig(photometric_profile="clahe_hsv_v1")
        )._image(0)


def test_fov_assets_are_in_existing_canonical_byte_fingerprint(corpus):
    from deepretina.integrity import prepared_content_fingerprint

    manifest, _, fov, _ = corpus
    before = prepared_content_fingerprint(manifest)
    changed = fov.copy()
    changed[0, 0] = True
    Image.fromarray(changed.astype(np.uint8) * 255).save(manifest.parent / "fov.png")
    assert prepared_content_fingerprint(manifest) != before


def test_new_module_is_in_explicit_training_source_fingerprint():
    import hashlib
    from pathlib import Path
    from unittest.mock import patch

    module = Path(u.__file__).with_name("photometry.py")
    before = u._training_source_sha256()
    original = Path.read_bytes

    def changed(path):
        data = original(path)
        return data + b"\n# synthetic mutation fixture\n" if path == module else data

    with patch.object(Path, "read_bytes", changed):
        assert u._training_source_sha256() != before
    assert len(before) == len(hashlib.sha256().hexdigest())


def test_default_raw_step_matches_genuine_frozen72_weights_optimizer_and_rng(tmp_path):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    from test_unified import make_manifest

    code = subprocess.check_output(
        ["git", "show", "72fb321187eca517823083aca80cedace9605570:src/deepretina/unified.py"],
        cwd=Path(__file__).parents[1],
        text=True,
    )
    name = "deepretina._photometry_reference72"
    baseline = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    baseline.__file__ = u.__file__
    sys.modules[name] = baseline
    try:
        exec(compile(code, u.__file__, "exec"), baseline.__dict__)  # noqa: S102 - immutable repository baseline
        from dataclasses import fields

        assert [f.name for f in fields(u.UnifiedConfig)][:-1] == [
            f.name for f in fields(baseline.UnifiedConfig)
        ]
        manifest = make_manifest(tmp_path)
        settings = {
            "epochs": 1,
            "epoch_samples": 4,
            "batch_size": 2,
            "patch_size": 32,
            "base_channels": 2,
            "depth": 1,
            "workers": 0,
            "device": "cpu",
            "augmentation": "basic",
            "foreground_probability": 0,
            "tile_size": 32,
            "overlap": 8,
            "max_steps": 2,
        }
        for module, destination in ((baseline, "old72"), (u, "newraw")):
            module.train_unified(manifest, tmp_path / destination, module.UnifiedConfig(**settings))
        old = torch.load(tmp_path / "old72/last.pt", weights_only=True)
        new = torch.load(tmp_path / "newraw/last.pt", weights_only=True)

        def exact(a, b):
            if torch.is_tensor(a):
                assert a.dtype == b.dtype and torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a:
                    exact(a[k], b[k])
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    exact(x, y)
            else:
                assert a == b

        exact(old["state_dict"], new["state_dict"])
        exact(old["training_state"], new["training_state"])
        assert old["steps"] == new["steps"] == 2
    finally:
        sys.modules.pop(name)


@pytest.mark.parametrize("profile", p.PROFILES[1:])
def test_lossless_raw_derived_cache_and_canonical_reader_match_after_photometry(tmp_path, profile):
    import hashlib

    from test_derived_cache import audit, opened
    from test_derived_cache import corpus as cache_corpus

    from deepretina.derived_cache import build_cache

    manifest, _ = cache_corpus(tmp_path / "canonical")
    photos = json.loads((manifest.parent / "photos.json").read_text())
    for photo in photos:
        photo.update(preprocessing_profile="legacy25_raw", normalized=False, pixels_per_degree=25.0)
    (manifest.parent / "photos.json").write_text(json.dumps(photos))
    audit(manifest.parent)
    report = build_cache(manifest, tmp_path / "raw-derived")
    cache = opened(report, manifest)
    from pathlib import Path

    derived = Path(report["root"])
    cache_index = (derived / "index.json").read_bytes()
    raw_assets = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in derived.rglob("*.npy")
    }
    cfg = u.UnifiedConfig(photometric_profile=profile)
    direct = u.UnifiedDataset(manifest, config=cfg)
    cached = u.UnifiedDataset(manifest, config=cfg, derived_cache=cache)
    for i in range(len(direct)):
        assert torch.equal(direct._image(i), cached._image(i))
    assert (derived / "index.json").read_bytes() == cache_index
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in raw_assets.items()
    )


def test_failed_augmented_crop_falls_back_to_same_photometric_native_image(corpus, monkeypatch):
    from deepretina import augment

    manifest, raw, fov, _ = corpus
    cfg = u.UnifiedConfig(
        photometric_profile="clahe_hsv_v1",
        patch_size=32,
        foreground_probability=0,
        augmentation="mild",
    )
    monkeypatch.setattr(
        augment,
        "augment",
        lambda image, target, valid, **kw: (
            torch.zeros_like(image),
            target,
            torch.zeros_like(valid),
        ),
    )
    dataset = u.UnifiedDataset(manifest, config=cfg)
    image, _, valid = dataset[0]
    full = (
        torch.from_numpy(p.apply(raw, "clahe_hsv_v1", fov=fov).transpose(2, 0, 1).copy()).float()
        / 255
    )
    assert torch.equal(image, full[:, 20:52, :32]) and valid.any()
