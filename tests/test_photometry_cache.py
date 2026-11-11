"""Small immutable RGB derivatives; no full corpus, real-model training or GPU."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina import photometry as p
from deepretina import photometry_cache as c
from deepretina import unified as u


@pytest.fixture
def corpus(tmp_path):
    from test_derived_cache import audit
    from test_derived_cache import corpus as fixture

    manifest, rows = fixture(tmp_path / "canonical")
    photos = json.loads((manifest.parent / "photos.json").read_text())
    for photo in photos:
        photo.update(preprocessing_profile="legacy25_raw", normalized=False, pixels_per_degree=25.0)
    (manifest.parent / "photos.json").write_text(json.dumps(photos))
    audit(manifest.parent)
    content = json.loads((manifest.parent / "integrity.json").read_text())[
        "prepared_content_sha256"
    ]
    return manifest, rows, content


def built(tmp_path, corpus, profile="clahe_hsv_v1", **options):
    manifest, _, content = corpus
    report = c.build(
        manifest,
        tmp_path / "processed",
        profile,
        expected_content_sha256=content,
        expected_descriptor_sha256=c.sha256(c.canonical(p.contract(profile))).hexdigest(),
        **options,
    )
    cache = c.VerifiedPhotometryCache(
        report["root"],
        manifest,
        profile,
        expected_fingerprint=report["fingerprint"],
        expected_canonical_fingerprint=content,
        allow_subset=options.get("subset_proof", False),
    )
    return report, cache


@pytest.mark.parametrize("profile", p.PROFILES[1:])
def test_full_toy_cache_matches_uncached_inputs_targets_crop_rng_and_no_double_apply(
    tmp_path, corpus, profile, monkeypatch
):
    manifest, rows, _ = corpus
    report, cache = built(tmp_path, corpus, profile)
    assert report["photos"] == 3 and report["coverage"] == "train_val"
    assert set(cache.index["photos"]) == {"photo-0", "photo-1", "photo-2"}
    proof = cache.verify_all_bytes()
    assert proof["encoded_byte_hashes_verified"] and proof["array_data_decoded"] is False
    cfg = u.UnifiedConfig(
        photometric_profile=profile, augmentation="basic", patch_size=32, foreground_probability=0
    )
    direct = u.UnifiedDataset(manifest, config=cfg)
    processed = u.UnifiedDataset(manifest, config=cfg, photometry_cache=cache)
    torch.manual_seed(42)
    a = direct[0]
    state = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached RGB must not be transformed again")

    monkeypatch.setattr(u, "apply_photometry", forbidden)
    torch.manual_seed(42)
    b = processed[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b)) and torch.equal(
        state, torch.get_rng_state()
    )
    assert (Path(report["root"]) / "index.json").stat().st_mode & 0o222 == 0
    cache.require_coverage([r for r in rows if r["split"] == "val"])
    with pytest.raises(ValueError, match="fallback"):
        cache.rgb("photo-3")


def test_dependency_descriptor_canonical_external_pin_and_profile_mismatches(tmp_path, corpus):
    report, cache = built(tmp_path, corpus)
    manifest, _, content = corpus
    arguments = {
        "expected_fingerprint": report["fingerprint"],
        "expected_canonical_fingerprint": content,
    }
    for mode, args in [
        ("raw", arguments),
        ("legacy_lc_clahe_v1", arguments),
        ("clahe_hsv_v1", dict(arguments, expected_fingerprint="0" * 64)),
        ("clahe_hsv_v1", dict(arguments, expected_canonical_fingerprint="0" * 64)),
    ]:
        with pytest.raises(ValueError):
            c.VerifiedPhotometryCache(report["root"], manifest, mode, **args)
    index = copy.deepcopy(cache.index)
    index["descriptor"]["backend_versions"]["scikit-image"] = "wrong"
    root = Path(report["root"])
    (root / "index.json").chmod(0o644)
    (root / "_SUCCESS").chmod(0o644)
    raw = c.canonical(index)
    (root / "index.json").write_bytes(raw)
    (root / "_SUCCESS").write_text(c.fingerprint(raw) + "\n")
    with pytest.raises(ValueError, match="descriptor/backend"):
        c.VerifiedPhotometryCache(
            root,
            manifest,
            "clahe_hsv_v1",
            expected_fingerprint=c.fingerprint(raw),
            expected_canonical_fingerprint=content,
        )


def test_mapped_bitflip_fails_both_read_and_streaming_audit_without_fallback(tmp_path, corpus):
    report, cache = built(tmp_path, corpus)
    path = Path(report["root"]) / cache.index["photos"]["photo-0"]["path"]
    path.chmod(0o644)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="byte hash"):
        cache.rgb("photo-0")
    with pytest.raises(ValueError, match="byte hash"):
        cache.verify_all_bytes()


def test_subset_proof_cannot_be_presented_as_full_cache(tmp_path, corpus):
    report, cache = built(tmp_path, corpus, photo_ids=["photo-0"], subset_proof=True)
    manifest, _, content = corpus
    assert cache.rgb("photo-0").dtype == np.uint8
    with pytest.raises(ValueError, match="Full TRAIN/VAL"):
        c.VerifiedPhotometryCache(
            report["root"],
            manifest,
            "clahe_hsv_v1",
            expected_fingerprint=report["fingerprint"],
            expected_canonical_fingerprint=content,
        )
    with pytest.raises(ValueError, match="coverage"):
        cache.require_coverage(u.read_unified_manifest(manifest, "train"))


def test_existing_output_and_unsafe_destination_never_overwrite(tmp_path, corpus):
    manifest, _, content = corpus
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")
    for path in (output, manifest.parent / "forbidden"):
        with pytest.raises(ValueError, match="Fresh output"):
            c.build(
                manifest,
                path,
                "clahe_hsv_v1",
                expected_content_sha256=content,
                expected_descriptor_sha256=c.sha256(
                    c.canonical(p.contract("clahe_hsv_v1"))
                ).hexdigest(),
            )
    assert marker.read_text() == "keep"


def test_wrong_source_alias_is_rejected_not_silently_bypassed(tmp_path, corpus):
    from test_derived_cache import audit

    manifest, rows, content = corpus
    Image.fromarray(np.zeros((36, 40, 3), np.uint8)).save(manifest.parent / "wrong.png")
    rows.append(dict(rows[0], id="alias", image="wrong.png"))
    manifest.write_text(json.dumps(rows))
    audit(manifest.parent)
    content = json.loads((manifest.parent / "integrity.json").read_text())[
        "prepared_content_sha256"
    ]
    with pytest.raises(ValueError, match="RGB aliases differ"):
        c.build(
            manifest,
            tmp_path / "processed",
            "clahe_hsv_v1",
            expected_content_sha256=content,
            expected_descriptor_sha256=c.sha256(
                c.canonical(p.contract("clahe_hsv_v1"))
            ).hexdigest(),
        )
    assert not (tmp_path / "processed").exists()


def test_cache_module_is_in_training_fingerprint(monkeypatch):
    before = u._training_source_sha256()
    original = Path.read_bytes

    def altered(path):
        return original(path) + (b"\n#fixture" if path.name == "photometry_cache.py" else b"")

    monkeypatch.setattr(Path, "read_bytes", altered)
    assert u._training_source_sha256() != before


def test_reader_releases_raw_arrays_and_keeps_only_bounded_tensor_LRU(tmp_path, corpus):
    import gc
    import weakref

    _, cache = built(tmp_path, corpus)
    refs = []
    for i in range(24):
        image = cache.rgb(f"photo-{i % 3}")
        assert not isinstance(image, np.memmap)
        refs.append(weakref.ref(image))
        del image
    gc.collect()
    assert all(ref() is None for ref in refs)
    manifest, _, _ = corpus
    dataset = u.UnifiedDataset(
        manifest, config=u.UnifiedConfig(photometric_profile="clahe_hsv_v1"), photometry_cache=cache
    )
    dataset._cache_limit = 20_000
    for i in range(12):
        dataset._image(i % 2)
        assert len(dataset._image_cache) <= 1 and dataset._cache_bytes <= dataset._cache_limit


def test_cached_training_resume_and_relocated_cache_preserve_optimizer_rng(tmp_path, corpus):
    import shutil

    manifest, _, content = corpus
    report, cache = built(tmp_path, corpus)
    cfg = u.UnifiedConfig(
        epochs=2,
        epoch_samples=4,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        workers=0,
        device="cpu",
        augmentation="basic",
        tile_size=32,
        overlap=8,
        require_all_heads=False,
        photometric_profile="clahe_hsv_v1",
    )
    u.train_unified(manifest, tmp_path / "uncached", cfg)
    options = {
        "photometry_cache_root": report["root"],
        "expected_photometry_cache_fingerprint": report["fingerprint"],
        "expected_content_sha256": content,
    }
    u.train_unified(manifest, tmp_path / "cached", cfg, stop_after_epoch=1, **options)
    before = (tmp_path / "cached/last.pt").read_bytes()
    with pytest.raises(ValueError, match="Resume photometry cache identity"):
        u.train_unified(
            manifest,
            tmp_path / "cached",
            cfg,
            resume=tmp_path / "cached/last.pt",
            expected_content_sha256=content,
        )
    assert (tmp_path / "cached/last.pt").read_bytes() == before
    relocated = tmp_path / "relocated"
    shutil.copytree(report["root"], relocated)
    options["photometry_cache_root"] = relocated
    u.train_unified(
        manifest, tmp_path / "cached", cfg, resume=tmp_path / "cached/last.pt", **options
    )
    a = torch.load(tmp_path / "uncached/last.pt", weights_only=True)
    b = torch.load(tmp_path / "cached/last.pt", weights_only=True)
    assert b["photometry_cache"] == cache.identity

    def exact(x, y):
        if torch.is_tensor(x):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                exact(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for i, j in zip(x, y):
                exact(i, j)
        else:
            assert x == y

    exact(a["state_dict"], b["state_dict"])
    exact(a["training_state"], b["training_state"])


def test_dataset_rejects_cache_descriptor_at_different_physical_scale(tmp_path, corpus):
    _, cache = built(tmp_path, corpus)
    manifest, _, _ = corpus
    config = u.UnifiedConfig(photometric_profile="clahe_hsv_v1")
    cache.index["descriptor"] = p.contract("clahe_hsv_v1", 35)
    with pytest.raises(ValueError, match="manifest/profile"):
        u.UnifiedDataset(manifest, config=config, photometry_cache=cache)
