"""Lossless derivatives stay explicit, portable, pinned and fail closed."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.derived_cache import VerifiedCache, build_cache, cache_fingerprint
from deepretina.integrity import prepared_content_fingerprint
from deepretina.unified import UnifiedConfig, UnifiedDataset, read_unified_sample


def corpus(root):
    root.mkdir()
    rows, photos = [], []
    for number, split in enumerate(("train", "train", "val", "test")):
        image = np.random.default_rng(number).integers(0, 256, (36, 40, 3), np.uint8)
        image_path, mask_path, fov_path = f"{number}.png", f"{number}.npz", f"{number}-fov.png"
        Image.fromarray(image).save(root / image_path)
        target = np.zeros((3, 36, 40), np.float32)
        target[0, 8:28, 12:30] = 1
        target[1, 8:18, 12:30] = 1
        target[2, 18:28, 12:30] = 1
        if number == 1:
            target[0, 9, 13] = np.float32(0.31)
        valid = np.ones_like(target, bool)
        valid[1:, :, :8] = False
        fov = np.ones((36, 40), bool)
        fov[:3] = False
        labels = {"target": target, "valid": valid, "observer": np.array([number], np.int16)}
        if number != 2:
            labels["fov"] = fov
        np.savez_compressed(root / mask_path, **labels)
        Image.fromarray(fov.astype(np.uint8) * 255).save(root / fov_path)
        rows.append(
            {
                "id": str(number),
                "dataset": "sample",
                "source_id": str(number),
                "photo_id": f"photo-{number}",
                "split": split,
                "task": "av",
                "image": image_path,
                "mask": mask_path,
                "preprocessing_profile": "legacy25_raw",
                "pixels_per_degree": 25.0,
            }
        )
        photos.append(
            {
                "photo_id": f"photo-{number}",
                "image": image_path,
                "fov": fov_path,
                "prepared_shape": list(image.shape),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows))
    (root / "photos.json").write_text(json.dumps(photos))
    audit(root)
    return manifest, rows


def audit(root):
    manifest = root / "manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "_SUCCESS").write_text(sha)
    (root / "integrity.json").write_text(
        json.dumps(
            {
                "fingerprint_schema": 2,
                "manifest_sha256": sha,
                "prepared_content_sha256": prepared_content_fingerprint(manifest),
            }
        )
    )


def opened(report, manifest, **kwargs):
    return VerifiedCache(
        report["root"],
        manifest,
        expected_cache_fingerprint=report["cache_fingerprint"],
        expected_canonical_fingerprint=report["canonical_fingerprint"],
        **kwargs,
    )


def test_binary_soft_fov_and_extra_arrays_are_exact_and_test_is_not_decoded(tmp_path):
    manifest, rows = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived")
    cache = opened(report, manifest)
    assert report["coverage"]["kind"] == "selected-splits"
    assert report["coverage"]["splits"] == ["train", "val"]
    cache.require_coverage(rows[:3])
    with pytest.raises(ValueError, match="lacks"):
        cache.require_coverage(rows)
    assert rows[3]["mask"] not in cache.index["source_to_derived"]
    for row in rows[:3]:
        with cache.labels(row["mask"]) as stored, np.load(manifest.parent / row["mask"]) as raw:
            assert stored["target"].astype(np.float32).tobytes() == raw["target"].tobytes()
            assert np.array_equal(stored["valid"], raw["valid"])
            assert stored.files == raw.files
            assert np.array_equal(stored["observer"], raw["observer"])
            assert stored["target"].dtype == (np.float32 if row["id"] == "1" else bool)
        for original, derived in zip(
            read_unified_sample(manifest.parent, row),
            read_unified_sample(manifest.parent, row, derived_cache=cache),
            strict=True,
        ):
            assert np.array_equal(original, derived)
    assert cache.stats["rgb_hits"] == 3
    assert report["construction"]["exact_equality_verified"]


def test_partial_coverage_needs_explicit_fallback_and_never_falls_back_for_corruption(tmp_path):
    manifest, rows = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived", max_photos=1)
    assert report["coverage"]["kind"] == "subset"
    missing = next(
        row for row in rows[:3] if row["photo_id"] not in report["coverage"]["photo_ids"]
    )
    with pytest.raises(ValueError, match="absent"):
        opened(report, manifest).rgb(missing["image"])
    cache = opened(report, manifest, allow_canonical_fallback=True)
    assert cache.rgb(missing["image"]).dtype == np.uint8
    assert cache.stats["rgb_canonical_fallback"] == 1
    with pytest.raises(ValueError, match="lacks"):
        cache.require_coverage(rows[:3])
    covered = next(row for row in rows if row["photo_id"] in report["coverage"]["photo_ids"])
    path = cache.root / cache.index["source_to_derived"][covered["image"]]["derived"]
    before = path.stat()
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        cache.rgb(covered["image"])
    with pytest.raises(ValueError, match="byte identity"):
        opened(report, manifest)


def test_external_pins_canonical_modification_and_incomplete_cache_rejected(tmp_path):
    manifest, _ = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived")
    for cache_pin, canonical_pin in (
        (None, report["canonical_fingerprint"]),
        (report["cache_fingerprint"], None),
        ("0" * 64, report["canonical_fingerprint"]),
    ):
        with pytest.raises(ValueError):
            VerifiedCache(
                report["root"],
                manifest,
                expected_cache_fingerprint=cache_pin,
                expected_canonical_fingerprint=canonical_pin,
            )
    marker = tmp_path / "derived" / report["canonical_fingerprint"] / "v1-train-val" / "_SUCCESS"
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        opened(report, manifest)
    marker.write_text(report["cache_fingerprint"])
    image = manifest.parent / "0.png"
    image.write_bytes(image.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed"):
        opened(report, manifest)


def test_paths_relocation_and_existing_output_are_safe(tmp_path):
    manifest, rows = corpus(tmp_path / "canonical")
    with pytest.raises(ValueError, match="outside"):
        build_cache(manifest, manifest.parent / "derived")
    report = build_cache(manifest, tmp_path / "derived")
    with pytest.raises(FileExistsError):
        build_cache(manifest, tmp_path / "derived")
    relocated = tmp_path / "relocated"
    shutil.copytree(manifest.parent, relocated / "data")
    shutil.copytree(report["root"], relocated / "cache")
    moved = dict(report, root=str(relocated / "cache"))
    cache = opened(moved, relocated / "data" / "manifest.json")
    assert cache.identity == opened(report, manifest).identity
    assert str(tmp_path) not in json.dumps(cache.identity)
    assert np.array_equal(
        cache.rgb(rows[0]["image"]), opened(report, manifest).rgb(rows[0]["image"])
    )
    for path in ("../escape", str(tmp_path / "absolute")):
        with pytest.raises(ValueError, match="escapes"):
            cache.rgb(path)
    index = json.loads((cache.root / "index.json").read_text())
    index["files"]["../escape"] = {"sha256": "0" * 64, "bytes": 1}
    (cache.root / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="escapes"):
        cache_fingerprint(cache.root)


@pytest.mark.parametrize("augmentation", ["basic", "mild", "strong"])
@pytest.mark.parametrize("view", ["patch", "whole"])
def test_dataset_cached_tensors_annotator_traces_and_rng_are_exact(tmp_path, augmentation, view):
    manifest, _ = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived")
    config = UnifiedConfig(augmentation=augmentation, train_view=view, patch_size=32, view_size=32)
    raw = UnifiedDataset(manifest, config=config, return_trace=True)
    derived = UnifiedDataset(
        manifest, config=config, return_trace=True, derived_cache=opened(report, manifest)
    )
    assert raw.support_summary() == derived.support_summary()
    for seed in [42, 51]:
        values, states = [], []
        for dataset in [raw, derived]:
            torch.manual_seed(seed)
            dataset._augmentation_generator = None
            values.append([dataset[i] for i in [0, 1, 0, 1]])
            states.append(
                (
                    torch.get_rng_state(),
                    dataset._augmentation_generator.get_state()
                    if dataset._augmentation_generator is not None
                    else None,
                )
            )
        for left, right in zip(*values, strict=True):
            for a, b in zip(left, right, strict=True):
                assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        assert torch.equal(states[0][0], states[1][0])
        if states[0][1] is not None:
            assert torch.equal(states[0][1], states[1][1])


def test_negative_zero_is_not_quantized_and_invalid_targets_rejected(tmp_path):
    manifest, _ = corpus(tmp_path / "canonical")
    path = manifest.parent / "0.npz"
    with np.load(path) as labels:
        arrays = {key: labels[key] for key in labels.files}
    arrays["target"][0, 0, 0] = np.float32(-0.0)
    np.savez_compressed(path, **arrays)
    audit(manifest.parent)
    report = build_cache(manifest, tmp_path / "derived")
    with opened(report, manifest).labels("0.npz") as labels:
        assert labels["target"].dtype == np.float32
        assert np.signbit(labels["target"][0, 0, 0])
    arrays["target"][0, 0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    audit(manifest.parent)
    with pytest.raises(ValueError, match="Invalid canonical target"):
        build_cache(manifest, tmp_path / "invalid")


def test_training_pins_cache_outside_config_and_resume_allows_only_path_relocation(tmp_path):
    from deepretina.unified import train_unified

    manifest, _ = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived")
    config = UnifiedConfig(
        epochs=2,
        epoch_samples=2,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        norm="group",
        workers=0,
        require_all_heads=False,
        augmentation="basic",
        device="cpu",
    )
    kwargs = {
        "derived_cache_root": report["root"],
        "expected_cache_fingerprint": report["cache_fingerprint"],
        "expected_content_sha256": report["canonical_fingerprint"],
        "validation_fn": lambda *a, **kw: {"mean_dice": 0.5, "macro_dataset_mean_dice": 0.5},
    }
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / "run"
        train_unified(manifest, output, config, stop_after_epoch=1, **kwargs)
        saved = torch.load(output / "last.pt", weights_only=True)
        assert "derived_cache" not in saved["config"]
        assert saved["derived_cache"] == opened(report, manifest).identity
        relocated = tmp_path / "relocated-cache"
        shutil.copytree(report["root"], relocated)
        kwargs["derived_cache_root"] = relocated
        train_unified(manifest, output, config, resume=output / "last.pt", **kwargs)
        assert torch.load(output / "last.pt", weights_only=True)["epoch"] == 2
        with pytest.raises(ValueError, match="identity changed"):
            train_unified(
                manifest,
                output,
                config,
                resume=output / "last.pt",
                expected_content_sha256=report["canonical_fingerprint"],
            )
    finally:
        torch.set_num_threads(previous)


def test_capacity_estimate_matches_actual_complete_small_cache(tmp_path):
    import importlib.util
    from pathlib import Path

    manifest, _ = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "derived")
    path = Path(__file__).parents[1] / "scripts/estimate_derived_cache.py"
    spec = importlib.util.spec_from_file_location("estimate_derived_cache", path)
    estimator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(estimator)
    estimate = estimator.estimate(
        manifest, report["root"], report["cache_fingerprint"], report["canonical_fingerprint"]
    )
    assert estimate["photos"] == 3 and estimate["records"] == 3
    assert estimate["total_estimated_bytes"] == report["bytes"]
