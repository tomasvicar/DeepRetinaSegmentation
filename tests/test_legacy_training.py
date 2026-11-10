"""CPU-only functional fresh-native training on synthetic prepared photographs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina import legacy_training as m
from deepretina.integrity import prepared_content_fingerprint
from deepretina.legacy.av import LegacyAVNet, checkpoint_payload, load_av
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.single_net import Unet


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    rows = []
    for split in ["train", "val", "test"]:
        for photo, task in [
            ("vascular", "av"),
            ("vascular", "av"),
            ("vascular", "vessels"),
            ("disc", "disc_cup"),
            ("disc", "disc_cup"),
        ]:
            ident = f"{split}-{task}-{len(rows)}"
            pid = f"{split}-{photo}"
            image = root / (pid + ".png")
            if not image.exists():
                rng = np.random.default_rng(len(rows))
                Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(image)
            channels = {"av": 3, "vessels": 1, "disc_cup": 2}[task]
            target = np.zeros((channels, 32, 48), np.float32)
            valid = np.ones_like(target, bool)
            target[0, 6:26, 8:40] = 1
            if task == "av":
                target[1, 6:16, 8:24] = 1
                target[2, 16:26, 24:40] = 1
                # Ambiguous/crossing vessel stays in the input, outside class CE.
                target[1:, 12, 30] = 1
                valid[1:, :2] = False
                valid[0, 0, 0] = False
                target[0, 0, 0] = 1
                if len(rows) % 2:
                    target[1:] = target[1:][::-1].copy()
                    target[0, 27, 6] = 1  # observer-specific ambiguous vessel
            if task == "disc_cup":
                target[1, 12:20, 16:32] = 1
                if len(rows) % 2:
                    valid[1] = False
            np.savez(
                root / (ident + ".npz"), target=target, valid=valid, fov=np.ones((32, 48), bool)
            )
            rows.append(
                {
                    "id": ident,
                    "photo_id": pid,
                    "source_id": pid,
                    "split_group_id": pid,
                    "split": split,
                    "task": task,
                    "dataset": "synthetic",
                    "image": image.name,
                    "mask": ident + ".npz",
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25.0,
                    "usable_for_training": True,
                }
            )
    Image.fromarray(np.full((32, 48), 255, np.uint8)).save(root / "fov.png")
    (root / "photos.json").write_text(
        json.dumps(
            [
                {"photo_id": pid, "image": pid + ".png", "fov": "fov.png"}
                for pid in sorted({r["photo_id"] for r in rows})
            ]
        )
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "_SUCCESS").write_text(digest)
    (root / "integrity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": digest,
                "fingerprint_schema": 2,
                "prepared_content_sha256": prepared_content_fingerprint(path),
                "preview_only": False,
            }
        )
    )
    return path


def tiny(task):
    if task == "av":
        arch = {
            "encoder_channels": [4, 2, 4, 8, 16],
            "decoder_channels": [16, 8, 4, 2],
            "head_channels": 4,
            "num_classes": 3,
        }
        return m.LegacyTrainingConfig(task=task, epochs=1, architecture=arch)
    return m.LegacyTrainingConfig(
        task=task,
        epochs=1,
        patch_size=32,
        architecture={
            "filters": 2,
            "in_size": 3,
            "out_size": len(m.HEADS[task]),
            "do": 0.0,
            "depth": 4,
        },
    )


@pytest.mark.parametrize("task", ["vessel", "disc", "universal", "av"])
def test_production_default_is_exact_native_constructor(task, monkeypatch):
    # Inspect safe metadata on meta device only; do not transfer old tensors.
    names = {
        "vessel": "legacy-single-net-vessels.pt",
        "disc": "legacy-single-net-disc-0.pt",
        "universal": "legacy-single-net-universal.pt",
        "av": "legacy-av-v9.pt",
    }
    path = Path(m.__file__).parent / "pretrained" / names[task]
    saved = torch.load(path, map_location="meta", weights_only=True)
    config = m.LegacyTrainingConfig(task=task)
    assert m.architecture_for(config) == saved["architecture"]
    with torch.device("meta"):
        fresh = m.build_fresh(config)
    assert isinstance(fresh, LegacyAVNet if task == "av" else Unet)
    assert {k: tuple(v.shape) for k, v in fresh.state_dict().items()} == {
        k: tuple(v.shape) for k, v in saved["state_dict"].items()
    }


@pytest.mark.parametrize("task", ["vessel", "disc", "universal", "av"])
def test_tiny_native_optimization_validation_safe_load_and_coverage(
    manifest, tmp_path, task, monkeypatch
):
    original = m.read_unified_sample

    def no_test(root, row):
        assert row["split"] in {"train", "val"}
        return original(root, row)

    monkeypatch.setattr(m, "read_unified_sample", no_test)
    config = tiny(task)
    torch.manual_seed(config.seed)
    initial = {k: v.clone() for k, v in m.build_fresh(config).state_dict().items()}
    output = tmp_path / ("run-" + task)

    def forbid_weight_loading(*args, **kwargs):
        raise AssertionError("Fresh native training must never load existing tensors")

    with monkeypatch.context() as context:
        context.setattr(torch, "load", forbid_weight_loading)
        best = m.train(manifest, output, config)
    model, payload = load_av(best) if task == "av" else load_single_net(best)
    last = torch.load(output / "last.pt", weights_only=True)
    assert any(
        not torch.equal(initial[k], v)
        for k, v in last["state_dict"].items()
        if v.is_floating_point() and k.endswith("weight")
    )
    assert (
        payload["lineage"]["loaded_checkpoint"] is None
        and payload["lineage"]["historical_tensor_transfer"] is False
    )
    assert payload["training_recipe"]["historical_optimizer_reproduction"] is False
    assert payload["architecture"] == m.architecture_for(config)
    assert payload["epoch"] == 1 and payload["validation"]["split"] == "val"
    coverage = payload["coverage"]
    dataset = m.NativeDataset(manifest, "train", config)
    assert set(coverage["photo_visits"]) == set(dataset.photo_ids)
    assert set(coverage["annotation_visits"]) == dataset.record_ids
    assert all(n > 0 for n in coverage["annotation_visits"].values())
    if task == "av":
        assert payload["preprocessing"]["profile"] == "legacy25_raw_av4"
        image, _, _, _ = dataset.sample(dataset.photo_ids[0])
    else:
        image = torch.rand(3, 32, 32) - 0.5
    with torch.no_grad():
        logits = model(image[None])
    assert (
        logits.shape[1] == (3 if task == "av" else len(m.HEADS[task]))
        and torch.isfinite(logits).all()
    )
    assert (
        json.loads((output / "training_summary.json").read_text())["test_evaluation_performed"]
        is False
    )


def test_av_filters_observers_before_sampling_and_preserves_full_frame_and_crossings(manifest):
    dataset = m.NativeDataset(manifest, "train", tiny("av"))
    assert len(dataset.photo_ids) == 1 and len(dataset.record_ids) == 2
    observed = []
    for _ in range(2):
        image, target, valid, ids = dataset.sample(dataset.photo_ids[0], crop=True)
        assert image.shape == (4, 32, 48)
        assert image[3, 12, 30] == 1 and not m.av_known(target, valid)[12, 30]
        assert image[3, 0, 0] == 0
        row = next(r for r in json.loads(manifest.read_text()) if r["id"] == ids[0])
        assert row["task"] == "av"
        raw_image, raw_target, raw_valid = m.read_unified_sample(manifest.parent, row)
        assert torch.equal(image[:3], raw_image - 0.5)
        assert torch.equal(image[3], ((raw_target[0] >= 0.5) & raw_valid[0]).float())
        assert torch.equal(target[1:3], raw_target[1:3])
        observed += ids
    assert set(observed) == dataset.record_ids


def test_masks_unknown_heads_and_unused_av_logit_have_exact_zero_gradient():
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 5, 8, 8)
    valid = torch.zeros_like(target, dtype=torch.bool)
    target[0, 1, 2, 3] = 1
    valid[0, 1:3, 2, 3] = True
    loss = m.native_loss(logits, target, valid, "av")
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    mask = torch.ones_like(logits.grad, dtype=torch.bool)
    mask[:, :, 2, 3] = False
    assert torch.count_nonzero(logits.grad[mask]) == 0
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    valid.zero_()
    valid[:, 0] = True
    m.native_loss(logits, target, valid, "universal").backward()
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0


def test_av_rejects_constant_channels_without_changing_original_arithmetic():
    target = torch.zeros(5, 16, 16)
    target[0, 4:12, 4:12] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    with pytest.raises(ValueError, match="constant channel"):
        m.av_input(torch.ones(3, 16, 16), target, valid, record_id="constant-rgb")
    target[0] = 0
    with pytest.raises(ValueError, match="constant channel"):
        m.av_input(torch.rand(3, 16, 16), target, valid, record_id="constant-vessel")


@pytest.mark.parametrize(
    "bad",
    [
        {"task": "av", "patch_size": 32},
        {"task": "av", "batch_size": 2},
        {"task": "vessel", "patch_size": 31},
    ],
)
def test_invalid_native_recipe_rejected(bad):
    with pytest.raises(ValueError):
        m.LegacyTrainingConfig(**bad).validate()


def test_preview_policy_and_existing_outputs_fail_before_training_writes(
    manifest, tmp_path, monkeypatch
):
    output = tmp_path / "preserved"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        m.train(manifest, output, tiny("disc"))
    assert (output / "sentinel").read_text() == "keep"

    def bad_policy(*args):
        raise ValueError("malformed policy")

    monkeypatch.setattr(m, "capture_geometry_policy", bad_policy)
    fresh = tmp_path / "not-created"
    with pytest.raises(ValueError, match="policy"):
        m.train(manifest, fresh, tiny("disc"))
    assert not fresh.exists()


def test_cli_shared_api_uses_explicit_config(manifest, tmp_path):
    parser = argparse.ArgumentParser()
    m.add_cli_args(parser)
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(m.asdict(tiny("vessel"))))
    args = parser.parse_args(
        ["--manifest", str(manifest), "--output", str(tmp_path / "cli-run"), "--config", str(cfg)]
    )
    result = m.train_from_args(args)
    assert result["task"] == "vessel" and result["fresh_initialization"] is True


def test_av_factory_default_is_unchanged_and_optional_metadata_is_copied():
    model = m.build_fresh(tiny("av"))
    baseline = checkpoint_payload(model)
    assert baseline["preprocessing"]["profile"] == "ves_class" and "geometry_policy" not in baseline
    profile = {"profile": "legacy25_raw_av4"}
    updated = checkpoint_payload(model, preprocessing=profile)
    profile["profile"] = "changed"
    assert updated["preprocessing"]["profile"] == "legacy25_raw_av4"


def test_unknown_av_labels_never_fall_back_to_vessel_only_observers(manifest):
    rows = json.loads(manifest.read_text())
    for row in rows:
        if row["split"] == "train" and row["task"] == "av":
            path = manifest.parent / row["mask"]
            with np.load(path) as archive:
                target = archive["target"]
                valid = archive["valid"]
                fov = archive["fov"]
            valid[1:] = False
            np.savez(path, target=target, valid=valid, fov=fov)
    with pytest.raises(ValueError, match="No eligible av"):
        m.NativeDataset(manifest, "train", tiny("av"))
    assert m.NativeDataset(manifest, "train", tiny("vessel")).record_ids


def test_observer_coverage_pass_cannot_be_replaced_by_smaller_sample_budget(manifest, tmp_path):
    config = tiny("av")
    config.epoch_samples = 1
    output = tmp_path / "too-small"
    with pytest.raises(ValueError, match="coverage-first"):
        m.train(manifest, output, config)
    assert not output.exists()


def test_cross_split_source_identity_leakage_is_rejected(manifest):
    rows = json.loads(manifest.read_text())
    rows[-1]["source_id"] = rows[0]["source_id"]
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="multiple splits"):
        m.NativeDataset(manifest, "train", tiny("vessel"))


def test_optional_av_geometry_policy_is_validated_and_copied():
    from deepretina.geometry_policy import validate_geometry_policy

    raw = (Path(__file__).parent / "reference/geometry_policy/policy.json").read_bytes()
    policy = {
        "format_version": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "document_json": raw.decode(),
    }
    validate_geometry_policy(policy)
    payload = checkpoint_payload(m.build_fresh(tiny("av")), geometry_policy=policy)
    policy["document_json"] = "{}"
    assert payload["geometry_policy"]["document_json"] == raw.decode()
    with pytest.raises(ValueError):
        checkpoint_payload(m.build_fresh(tiny("av")), geometry_policy=policy)


def test_exact_nonbinary_constant_rgb_cannot_evade_av_preflight_by_roundoff():
    image = torch.full((3, 17, 19), 0.1)
    target = torch.zeros(5, 17, 19)
    target[0, 4:12, 4:12] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    with pytest.raises(ValueError, match="constant channel"):
        m.av_input(image, target, valid, record_id="constant-point-one")


def test_native_cli_parser_is_stdlib_only(monkeypatch):
    import builtins
    import runpy

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name in {"deepretina.legacy_training", "legacy_training"}
        ):
            raise AssertionError("Parser imported model/training runtime: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = Path(m.__file__).with_name("legacy_training_cli.py")
    namespace = runpy.run_path(str(path), run_name="native_stdlib_parser_probe")
    parser = namespace["add_cli_args"](argparse.ArgumentParser())
    args = parser.parse_args(["--manifest", "manifest.json", "--output", "new-run", "--task", "av"])
    assert args.task == "av"
