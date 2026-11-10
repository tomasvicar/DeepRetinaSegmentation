"""Scientific contracts for one partially supervised five-output RGB network."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.unified import (
    CHANNELS,
    UnifiedConfig,
    UnifiedDataset,
    build_unified,
    decode_probabilities,
    evaluate_unified_model,
    load_unified,
    map_record_targets,
    predict_probabilities,
    predict_unified,
    read_unified_manifest,
    restore_probabilities,
    train_unified,
    unified_loss,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "task,count,indices", [("vessels", 1, [0]), ("av", 3, [0, 1, 2]), ("disc_cup", 2, [3, 4])]
)
def test_mapping_never_makes_missing_annotations_negative(task, count, indices):
    raw = np.ones((count, 3, 4), np.float32)
    valid = np.ones_like(raw, bool)
    target, support = map_record_targets({"id": "example", "task": task}, raw, valid)
    assert target.shape == (5, 3, 4) and support.shape == target.shape
    for index in range(5):
        assert support[index].all() == (index in indices)
        assert bool(target[index].all()) == (index in indices)


def test_unknown_heads_and_unclassified_av_pixels_have_zero_logit_gradient():
    logits = torch.zeros(1, 5, 3, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    target[:, 0] = 1
    valid[:, 0] = True
    loss, parts = unified_loss(logits, target, valid)
    loss.backward()
    assert set(parts) == {"vessel"}
    assert torch.count_nonzero(logits.grad[:, 0]) > 0
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0
    logits = torch.zeros(1, 5, 3, 4, requires_grad=True)
    valid[:] = False
    target[:] = 0
    target[0, 1, 0, 0] = 1
    target[0, 2, 0, 1] = 1
    valid[:, 1:3] = True
    loss, parts = unified_loss(logits, target, valid)
    loss.backward()
    assert set(parts) == {"av"}
    assert (
        torch.count_nonzero(logits.grad[:, 0]) == 0 and torch.count_nonzero(logits.grad[:, 3:]) == 0
    )
    assert torch.count_nonzero(logits.grad[:, 1:3, :, 2:]) == 0
    assert torch.equal(logits.grad[:, 1] + logits.grad[:, 2], torch.zeros_like(logits.grad[:, 1]))


def test_all_five_outputs_receive_correctly_masked_gradients_in_one_rgb_model():
    cfg = UnifiedConfig(base_channels=2, depth=1, norm="group", augmentation="basic")
    model = build_unified(cfg)
    image = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 5, 32, 32)
    valid = torch.zeros_like(target, dtype=torch.bool)
    target[:, 0, 4:28, 4:28] = 1
    target[:, 3, 8:24, 8:24] = 1
    target[:, 4, 12:20, 12:20] = 1
    valid[:, [0, 3, 4]] = True
    target[:, 1, 8:12, 8:12] = 1
    target[:, 2, 20:24, 20:24] = 1
    valid[:, 1:3, 8:12, 8:12] = True
    valid[:, 1:3, 20:24, 20:24] = True
    loss, parts = unified_loss(model((image - 0.5) / 0.5), target, valid)
    loss.backward()
    assert set(parts) == {"vessel", "av", "disc", "cup"}
    assert model.head.out_channels == 5 and model.encoders[0][0].in_channels == 3
    assert all(model.head.weight.grad[i].abs().sum() > 0 for i in range(5))


def test_conditional_av_is_offset_invariant_and_has_no_independent_class_cutoff():
    class Constant(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.shift = shift

        def forward(self, x):
            values = torch.tensor([3.0, -10.0 + self.shift, -11.0 + self.shift, 2.0, 3.0]).reshape(
                1, 5, 1, 1
            )
            return values.expand(x.shape[0], 5, *x.shape[-2:])

    image = torch.zeros(3, 35, 47)
    a = predict_probabilities(Constant(0), image, "cpu", tile_size=32, overlap=8)
    b = predict_probabilities(Constant(100), image, "cpu", tile_size=32, overlap=8)
    np.testing.assert_allclose(a, b, atol=1e-7)
    np.testing.assert_allclose(a[1] + a[2], 1, atol=2e-7)
    masks = decode_probabilities(a)
    assert masks[0].all() and masks[1].all() and not masks[2].any()
    assert not (masks[1] & masks[2]).any() and np.array_equal(masks[1] | masks[2], masks[0])
    a[3] = 0.1
    assert not decode_probabilities(a)[4].any()


def make_manifest(root):
    rows = []
    for split_index, split in enumerate(["train", "val", "test"]):
        for kind in ["av", "disc_cup"]:
            ident = f"{split}_{kind}"
            rgb = np.full((32, 40, 3), 70 + split_index * 20 + (5 if kind == "av" else 0), np.uint8)
            Image.fromarray(rgb).save(root / f"{ident}.png")
            count = 3 if kind == "av" else 2
            target = np.zeros((count, 32, 40), np.float32)
            valid = np.ones_like(target, bool)
            if kind == "av":
                target[0, 14:18, 10:30] = 1
                valid[1:] = False
                target[1, 16, 18] = 1
                target[2, 16, 21] = 1
                valid[1:, 16, 18] = True
                valid[1:, 16, 21] = True
            else:
                target[0, 6:26, 10:30] = 1
                target[1, 12:20, 16:24] = 1
            np.savez_compressed(
                root / f"{ident}.npz", target=target, valid=valid, fov=np.ones((32, 40), bool)
            )
            rows.append(
                {
                    "id": ident,
                    "dataset": "av_source" if kind == "av" else "disc_source",
                    "task": kind,
                    "split": split,
                    "source_id": f"{split}:{kind}",
                    "photo_id": f"{split}:{kind}",
                    "image": f"{ident}.png",
                    "mask": f"{ident}.npz",
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25.0,
                    "usable_for_training": True,
                }
            )
    path = root / "manifest.json"
    path.write_text(json.dumps(rows))
    return path


def test_coannotations_are_merged_per_photo_without_merging_patient_eyes(tmp_path):
    path = make_manifest(tmp_path)
    rows = json.loads(path.read_text())
    av = next(r for r in rows if r["id"] == "train_av")
    with np.load(tmp_path / av["mask"]) as labels:
        binary = np.zeros_like(labels["target"][:1])
        support = np.ones_like(binary, bool)
    np.savez_compressed(tmp_path / "second_vessel.npz", target=binary, valid=support)
    rows.append(dict(av, id="train_vessel_observer2", task="vessels", mask="second_vessel.npz"))
    # A patient's two different photographs stay separate even with a common source group.
    for row in rows:
        if row["split"] == "train":
            row["source_id"] = "patient:paired_eyes"
    path.write_text(json.dumps(rows))
    dataset = UnifiedDataset(path, "train", UnifiedConfig(augmentation="basic"))
    assert len(dataset) == 2
    index = next(i for i, g in enumerate(dataset.groups) if any(r["task"] == "av" for r in g))
    _, target, valid = dataset.sample(index)
    assert target[0, 16, 18] == 1 and target[1, 16, 18] == 1
    assert valid[0].all() and not valid[3:].any()
    assert len(dataset.groups[index]) == 2


def test_incorrect_photo_alias_pixels_and_cross_split_ids_are_rejected(tmp_path):
    path = make_manifest(tmp_path)
    rows = json.loads(path.read_text())
    train = [r for r in rows if r["split"] == "train"]
    train[1]["photo_id"] = train[0]["photo_id"]
    path.write_text(json.dumps(rows))
    dataset = UnifiedDataset(path, "train")
    with pytest.raises(ValueError, match="different RGB pixels"):
        dataset.sample(0)
    rows = json.loads(path.read_text())
    rows[-1]["source_id"] = rows[0]["source_id"]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="multiple splits"):
        read_unified_manifest(path, "train")


def test_checkpoint_roundtrip_and_functionality_label_cover_all_heads(tmp_path):
    path = make_manifest(tmp_path)
    cfg = UnifiedConfig(
        epochs=1,
        epoch_samples=4,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        norm="batch",
        workers=0,
        device="cpu",
        augmentation="basic",
        foreground_probability=0,
        tile_size=32,
        overlap=8,
        max_steps=2,
    )
    output = tmp_path / "run"
    checkpoint = train_unified(path, output, cfg)
    model, payload, device = load_unified(checkpoint, "cpu")
    assert payload["family"] == "unified" and payload["channels"] == list(CHANNELS)
    assert payload["supervised_channels"] == [True] * 5 and payload["purpose"] == "functionality"
    assert torch.load(output / "last.pt", weights_only=True)["training_state"]["optimizer"]
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["accuracy_claim"] is False
    assert metadata["support"]["datasets"] == {"av_source": 1, "disc_source": 1}
    scores = evaluate_unified_model(model, path, "test", device, cfg)
    assert scores["photos"] == 2 and all(
        scores["channels"][name]["dice"] is not None for name in CHANNELS
    )
    assert scores["av_labeled_pixels"] == 2
    raw_paths = predict_unified(
        checkpoint,
        tmp_path / "test_av.png",
        tmp_path / "prediction",
        device="cpu",
        fov=np.ones((32, 40), bool),
        field_of_view_degrees=40 / 25,
    )
    assert Image.open(raw_paths["overlay"]).size == (40, 32)
    with np.load(raw_paths["probabilities"], allow_pickle=False) as exported:
        assert exported["probability"].shape == (5, 32, 40)
        np.testing.assert_allclose(exported["probability"][1:3].sum(axis=0), 1, atol=3e-7)
    vessel = np.asarray(Image.open(raw_paths["vessel"])) > 0
    artery = np.asarray(Image.open(raw_paths["artery"])) > 0
    vein = np.asarray(Image.open(raw_paths["vein"])) > 0
    assert np.array_equal(artery | vein, vessel) and not (artery & vein).any()
    assert (
        json.loads((tmp_path / "prediction/prediction.json").read_text())["purpose"]
        == "functionality"
    )
    with pytest.raises(FileExistsError):
        train_unified(path, output, cfg)


@pytest.mark.parametrize("augmentation", ["basic", "mild"])
def test_epoch_boundary_resume_restores_optimizer_scheduler_and_rng_exactly(tmp_path, augmentation):
    path = make_manifest(tmp_path)
    cfg = UnifiedConfig(
        epochs=2,
        epoch_samples=4,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        norm="group",
        workers=0,
        device="cpu",
        augmentation=augmentation,
        foreground_probability=0.5,
        tile_size=32,
        overlap=8,
    )
    train_unified(path, tmp_path / "full", cfg)
    train_unified(path, tmp_path / "resumed", cfg, stop_after_epoch=1)
    first = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert first["epoch"] == 1
    train_unified(path, tmp_path / "resumed", cfg, resume=tmp_path / "resumed/last.pt")
    expected = torch.load(tmp_path / "full/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert actual["epoch"] == 2 and actual["steps"] == expected["steps"]
    assert all(
        torch.equal(expected["state_dict"][key], actual["state_dict"][key])
        for key in expected["state_dict"]
    )
    assert actual["val_mean_dice"] == expected["val_mean_dice"]
    assert actual["coverage"] == expected["coverage"]
    assert actual["coverage"]["seen_photos"] == 2
    assert sum(actual["coverage"]["photo_visits"].values()) == 8
    train_unified(
        path, tmp_path / "resumed", cfg, resume=tmp_path / "resumed/last.pt"
    )  # completed resume is harmless
    with pytest.raises(ValueError, match="configuration changed"):
        train_unified(
            path, tmp_path / "resumed", replace(cfg, lr=0.003), resume=tmp_path / "resumed/last.pt"
        )


def test_default_model_is_compact_and_still_shared():
    model = build_unified(UnifiedConfig())
    parameters = sum(p.numel() for p in model.parameters())
    assert 500000 < parameters < 1300000
    assert model.head.out_channels == 5


def test_inference_centers_rgb_only_once_and_rejects_centered_input():
    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, image):
            self.inputs.append(image.detach().clone())
            return torch.zeros(image.shape[0], 5, *image.shape[-2:])

    model = Capture()
    image = torch.linspace(0, 1, 3 * 32 * 40).reshape(3, 32, 40)
    predict_probabilities(model, image, "cpu", tile_size=32, overlap=8)
    assert min(float(t.min()) for t in model.inputs) == -0.5
    assert max(float(t.max()) for t in model.inputs) == 0.5
    with pytest.raises(ValueError, match="already-centered"):
        predict_probabilities(model, image - 0.5, "cpu", tile_size=32, overlap=8)


def test_full_data_rejects_wrong_physical_preprocessing_profile(tmp_path):
    path = make_manifest(tmp_path)
    rows = json.loads(path.read_text())
    rows[0]["preprocessing_profile"] = "modern_cap_1024"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="different preprocessing"):
        UnifiedDataset(path, "train", UnifiedConfig(purpose="full_data"))
    rows[0]["preprocessing_profile"] = "legacy25_raw"
    rows[0]["preparation_max_size"] = 1024
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="must not use modern"):
        UnifiedDataset(path, "train", UnifiedConfig(purpose="full_data"))


def test_tiled_conditional_fusion_weights_classes_by_vessel_confidence():
    class ContextDependent(torch.nn.Module):
        def forward(self, image):
            left = image[0, 0, 0, 0] < -0.3
            pv, qa = (0.4, 0.999) if left else (0.9, 0.1)
            values = torch.tensor(
                [np.log(pv / (1 - pv)), np.log(qa), np.log(1 - qa), 0.0, 0.0], dtype=torch.float32
            )
            return values[None, :, None, None].expand(image.shape[0], 5, *image.shape[-2:])

    image = torch.linspace(0, 1, 48)[None, None].expand(3, 32, 48).clone()
    probability = predict_probabilities(ContextDependent(), image, "cpu", tile_size=32, overlap=16)
    # At this overlap pixel naive Q-averaging chooses artery; joint fusion picks vein.
    assert probability[0, 16, 23] > 0.5
    assert probability[2, 16, 23] > probability[1, 16, 23]
    masks = decode_probabilities(probability)
    assert masks[2, 16, 23] and not masks[1, 16, 23]


def test_restoration_interpolates_joint_av_not_background_conditional_guesses():
    class Geometry:
        def restore(self, probability, *, order):
            return torch.nn.functional.interpolate(
                torch.from_numpy(probability)[None],
                size=(2, 3),
                mode="bilinear",
                align_corners=False,
            )[0].numpy()

    probability = np.ones((5, 2, 2), np.float32) * 0.5
    probability[0, :, 0] = 0.4
    probability[0, :, 1] = 0.9
    probability[1, :, 0] = 0.999
    probability[1, :, 1] = 0.1
    probability[2] = 1 - probability[1]
    restored = restore_probabilities(probability, Geometry())
    np.testing.assert_allclose(restored[1:3].sum(axis=0), 1, atol=1e-7)
    assert restored[0, 0, 1] > 0.5 and restored[2, 0, 1] > restored[1, 0, 1]


def test_module_cli_overrides_config_without_starting_unrequested_work(
    tmp_path, monkeypatch, capsys
):
    from deepretina import unified

    path = tmp_path / "settings.toml"
    path.write_text("epochs = 60\nbase_channels = 12\n")
    seen = []

    def capture(manifest, output, config, *, resume, stop_after_epoch):
        seen.append(config)
        return output / "best.pt"

    monkeypatch.setattr(unified, "train_unified", capture)
    unified.main(
        [
            "train",
            "--manifest",
            "prepared/manifest.json",
            "--output",
            "runs/example",
            "--config",
            str(path),
            "--epochs",
            "1",
            "--epoch-samples",
            "4",
            "--max-steps",
            "2",
            "--purpose",
            "functionality",
            "--device",
            "cpu",
        ]
    )
    assert seen[0].epochs == 1 and seen[0].epoch_samples == 4 and seen[0].max_steps == 2
    assert seen[0].purpose == "functionality" and seen[0].base_channels == 12
    assert '"purpose": "functionality"' in capsys.readouterr().out


def test_full_data_never_accepts_preview_records(tmp_path):
    path = make_manifest(tmp_path)
    rows = json.loads(path.read_text())
    rows[0]["preparation_preview"] = True
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="rejects preview"):
        UnifiedDataset(path, "train", UnifiedConfig(purpose="full_data"))
    assert len(UnifiedDataset(path, "train", UnifiedConfig(purpose="functionality"))) == 2


def test_coverage_sampler_visits_all_photos_then_balances_extras():
    from deepretina.unified import CoverageSampler

    generator = torch.Generator().manual_seed(42)
    sampler = CoverageSampler([0.01, 0.01, 100.0], 20, generator)
    indices = list(sampler)
    assert sorted(indices[:3]) == [0, 1, 2]
    assert indices[3:] == [2] * 17
    saved = generator.get_state()
    subsequent = list(sampler)
    generator.set_state(saved)
    assert subsequent == list(sampler)


def test_full_data_budget_must_cover_every_photo(tmp_path):
    manifest = make_manifest(tmp_path)
    cfg = UnifiedConfig(purpose="full_data", epoch_samples=1, workers=0)
    with pytest.raises(ValueError, match="epoch_samples >= train photos"):
        train_unified(manifest, tmp_path / "run", cfg)
    with pytest.raises(ValueError, match="coverage_first"):
        train_unified(
            manifest, tmp_path / "run", replace(cfg, epoch_samples=4, coverage_first=False)
        )


def test_prefetched_samples_do_not_count_as_delivered_coverage(tmp_path):
    manifest = make_manifest(tmp_path)
    cfg = UnifiedConfig(
        epochs=1,
        epoch_samples=20,
        batch_size=1,
        max_steps=1,
        workers=2,
        device="cpu",
        patch_size=32,
        base_channels=2,
        depth=1,
        norm="group",
        tile_size=32,
        overlap=8,
        augmentation="basic",
    )
    train_unified(manifest, tmp_path / "run", cfg)
    coverage = json.loads((tmp_path / "run/coverage.json").read_text())
    assert coverage["seen_photos"] == 1
    assert sum(coverage["photo_visits"].values()) == 1
    assert coverage["contributing_annotation_records"] == 1


def test_training_uses_each_vascular_observer_bundle_without_inconsistent_av(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = json.loads(manifest.read_text())
    av = next(row for row in rows if row["id"] == "train_av")
    np.savez(
        tmp_path / "binary.npz",
        target=np.zeros((1, 32, 40), np.float32),
        valid=np.ones((1, 32, 40), bool),
    )
    rows.append(dict(av, id="binary_observer", task="vessels", mask="binary.npz"))
    manifest.write_text(json.dumps(rows))
    dataset = UnifiedDataset(manifest)
    index = next(i for i, g in enumerate(dataset.groups) if len(g) == 2)
    seen = set()
    torch.manual_seed(11)
    for _ in range(40):
        _, target, valid, trace = dataset.sample(index, random_annotator=True, return_trace=True)
        seen.add(trace[0][0])
        if trace[0][0] == "binary_observer":
            assert not valid[1:3].any() and not target[0].any()
        else:
            assert valid[1:3].any() and target[0].any()
    assert seen == {"binary_observer", "train_av"}


@pytest.mark.parametrize(
    "metric,expected_epoch", [("mean_dice", 1), ("macro_dataset_mean_dice", 2)]
)
def test_best_checkpoint_uses_declared_selection_metric(tmp_path, metric, expected_epoch):
    manifest = make_manifest(tmp_path)
    config = UnifiedConfig(
        epochs=2,
        epoch_samples=2,
        batch_size=2,
        patch_size=32,
        base_channels=2,
        depth=1,
        workers=0,
        device="cpu",
        augmentation="basic",
        tile_size=32,
        overlap=8,
        selection_metric=metric,
    )
    scores = iter(
        [
            {"mean_dice": 0.9, "macro_dataset_mean_dice": 0.2},
            {"mean_dice": 0.8, "macro_dataset_mean_dice": 0.3},
        ]
    )
    path = train_unified(
        manifest, tmp_path / "selection", config, validation_fn=lambda *args, **kwargs: next(scores)
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == expected_epoch
    assert saved["config"]["selection_metric"] == metric
    validation = json.loads((path.parent / "validation.json").read_text())
    assert validation["selection_metric"] == metric
    assert validation["selection_score"] == saved["val_selection_score"]


def test_historical_resume_is_exact_without_original_checkpoint(tmp_path):
    import shutil
    from pathlib import Path

    manifest = make_manifest(tmp_path)
    teacher = tmp_path / "teacher.pt"
    shutil.copyfile(
        Path(__file__).parents[1] / "src/deepretina/pretrained/legacy-single-net-universal.pt",
        teacher,
    )
    cfg = UnifiedConfig(
        architecture="legacy_universal",
        legacy_initialization=str(teacher),
        allow_historical_exposure=True,
        epochs=2,
        epoch_samples=2,
        batch_size=2,
        patch_size=32,
        workers=0,
        device="cpu",
        augmentation="basic",
        tile_size=32,
        overlap=8,
    )
    train_unified(manifest, tmp_path / "full", cfg)
    train_unified(manifest, tmp_path / "resumed", cfg, stop_after_epoch=1)
    teacher.unlink()
    train_unified(manifest, tmp_path / "resumed", cfg, resume=tmp_path / "resumed/last.pt")
    expected = torch.load(tmp_path / "full/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert all(
        torch.equal(value, actual["state_dict"][key])
        for key, value in expected["state_dict"].items()
    )
    assert actual["lineage"] == expected["lineage"]
    assert actual["accuracy_claim"] is False
    assert actual["coverage"] == expected["coverage"]
