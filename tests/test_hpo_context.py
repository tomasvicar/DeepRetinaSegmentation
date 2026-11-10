"""Context, normalization and crop changes stay opt-in and keep label geometry."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from deepretina.dataset import CropDataset, letterbox_geometry, whole_view
from deepretina.engine import TrainConfig, evaluate, train
from deepretina.inference import (
    load_checkpoint,
    predict,
    probabilities_for_config,
    tiled_probabilities,
)
from deepretina.model import UNet


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def manifest(tmp_path, shape=(48, 80)):
    h, w = shape
    image = np.zeros((h, w, 3), np.uint8)
    image[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 220
    target = np.zeros((2, h, w), np.float32)
    target[0, h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 1
    target[1, h // 2 : h // 2 + 2, w // 2 : w // 2 + 2] = 1
    rows = []
    for i, split in enumerate(["train", "val", "test"]):
        Image.fromarray(image).save(tmp_path / f"{i}.png")
        np.savez_compressed(tmp_path / f"{i}.npz", target=target, valid=np.ones_like(target, bool))
        rows.append(
            {
                "id": str(i),
                "dataset": "synthetic",
                "task": "disc_cup",
                "split": split,
                "image": f"{i}.png",
                "mask": f"{i}.npz",
                "preparation_max_size": 80,
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows))
    return path


def test_whole_view_geometry_nearest_labels_and_padding():
    image = torch.ones(3, 10, 20)
    target = torch.ones(2, 10, 20)
    valid = torch.ones_like(target, dtype=torch.bool)
    rgb, labels, support = whole_view(image, target, valid, 32)
    assert letterbox_geometry(10, 20, 32) == (16, 32, 8, 0)
    assert rgb.shape == (3, 32, 32) and labels.shape == (2, 32, 32)
    assert torch.equal(labels.bool(), support)
    assert support[:, 8:24].all() and not support[:, :8].any() and not support[:, 24:].any()
    assert torch.all(rgb[:, 8:24] == 1) and torch.all(rgb[:, :8] == 0)


def test_whole_inference_exactly_uses_training_view_and_restores_geometry():
    class Point(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    image = torch.linspace(0, 1, 13 * 29).reshape(1, 13, 29).repeat(3, 1, 1)
    model = Point()
    model.train_view = "whole"
    model.view_size = 32
    actual = probabilities_for_config(model, image, "cpu")
    resized, _, _ = whole_view(image, size=32)
    nh, nw, top, left = letterbox_geometry(13, 29, 32)
    cropped = (((resized[:1] - 0.5) / 0.5).sigmoid())[None, ..., top : top + nh, left : left + nw]
    expected = F.interpolate(cropped, size=(13, 29), mode="bilinear", align_corners=False)[
        0
    ].numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1, 13, 29)


def test_default_dispatch_is_identical_to_original_tiling():
    model = UNet(2, 4, 2).eval()
    image = torch.rand(3, 35, 39)
    original = tiled_probabilities(model, image, "cpu", 16, 4)
    dispatched = probabilities_for_config(model, image, "cpu", tile_size=16, overlap=4)
    np.testing.assert_array_equal(original, dispatched)


def test_foreground_probability_one_always_keeps_a_positive_valid_pixel(tmp_path):
    path = manifest(tmp_path, shape=(100, 100))
    with np.load(tmp_path / "0.npz") as data:
        target = data["target"].copy()
        valid = data["valid"].copy()
    target[:] = 0
    target[0, 49, 49] = 1
    np.savez_compressed(tmp_path / "0.npz", target=target, valid=valid)
    dataset = CropDataset(path, "disc_cup", 16, 1, foreground_probability=1)
    torch.manual_seed(9)
    for _ in range(20):
        _, mask, support = dataset[0]
        assert (mask.bool() & support).any()


def test_basic_default_crop_draws_and_labels_remain_unchanged(tmp_path):
    path = manifest(tmp_path)
    default = CropDataset(path, "disc_cup", 32, 1)
    explicit = CropDataset(
        path, "disc_cup", 32, 1, train_view="patch", foreground_probability=0, augmentation="basic"
    )
    torch.manual_seed(8)
    a = default[0]
    torch.manual_seed(8)
    b = explicit[0]
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("norm", ["group", "batch"])
def test_whole_train_checkpoint_restore_predict(norm, tmp_path):
    path = manifest(tmp_path)
    config = TrainConfig(
        "disc_cup",
        epochs=2,
        max_steps=2,
        batch_size=1,
        patch_size=16,
        base_channels=2,
        depth=1,
        samples_per_image=1,
        device="cpu",
        train_view="whole",
        view_size=32,
        norm=norm,
        scheduler="cosine",
        tile_size=32,
        overlap=8,
    )
    checkpoint = train(path, tmp_path / "run", config)
    model, saved, _ = load_checkpoint(checkpoint, "cpu")
    assert model.train_view == "whole" and model.view_size == 32
    assert saved["config"] == asdict(config)
    stats = evaluate(checkpoint, path, device="cpu", tile_size=32, overlap=8)
    assert stats["images"] == 1
    paths = predict(
        checkpoint,
        tmp_path / "2.png",
        tmp_path / "prediction",
        device="cpu",
        tile_size=32,
        overlap=8,
    )
    assert Image.open(paths["overlay"]).size == (80, 48)
    metadata = json.loads((tmp_path / "prediction/prediction.json").read_text())
    assert metadata["train_view"] == "whole" and metadata["view_size"] == 32


def test_old_checkpoint_absent_new_fields_loads_original_defaults(tmp_path):
    torch.manual_seed(13)
    original = UNet(2, 4, 2).eval()
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 1,
            "config": {"task": "disc_cup", "base_channels": 4, "depth": 2},
            "model": original.state_dict(),
        },
        path,
    )
    loaded, _, _ = load_checkpoint(path, "cpu")
    image = torch.rand(1, 3, 35, 27)
    assert torch.equal(original(image), loaded(image))
    assert loaded.train_view == "patch" and loaded.view_size == 512


def test_augmentation_rng_does_not_change_basic_crop_sequence(tmp_path, monkeypatch):
    from deepretina import augment

    def private_draws(image, target, valid, *, level, generator):
        torch.rand(13, generator=generator)
        return image, target, valid

    monkeypatch.setattr(augment, "augment", private_draws)
    path = manifest(tmp_path)
    basic = CropDataset(path, "disc_cup", 32, 1, augmentation="basic")
    augmented = CropDataset(path, "disc_cup", 32, 1, augmentation="mild")
    torch.manual_seed(81)
    expected = [basic[0] for _ in range(4)]
    torch.manual_seed(81)
    actual = [augmented[0] for _ in range(4)]
    assert all(
        torch.equal(x, y)
        for before, after in zip(expected, actual, strict=True)
        for x, y in zip(before, after, strict=True)
    )


def test_balanced_source_sampling_preserves_explicit_budget_and_target_weight():
    from deepretina.engine import training_sampler

    class ImbalancedSources:
        rows = [{"dataset": "target"}] + [{"dataset": "donor"} for _ in range(9)]

        def __len__(self):
            return 80

    dataset = ImbalancedSources()
    config = TrainConfig("disc_cup", sampling="dataset_balanced", epoch_samples=10000)
    generator = torch.Generator().manual_seed(5)
    chosen = list(training_sampler(dataset, config, generator))
    assert len(chosen) == 10000
    target_fraction = sum(
        dataset.rows[index % 10]["dataset"] == "target" for index in chosen
    ) / len(chosen)
    assert 0.47 < target_fraction < 0.53
    repeated = list(training_sampler(dataset, config, torch.Generator().manual_seed(5)))
    assert chosen == repeated
    assert training_sampler(dataset, TrainConfig("disc_cup"), torch.Generator()) is None


def test_explicit_uniform_sampler_uses_requested_number_of_samples():
    from deepretina.engine import training_sampler

    class Dataset:
        def __init__(self):
            self.rows = [{"dataset": "target"}]

        def __len__(self):
            return 8

    sampler = training_sampler(
        Dataset(), TrainConfig("disc_cup", epoch_samples=37), torch.Generator().manual_seed(1)
    )
    chosen = list(sampler)
    assert len(chosen) == 37 and min(chosen) >= 0 and max(chosen) < 8
