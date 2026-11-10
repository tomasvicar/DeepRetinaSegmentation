import json

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.dataset import CropDataset, read_manifest
from deepretina.engine import (
    TrainConfig,
    _channel_metrics,
    evaluate,
    score_model,
    segmentation_loss,
    train,
)
from deepretina.inference import binary_masks, predict, tiled_probabilities
from deepretina.model import UNet


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_model_preserves_arbitrary_geometry():
    model = UNet(3, base_channels=4, depth=2)
    for h, w in [(35, 27), (1, 3), (32, 32)]:
        assert model(torch.zeros(1, 3, h, w)).shape == (1, 3, h, w)


def test_unknown_labels_do_not_contribute_gradient():
    logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
    target = torch.ones_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[:, 1] = False
    loss = segmentation_loss(logits, target, valid)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0
    assert torch.count_nonzero(logits.grad[:, 0]) == 64
    no_labels = segmentation_loss(logits, target, torch.zeros_like(valid))
    assert no_labels.item() == 0


def test_tiling_preserves_pixels_and_has_no_uncovered_edges():
    class PointModel(torch.nn.Module):
        def forward(self, image):
            return image[:, :1]

    image = torch.linspace(0, 1, 37 * 53).reshape(1, 37, 53).expand(3, -1, -1)
    result = tiled_probabilities(PointModel(), image, "cpu", tile_size=16, overlap=7)
    expected = ((image[:1] - 0.5) / 0.5).sigmoid().numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError):
        tiled_probabilities(PointModel(), image, "cpu", tile_size=16, overlap=16)


def test_hierarchy_and_av_conflict_resolution():
    probabilities = np.array([[[0.1, 0.8, 0.9]], [[0.8, 0.9, 0.8]], [[0.9, 0.8, 0.9]]], np.float32)
    masks = binary_masks(probabilities, "av")
    assert not masks[:, 0, 0].any()
    assert masks[1, 0, 1] and not masks[2, 0, 1]
    assert masks[2, 0, 2] and not masks[1, 0, 2]
    cup = binary_masks(probabilities[:2], "disc_cup")
    assert not (cup[1] & ~cup[0]).any()
    assert _channel_metrics(0, 0, 0, 64)["dice"] is None


def make_manifest(tmp_path, task="vessels"):
    count = {"vessels": 1, "av": 3, "disc_cup": 2}[task]
    rows = []
    for i, split in enumerate(["train", "val", "test"]):
        image = np.zeros((24, 28, 3), np.uint8)
        image[4:20, 8:16] = 230
        Image.fromarray(image).save(tmp_path / f"{i}.png")
        target = np.zeros((count, 24, 28), np.float32)
        target[:, 4:20, 8:16] = 1
        np.savez_compressed(
            tmp_path / f"{i}.npz", target=target, valid=np.ones_like(target, dtype=bool)
        )
        rows.append(
            {
                "id": str(i),
                "dataset": "synthetic",
                "task": task,
                "split": split,
                "image": f"{i}.png",
                "mask": f"{i}.npz",
                "preparation_max_size": 16,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    return manifest


@pytest.mark.parametrize("task", ["vessels", "av", "disc_cup"])
def test_train_save_reload_evaluate_predict(tmp_path, task):
    manifest = make_manifest(tmp_path, task)
    config = TrainConfig(
        task,
        epochs=1,
        batch_size=1,
        patch_size=32,
        base_channels=2,
        depth=1,
        samples_per_image=1,
        max_steps=1,
        device="cpu",
        tile_size=32,
        overlap=8,
    )
    checkpoint = train(manifest, tmp_path / "run", config)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["steps"] == 1
    assert saved["config"]["task"] == task
    assert saved["inference_max_size"] == 16
    metrics = evaluate(checkpoint, manifest, device="cpu", tile_size=32, overlap=8)
    assert metrics["images"] == 1
    assert metrics["mean_dice"] is not None
    assert metrics["by_dataset"]["synthetic"]["channels"] == metrics["channels"]
    assert metrics["metric_support"][
        "artery" if task == "av" else next(iter(metrics["channels"]))
    ] == (
        "reference_class_labeled_vessel_pixels"
        if task == "av"
        else "annotated_valid_pixels_within_field_of_view"
    )
    assert bool(metrics["av_accuracy_description"]) == (task == "av")
    assert bool(metrics["av_conditional_accuracy_description"]) == (task == "av")
    paths = predict(
        checkpoint,
        tmp_path / "2.png",
        tmp_path / "prediction",
        device="cpu",
        tile_size=32,
        overlap=8,
    )
    assert Image.open(paths["overlay"]).size == (28, 24)
    metadata = json.loads((tmp_path / "prediction" / "prediction.json").read_text())
    assert metadata["inference_shape"] == [14, 16]
    with np.load(paths["probabilities"]) as data:
        assert data["probability"].shape[-2:] == (24, 28)
    with pytest.raises(ValueError, match="already contains"):
        train(manifest, tmp_path / "run", config)


def test_crop_geometry_alignment_and_seed(tmp_path):
    manifest = make_manifest(tmp_path)
    dataset = CropDataset(manifest, "vessels", 32, 1)
    torch.manual_seed(19)
    image, target, valid = dataset[0]
    torch.manual_seed(19)
    repeated = dataset[0]
    assert all(torch.equal(a, b) for a, b in zip((image, target, valid), repeated))
    assert torch.equal(image[0] > 0, target[0] > 0)
    assert not valid.all()  # padding must not become supervised background


def test_crop_cache_is_bounded_and_does_not_mutate_source(tmp_path):
    manifest = make_manifest(tmp_path)
    dataset = CropDataset(manifest, "vessels", 32, 1)
    dataset[0]
    cached = tuple(t.clone() for t in dataset._cache[0])
    for _ in range(4):
        dataset[0]
    assert all(torch.equal(a, b) for a, b in zip(cached, dataset._cache[0], strict=True))
    dataset._cache_limit = 0
    dataset._cache.clear()
    dataset._cache_bytes = 0
    dataset[0]
    assert dataset._cache_bytes == 0
    assert not dataset._cache


def test_manifest_rejects_source_leakage_between_tasks(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = json.loads(manifest.read_text())
    rows[0]["source_id"] = "drive:01"
    rows[1]["source_id"] = "drive:01"
    rows[1]["task"] = "av"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="multiple splits"):
        read_manifest(manifest, "vessels", "train")


def test_av_accuracy_support_distinguishes_detection_from_classification(tmp_path):
    manifest = make_manifest(tmp_path, "av")
    target = np.zeros((3, 24, 28), np.float32)
    target[0, 0, :2] = 1
    target[1, 0, 0] = 1
    target[2, 0, 1] = 1
    valid = np.zeros_like(target, dtype=bool)
    valid[0] = True
    valid[1:, 0, :2] = True
    np.savez_compressed(tmp_path / "2.npz", target=target, valid=valid)

    class MissedVessel(torch.nn.Module):
        def forward(self, image):
            logits = torch.tensor([-10.0, 10.0, -10.0]).reshape(1, 3, 1, 1)
            return logits.expand(image.shape[0], 3, *image.shape[-2:])

    metrics = score_model(MissedVessel(), manifest, "av", "test", "cpu", 32, 8)
    assert metrics["av_accuracy"] == 0.0
    assert metrics["av_conditional_accuracy"] == 0.5
    assert metrics["av_labeled_pixels"] == 2
    assert metrics["channels"]["artery"]["valid_pixels"] == 2
    assert metrics["channels"]["vessel"]["valid_pixels"] == 24 * 28
