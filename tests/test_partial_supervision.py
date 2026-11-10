"""Partial optic-disc labels must never produce a supposedly trained cup output."""

import json

import numpy as np
import torch
from PIL import Image

from deepretina.dataset import read_manifest
from deepretina.engine import TrainConfig, evaluate, train
from deepretina.inference import predict


def test_disc_only_checkpoint_marks_and_suppresses_untrained_cup(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = []
        for i, split in enumerate(("train", "val", "test")):
            image = np.full((16, 16, 3), 100, np.uint8)
            Image.fromarray(image).save(tmp_path / f"{i}.png")
            target = np.zeros((2, 16, 16), np.float32)
            target[0, 4:12, 4:12] = 1
            target[1, 6:10, 6:10] = 1  # Exists only in held-out test annotation.
            valid = np.ones_like(target, bool)
            if split != "test":
                valid[1] = False
            np.savez_compressed(tmp_path / f"{i}.npz", target=target, valid=valid)
            rows.append(
                {
                    "id": str(i),
                    "source_id": f"onhsd:{i}",
                    "dataset": "onhsd",
                    "task": "disc_cup",
                    "split": split,
                    "image": f"{i}.png",
                    "mask": f"{i}.npz",
                    "usable_for_training": True,
                }
            )
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(rows))
        config = TrainConfig(
            "disc_cup",
            epochs=1,
            max_steps=1,
            batch_size=1,
            patch_size=16,
            base_channels=2,
            depth=1,
            samples_per_image=1,
            device="cpu",
            tile_size=16,
            overlap=4,
        )
        checkpoint = train(manifest, tmp_path / "run", config)
        saved = torch.load(checkpoint, weights_only=True)
        assert saved["supervised_channels"] == [True, False]
        scores = evaluate(checkpoint, manifest, device="cpu", tile_size=16, overlap=4)
        assert scores["channels"]["disc"]["dice"] is not None
        assert scores["channels"]["cup"]["dice"] is None
        assert scores["channels"]["cup"]["valid_pixels"] == 0
        paths = predict(
            checkpoint, tmp_path / "2.png", tmp_path / "pred", device="cpu", tile_size=16, overlap=4
        )
        assert not np.asarray(Image.open(paths["cup"])).any()
        info = json.loads((tmp_path / "pred/prediction.json").read_text())
        assert info["unsupported_channels"] == ["cup"]
        with np.load(paths["probabilities"], allow_pickle=False) as data:
            assert data["supervised_channels"].tolist() == [True, False]
    finally:
        torch.set_num_threads(previous)


def test_unusable_annotation_is_preserved_but_excluded_from_training(tmp_path):
    rows = [
        {
            "id": "bad",
            "image": "bad.png",
            "task": "disc_cup",
            "split": "train",
            "source_id": "onhsd:bad",
            "usable_for_training": False,
        },
        {
            "id": "good",
            "image": "good.png",
            "task": "disc_cup",
            "split": "train",
            "source_id": "onhsd:good",
            "usable_for_training": True,
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    assert [row["id"] for row in read_manifest(manifest, "disc_cup", "train")] == ["good"]
