"""The shared package must include usable, intact inference checkpoints."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.bundled import bundled_checkpoint, model_catalog
from deepretina.cli import main


def test_original_and_explicit_pilot_models_exist_and_match_checksums():
    catalog = model_catalog()["models"]
    assert set(catalog) == {
        "legacy-vessels",
        "legacy-universal",
        "legacy-disc-0",
        "legacy-disc-1",
        "legacy-av",
        "pilot-vessels",
        "pilot-av",
        "pilot-disc-cup",
    }
    for name, info in catalog.items():
        with bundled_checkpoint(name) as checkpoint:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if name.startswith("pilot-"):
            assert saved["config"]["task"] == info["task"]
        else:
            assert saved["family"] == info["family"]
            assert "state_dict" in saved
        assert saved["channels"] == info["channels"]


def test_cli_predicts_without_dataset_or_external_checkpoint(tmp_path, capsys):
    image = tmp_path / "fundus.png"
    Image.fromarray(np.full((35, 45, 3), 100, np.uint8)).save(image)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        main(
            [
                "predict",
                "--model",
                "pilot-av",
                "--input",
                str(image),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
                "--tile-size",
                "32",
                "--overlap",
                "8",
            ]
        )
    finally:
        torch.set_num_threads(previous)
    result = json.loads(capsys.readouterr().out)
    assert Image.open(result["artery"]).size == (45, 35)
    with np.load(result["probabilities"], allow_pickle=False) as data:
        assert data["probability"].shape == (3, 35, 45)
        assert np.isfinite(data["probability"]).all()
    metadata = json.loads((tmp_path / "out/prediction.json").read_text())
    assert metadata["unsupported_channels"] == []


def test_corrupt_model_is_rejected_before_loading(tmp_path, monkeypatch):
    from deepretina import bundled

    registry = tmp_path / "pretrained"
    registry.mkdir()
    (registry / "broken.pt").write_bytes(b"truncated transmission")
    (registry / "registry.json").write_text(
        json.dumps(
            {"models": {"av": {"filename": "broken.pt", "sha256": "not-the-actual-checksum"}}}
        )
    )
    monkeypatch.setattr(bundled.resources, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"), bundled_checkpoint("av"):
        pytest.fail("Corrupt weights must not be exposed")
