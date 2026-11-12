"""Actual preview bytes/recipe and precision override; no quality evaluation."""

import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from deepretina.bundled import bundled_checkpoint, model_catalog
from deepretina.cli import main
from deepretina.unified import UnifiedConfig


def test_preview_recipe_exact_actual_checkpoint_except_portable_device():
    root = Path(__file__).parents[1]
    preset = tomllib.loads((root / "configs/unified-v3-tpe-preview.toml").read_text())
    config = UnifiedConfig(**preset)
    config.validate()
    with bundled_checkpoint("unified-v3-tpe-preview") as path:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert path.stat().st_size == 8894002
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == "20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4"
        )
    # Historical BEST58 predates the opt-in photometry field. Admit only its
    # explicit raw default; retain exact comparison for every other saved field.
    assert payload["config"].get("photometric_profile", "raw") == "raw"
    assert config.photometric_profile == "raw"
    expected = dict(payload["config"], device="auto")
    expected.setdefault("photometric_profile", "raw")
    assert asdict(config) == expected
    assert payload["config"]["lr"] == 0.000287722894608172
    assert payload["config"]["vascular_sampling_weight"] == 2.553934065831788
    assert payload["epoch"] == 58 and "training_state" not in payload
    assert model_catalog()["aliases"] == {
        "vessels": "legacy-vessels",
        "disc_cup": "legacy-universal",
        "av": "legacy-av",
    }


def test_cpu_precision_override_is_forwarded_without_training(tmp_path, monkeypatch, capsys):
    from deepretina import unified_cli

    called = []

    def train(manifest, output, config, **kwargs):
        called.append(config)
        return output / "best.pt"

    monkeypatch.setattr(unified_cli, "train_unified", train)
    config = Path(__file__).parents[1] / "configs/unified-v3-tpe-preview.toml"
    main(
        [
            "train-unified",
            "--manifest",
            str(tmp_path / "unused.json"),
            "--output",
            str(tmp_path / "run"),
            "--config",
            str(config),
            "--device",
            "cpu",
            "--mixed-precision",
            "none",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert len(called) == 1 and called[0].mixed_precision == "none" and called[0].device == "cpu"
    assert called[0].base_channels == 16 and called[0].augmentation == "basic"
    assert result["experimental"] is True and not (tmp_path / "run").exists()
