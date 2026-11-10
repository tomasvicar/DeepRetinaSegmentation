from pathlib import Path

import pytest

from deepretina.cli import main


def test_train_config_explicit_options_override_toml(tmp_path, monkeypatch, capsys):
    from deepretina import engine

    config = tmp_path / "settings.toml"
    config.write_text('task = "av"\nepochs = 50\nbase_channels = 16\n')
    seen = []

    def fake_train(manifest, output, settings):
        seen.append((manifest, output, settings))
        return output / "best.pt"

    monkeypatch.setattr(engine, "train", fake_train)
    main(
        [
            "train",
            "--manifest",
            "data/manifest.json",
            "--output",
            "runs/example",
            "--config",
            str(config),
            "--epochs",
            "3",
            "--device",
            "cpu",
        ]
    )
    manifest, output, settings = seen.pop()
    assert manifest == Path("data/manifest.json")
    assert output == Path("runs/example")
    assert (settings.task, settings.epochs, settings.base_channels, settings.device) == (
        "av",
        3,
        16,
        "cpu",
    )
    assert '"checkpoint": "runs/example/best.pt"' in capsys.readouterr().out


def test_train_rejects_unknown_config_before_execution(tmp_path, monkeypatch, capsys):
    from deepretina import engine

    config = tmp_path / "settings.toml"
    config.write_text('task = "av"\nepoches = 50\n')
    monkeypatch.setattr(engine, "train", lambda *args: pytest.fail("Training must not start"))
    with pytest.raises(SystemExit) as error:
        main(["train", "--manifest", "missing.json", "--output", "unused", "--config", str(config)])
    assert error.value.code == 2
    assert "Unknown training settings" in capsys.readouterr().err
