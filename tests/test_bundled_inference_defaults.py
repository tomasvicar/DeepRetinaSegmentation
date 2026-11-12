"""Prediction-only defaults routing with fake metadata/predictors; no NN or weights."""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import deepretina
from deepretina import bundled, cli
from deepretina import bundled_inference as b


@pytest.fixture
def registry(monkeypatch):
    value = {
        "aliases": {"alias": "unified"},
        "models": {
            "unified": {
                "family": "unified",
                "inference_defaults": {
                    "tta": "rot4",
                    "threshold": 0.35,
                    "tile_size": 768,
                    "overlap": 192,
                },
            },
            "old-unified": {"family": "unified"},
            "original": {"family": "legacy_single_net"},
            "pilot": {"family": "pilot"},
        },
    }
    monkeypatch.setattr(b, "model_catalog", lambda: value)
    monkeypatch.setattr(bundled, "model_catalog", lambda: value)
    return value


def test_model_and_alias_defaults_and_explicit_overrides(registry):
    expected = {"tta": "rot4", "threshold": 0.35, "tile_size": 768, "overlap": 192}
    assert (
        b.resolve_prediction_options("alias") == b.resolve_prediction_options("unified") == expected
    )
    assert b.resolve_prediction_options(
        "alias", tta="none", threshold=0.5, tile_size=512, overlap=64
    ) == {"tta": "none", "threshold": 0.5, "tile_size": 512, "overlap": 64}
    assert b.resolve_prediction_options("alias", threshold=0.4) == {**expected, "threshold": 0.4}


@pytest.mark.parametrize("model", [None, "old-unified", "original", "pilot"])
def test_historical_defaults_and_saved_tiles_unchanged(registry, model):
    assert b.resolve_prediction_options(model) == {
        "tta": "none",
        "threshold": 0.5,
        "tile_size": None,
        "overlap": None,
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"other": 1},
        None,
        [],
        {"tta": "bogus"},
        {"threshold": True},
        {"threshold": float("nan")},
        {"threshold": 0},
        {"threshold": 1},
        {"tile_size": True},
        {"tile_size": 7},
        {"overlap": -1},
        {"tile_size": 32, "overlap": 32},
    ],
)
def test_invalid_stored_defaults_cannot_be_hidden_by_overrides(registry, settings):
    registry["models"]["unified"]["inference_defaults"] = settings
    with pytest.raises(ValueError):
        b.resolve_prediction_options(
            "unified", tta="none", threshold=0.5, tile_size=512, overlap=64
        )


@pytest.mark.parametrize("model", ["original", "pilot"])
def test_nonunified_cannot_advertise_defaults(registry, model):
    registry["models"][model]["inference_defaults"] = {}
    with pytest.raises(ValueError, match="bundled unified"):
        b.resolve_prediction_options(model)


def test_unknown_model_and_option_rejected(registry):
    with pytest.raises(ValueError, match="Unknown bundled"):
        b.resolve_prediction_options("missing")
    with pytest.raises(ValueError, match="Unknown inference"):
        b.resolve_prediction_options("unified", preset="unreviewed")


@contextmanager
def fake_checkpoint(name):
    yield Path("no-model-read.pt")


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], ("rot4", 0.35, 768, 192)),
        (
            ["--tta", "none", "--threshold", ".5", "--tile-size", "512", "--overlap", "64"],
            ("none", 0.5, 512, 64),
        ),
    ],
)
def test_CLI_to_mocked_unified_predict_preserves_resolved_protocol(
    registry, monkeypatch, tmp_path, extra, expected
):
    seen = []
    monkeypatch.setattr(bundled, "bundled_checkpoint", fake_checkpoint)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: {"family": "unified"})
    )
    monkeypatch.setitem(
        sys.modules,
        "deepretina.unified_cli",
        SimpleNamespace(
            apply_unified=lambda args, path: seen.append(
                (args.tta, args.threshold, args.tile_size, args.overlap)
            )
        ),
    )
    cli.main(
        [
            "predict",
            "--model",
            "alias",
            "--input",
            "unread.png",
            "--output",
            str(tmp_path / "new"),
            *extra,
        ]
    )
    assert seen == [expected]


def test_custom_checkpoint_fallback_and_evaluation_not_preset(registry, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        cli,
        "_apply_model",
        lambda args, path: seen.append(
            (args.command, args.tta, args.threshold, args.tile_size, args.overlap)
        ),
    )
    monkeypatch.setattr(bundled, "bundled_checkpoint", fake_checkpoint)
    cli.main(
        [
            "predict",
            "--checkpoint",
            "unread.pt",
            "--input",
            "unread.png",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    cli.main(
        [
            "evaluate",
            "--model",
            "alias",
            "--manifest",
            "unread.json",
            "--split",
            "val",
            "--output",
            str(tmp_path / "evaluation.json"),
        ]
    )
    assert seen == [("predict", "none", 0.5, None, None), ("evaluate", "none", 0.5, None, None)]


def test_public_API_shares_resolver_and_forwards_raw_input_options(registry, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(b, "bundled_checkpoint", fake_checkpoint)
    monkeypatch.setitem(
        sys.modules,
        "deepretina.unified",
        SimpleNamespace(
            predict_unified=lambda checkpoint, image, output, **kwargs: seen.append(
                (checkpoint, image, output, kwargs)
            )
        ),
    )
    deepretina.predict_bundled_unified(
        "alias",
        "unread.png",
        tmp_path / "new",
        device="cpu",
        dataset="drive",
        threshold=0.5,
        tta="none",
    )
    assert seen[0][1] == Path("unread.png")
    assert seen[0][3] == {
        "tta": "none",
        "threshold": 0.5,
        "tile_size": 768,
        "overlap": 192,
        "device": "cpu",
        "dataset": "drive",
    }
    with pytest.raises(ValueError, match="bundled unified"):
        deepretina.predict_bundled_unified("original", "unread.png", tmp_path / "other")


def test_all_existing_registry_models_and_aliases_keep_previous_defaults():
    real = bundled.model_catalog()
    for name in list(real["models"]) + list(real["aliases"]):
        assert "inference_defaults" not in real["models"][real.get("aliases", {}).get(name, name)]
        assert b.resolve_prediction_options(name, registry=real) == {
            "tta": "none",
            "threshold": 0.5,
            "tile_size": None,
            "overlap": None,
        }


def test_loader_API_is_lazy_and_no_tensor_calls_in_routing():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, deepretina; from deepretina.bundled_inference import resolve_prediction_options; assert callable(deepretina.predict_bundled_unified); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "option",
    [
        {"tta": "bogus"},
        {"threshold": True},
        {"threshold": float("inf")},
        {"threshold": 0},
        {"threshold": 1},
        {"tile_size": True},
        {"overlap": -1},
    ],
)
def test_invalid_explicit_unified_options_rejected(registry, option):
    with pytest.raises(ValueError):
        b.resolve_prediction_options("unified", **option)


def test_original_and_custom_explicit_values_keep_backend_contract(registry):
    # Preserve the old backend's validation/threshold semantics, including0/1.
    for model in (None, "original", "pilot"):
        assert b.resolve_prediction_options(model, threshold=0)["threshold"] == 0
