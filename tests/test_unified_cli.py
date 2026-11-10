"""Public unified CLI round trips use generated functionality data, never quality claims."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepretina.cli import main
from deepretina.derived_cache import build_cache
from deepretina.integrity import prepared_content_fingerprint
from deepretina.unified import UnifiedConfig


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def corpus(root):
    root.mkdir()
    rows, photos = [], []
    for number, split in enumerate(("train", "train", "val")):
        rng = np.random.default_rng(number)
        image = rng.integers(30, 220, (64, 64, 3), np.uint8)
        image_name, fov_name = f"{number}.png", f"{number}-fov.png"
        Image.fromarray(image).save(root / image_name)
        Image.fromarray(np.full((64, 64), 255, np.uint8)).save(root / fov_name)
        av = np.zeros((3, 64, 64), np.float32)
        av[0, 12:52, 20:44] = 1
        av[1, 12:32, 20:44] = 1
        av[2, 32:52, 20:44] = 1
        dc = np.zeros((2, 64, 64), np.float32)
        dc[0, 16:48, 16:48] = 1
        dc[1, 24:40, 24:40] = 1
        for task, target in [("av", av), ("disc_cup", dc)]:
            mask = f"{number}-{task}.npz"
            valid = np.ones_like(target, bool)
            if task == "av":
                valid[1:] = target[0].astype(bool)
            np.savez_compressed(
                root / mask, target=target, valid=valid, fov=np.ones((64, 64), bool)
            )
            rows.append(
                {
                    "id": f"{number}-{task}",
                    "dataset": "synthetic",
                    "source_id": f"subject-{number}",
                    "photo_id": f"photo-{number}",
                    "split": split,
                    "task": task,
                    "image": image_name,
                    "mask": mask,
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25,
                }
            )
        photos.append({"image": image_name, "fov": fov_name})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows))
    (root / "photos.json").write_text(json.dumps(photos))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "_SUCCESS").write_text(digest)
    content = prepared_content_fingerprint(manifest)
    (root / "integrity.json").write_text(
        json.dumps(
            {"fingerprint_schema": 2, "manifest_sha256": digest, "prepared_content_sha256": content}
        )
    )
    return manifest, content


def tiny_config():
    return UnifiedConfig(
        base_channels=2,
        depth=1,
        norm="group",
        epochs=2,
        epoch_samples=2,
        batch_size=1,
        workers=0,
        patch_size=32,
        tile_size=32,
        overlap=8,
        augmentation="basic",
        device="cpu",
        purpose="functionality",
    )


def write_config(path, config):
    settings = asdict(config)
    if path.suffix == ".json":
        path.write_text(json.dumps(settings))
    else:
        # TOML has no null; omitted optional values retain dataclass defaults.
        path.write_text(
            "\n".join(
                f"{key} = {json.dumps(value)}"
                for key, value in settings.items()
                if value is not None
            )
        )


@pytest.fixture
def trained(tmp_path, capsys):
    manifest, pin = corpus(tmp_path / "data")
    config = tmp_path / "settings.json"
    write_config(config, tiny_config())
    output = tmp_path / "train"
    main(
        [
            "train-unified",
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--output",
            str(output),
            "--expected-content-sha256",
            pin,
            "--device",
            "cpu",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["experimental"] and result["purpose"] == "functionality"
    return Path(result["checkpoint"]), manifest


def predict_args(checkpoint, manifest, output):
    return [
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(manifest.parent / "0.png"),
        "--fov",
        str(manifest.parent / "0-fov.png"),
        "--field-of-view-degrees",
        "2.56",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]


def test_actual_generated_checkpoint_predicts_five_native_outputs_and_default_tiles(
    trained, tmp_path, capsys, monkeypatch
):
    from deepretina import unified

    checkpoint, manifest = trained
    called = []
    actual = unified.predict_probabilities

    def observe(model, image, device, config, tile_size, overlap):
        called.append((tile_size, overlap))
        return actual(model, image, device, config, tile_size, overlap)

    monkeypatch.setattr(unified, "predict_probabilities", observe)
    main(predict_args(checkpoint, manifest, tmp_path / "prediction"))
    paths = json.loads(capsys.readouterr().out)
    info = json.loads(Path(paths["metadata"]).read_text())
    assert called == [(32, 8)]
    assert info["family"] == "unified" and info["purpose"] == "functionality"
    assert info["inference"]["overrides"] == {}
    assert info["inference"]["tile_size"] == 32
    for name in ["vessel", "artery", "vein", "disc", "cup", "av", "overlay"]:
        assert Image.open(paths[name]).size == (64, 64)
    with np.load(paths["probabilities"]) as saved:
        assert saved["probability"].shape == (5, 64, 64)
        np.testing.assert_allclose(saved["probability"][1:3].sum(0), 1, atol=1e-6)
    main(
        predict_args(checkpoint, manifest, tmp_path / "override")
        + ["--tile-size", "48", "--overlap", "16"]
    )
    paths = json.loads(capsys.readouterr().out)
    info = json.loads(Path(paths["metadata"]).read_text())
    assert called[-1] == (48, 16)
    assert info["inference"]["overrides"] == {"tile_size": 48, "overlap": 16}


@pytest.mark.parametrize("suffix", [".json", ".toml"])
def test_public_training_resume_restores_config_and_content_pin_exactly(tmp_path, capsys, suffix):
    manifest, pin = corpus(tmp_path / "data")
    settings = tmp_path / f"settings{suffix}"
    write_config(settings, tiny_config())
    common = [
        "train-unified",
        "--manifest",
        str(manifest),
        "--config",
        str(settings),
        "--expected-content-sha256",
        pin,
        "--device",
        "cpu",
    ]
    main(common + ["--output", str(tmp_path / "full")])
    capsys.readouterr()
    main(common + ["--output", str(tmp_path / "resume"), "--stop-after-epoch", "1"])
    capsys.readouterr()
    last = tmp_path / "resume/last.pt"
    assert torch.load(last, weights_only=True)["epoch"] == 1
    main(
        [
            "train-unified",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "resume"),
            "--resume",
            str(last),
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(last, weights_only=True)
    assert b["epoch"] == 2 and b["prepared_content_sha256"] == pin
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key])


def test_public_validation_requires_same_pinned_corpus_and_rejects_test(trained, tmp_path, capsys):
    checkpoint, manifest = trained
    args = [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "metrics.json"),
        "--device",
        "cpu",
    ]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert "frozen-selection" in capsys.readouterr().err
    assert not (tmp_path / "metrics.json").exists()
    main(args + ["--split", "val"])
    result = json.loads(capsys.readouterr().out)
    assert result["split"] == "val" and result["test_evaluation_performed"] is False
    assert result["prepared_content_sha256"]
    assert result["inference"]["tile_size"] == 32
    (tmp_path / "metrics.json").unlink()
    target = manifest.parent / "2-disc_cup.npz"
    with np.load(target) as stored:
        values = {key: stored[key] for key in stored.files}
    values["target"][0, 10, 10] = 1
    np.savez_compressed(target, **values)
    with pytest.raises(SystemExit) as exc:
        main(args + ["--split", "val"])
    assert exc.value.code == 2
    assert "content changed" in capsys.readouterr().err
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--normalized"],
        ["--prepared"],
        ["--prepared-array", "missing.npy"],
        ["--dicoms", "r", "g", "b", "v"],
        ["--av-raw-proxy"],
        ["--av-raw-matlab"],
        ["--vessel-mask", "vessel.png"],
        ["--vessel-checkpoint", "vessel.pt"],
        ["--vessel-normalized"],
        ["--fov-diameter-pixels", "100"],
        ["--original-size", "64", "64"],
        ["--pixels-per-degree", "35"],
        ["--overlap", "32"],
    ],
)
def test_unified_prediction_rejects_inapplicable_modes_before_outputs(
    trained, tmp_path, capsys, options
):
    checkpoint, manifest = trained
    with pytest.raises(SystemExit) as exc:
        main(predict_args(checkpoint, manifest, tmp_path / "bad") + options)
    assert exc.value.code == 2
    assert not (tmp_path / "bad").exists()
    capsys.readouterr()


def test_folder_and_whole_view_override_contract(trained, tmp_path, capsys):
    checkpoint, manifest = trained
    folder = tmp_path / "images"
    folder.mkdir()
    for n in [0, 1]:
        shutil.copy(manifest.parent / f"{n}.png", folder / f"{n}.png")
    args = [
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(folder),
        "--field-of-view-degrees",
        "2.56",
        "--output",
        str(tmp_path / "batch"),
        "--device",
        "cpu",
    ]
    main(args)
    paths = json.loads(capsys.readouterr().out)
    assert len(paths) == 2
    assert all(Image.open(p["vessel"]).size == (64, 64) for p in paths)
    with pytest.raises(SystemExit):
        main(args + ["--fov", str(manifest.parent / "0-fov.png")])
    capsys.readouterr()
    payload = torch.load(checkpoint, weights_only=True)
    payload["config"]["train_view"] = "whole"
    whole = tmp_path / "whole.pt"
    torch.save(payload, whole)
    with pytest.raises(SystemExit) as exc:
        main(predict_args(whole, manifest, tmp_path / "bad") + ["--tile-size", "48"])
    assert exc.value.code == 2
    assert "Whole-view" in capsys.readouterr().err
    assert not (tmp_path / "bad").exists()


def test_pilot_tile_defaults_remain_512_64(tmp_path, monkeypatch, capsys):
    import deepretina.inference as pilot

    checkpoint = tmp_path / "pilot.pt"
    torch.save({"family": "pilot"}, checkpoint)
    called = []

    def capture(*args, **kwargs):
        called.append(kwargs)
        return {}

    monkeypatch.setattr(pilot, "predict", capture)
    main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--input",
            "synthetic.png",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    capsys.readouterr()
    assert called[0]["tile_size"] == 512 and called[0]["overlap"] == 64


def test_unified_validation_rejects_unpinned_checkpoint_and_changed_manifest(
    trained, tmp_path, capsys
):
    checkpoint, manifest = trained
    payload = torch.load(checkpoint, weights_only=True)
    payload["prepared_content_sha256"] = None
    unpinned = tmp_path / "unpinned.pt"
    torch.save(payload, unpinned)
    common = [
        "evaluate",
        "--manifest",
        str(manifest),
        "--split",
        "val",
        "--output",
        str(tmp_path / "bad.json"),
        "--device",
        "cpu",
    ]
    with pytest.raises(SystemExit):
        main(common + ["--checkpoint", str(unpinned)])
    assert "checkpoint-pinned" in capsys.readouterr().err
    changed = tmp_path / "different.json"
    changed.write_text(manifest.read_text() + "\n")
    common[common.index("--manifest") + 1] = str(changed)
    with pytest.raises(SystemExit):
        main(common + ["--checkpoint", str(checkpoint)])
    assert "manifest differs" in capsys.readouterr().err
    assert not (tmp_path / "bad.json").exists()


def test_unified_training_rejects_unknown_or_nonobject_config(tmp_path, capsys):
    config = tmp_path / "bad.json"
    for settings in [{"typo_epochs": 1}, [], {"lr": {"value": 0.01}}]:
        config.write_text(json.dumps(settings))
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "train-unified",
                    "--config",
                    str(config),
                    "--manifest",
                    "missing.json",
                    "--output",
                    str(tmp_path / "bad"),
                ]
            )
        assert exc.value.code == 2
        capsys.readouterr()
        assert not (tmp_path / "bad").exists()


def test_existing_validation_report_is_preserved_before_model_or_evaluator(
    trained, tmp_path, capsys, monkeypatch
):
    import deepretina.unified_cli as public

    checkpoint, manifest = trained
    report = tmp_path / "existing.json"
    report.write_text("scientific report to preserve\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing report must be rejected before loading model or scoring")

    monkeypatch.setattr(public, "load_unified", forbidden)
    monkeypatch.setattr(public, "evaluate_unified_model", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--split",
                "val",
                "--output",
                str(report),
                "--device",
                "cpu",
            ]
        )
    assert error.value.code == 2
    assert "preserved" in capsys.readouterr().err
    assert report.read_text() == "scientific report to preserve\n"


def test_validation_exclusive_write_handles_late_report_and_hashes_loaded_snapshot(
    trained, tmp_path, capsys, monkeypatch
):
    import io

    import deepretina.unified_cli as public

    checkpoint, manifest = trained
    snapshot = checkpoint.read_bytes()
    original = torch.load(io.BytesIO(snapshot), weights_only=True)
    key = next(k for k, value in original["state_dict"].items() if value.is_floating_point())
    replacement = dict(original)
    replacement["state_dict"] = dict(original["state_dict"])
    replacement["state_dict"][key] = original["state_dict"][key] + 1
    replacement_path = tmp_path / "replacement.pt"
    torch.save(replacement, replacement_path)
    report = tmp_path / "metrics.json"

    def replace_during_evaluation(model, *args, **kwargs):
        assert torch.equal(model.state_dict()[key], original["state_dict"][key])
        checkpoint.write_bytes(replacement_path.read_bytes())
        return {"split": "val"}

    monkeypatch.setattr(public, "evaluate_unified_model", replace_during_evaluation)
    command = [
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--split",
        "val",
        "--output",
        str(report),
        "--device",
        "cpu",
    ]
    main(command)
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert result["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report.unlink()
    checkpoint.write_bytes(snapshot)

    def publish_another_report(*args, **kwargs):
        report.write_text("concurrent scientific result")
        return {"split": "val"}

    monkeypatch.setattr(public, "evaluate_unified_model", publish_another_report)
    with pytest.raises(SystemExit) as error:
        main(command)
    assert error.value.code == 2
    assert report.read_text() == "concurrent scientific result"
    capsys.readouterr()


def test_prediction_hash_and_weights_use_same_snapshot_when_checkpoint_path_replaced(
    trained, tmp_path, capsys, monkeypatch
):
    import io

    from deepretina import unified

    checkpoint, manifest = trained
    snapshot = checkpoint.read_bytes()
    expected_sha = hashlib.sha256(snapshot).hexdigest()
    main(predict_args(checkpoint, manifest, tmp_path / "before"))
    before = json.loads(capsys.readouterr().out)
    payload = torch.load(io.BytesIO(snapshot), weights_only=True)
    key = next(k for k, value in payload["state_dict"].items() if value.is_floating_point())
    payload["state_dict"][key] += 1
    replacement = tmp_path / "replacement.pt"
    torch.save(payload, replacement)
    actual_load = torch.load
    intercepted = []

    def replace_after_snapshot(source, *args, **kwargs):
        if isinstance(source, io.BytesIO):
            assert kwargs["weights_only"] is True
            checkpoint.write_bytes(replacement.read_bytes())
            intercepted.append(True)
        return actual_load(source, *args, **kwargs)

    monkeypatch.setattr(unified.torch, "load", replace_after_snapshot)
    main(predict_args(checkpoint, manifest, tmp_path / "after"))
    after = json.loads(capsys.readouterr().out)
    info = json.loads(Path(after["metadata"]).read_text())
    assert intercepted and info["checkpoint_sha256"] == expected_sha
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha
    with np.load(before["probabilities"]) as a, np.load(after["probabilities"]) as b:
        np.testing.assert_array_equal(a["probability"], b["probability"])


def test_public_cached_training_matches_canonical_and_resumes_after_cache_relocation(
    tmp_path, capsys
):
    manifest, pin = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "cache")
    settings = tmp_path / "config.json"
    write_config(settings, tiny_config())
    common = [
        "train-unified",
        "--manifest",
        str(manifest),
        "--config",
        str(settings),
        "--expected-content-sha256",
        pin,
        "--device",
        "cpu",
    ]
    main(common + ["--output", str(tmp_path / "uncached")])
    capsys.readouterr()
    main(
        common
        + [
            "--output",
            str(tmp_path / "cached"),
            "--stop-after-epoch",
            "1",
            "--derived-cache",
            report["root"],
            "--expected-cache-fingerprint",
            report["cache_fingerprint"],
        ]
    )
    capsys.readouterr()
    moved = tmp_path / "relocated-cache"
    shutil.move(report["root"], moved)
    checkpoint = tmp_path / "cached/last.pt"
    # The checkpoint restores both content pins; the runtime path is explicit.
    main(
        [
            "train-unified",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "cached"),
            "--resume",
            str(checkpoint),
            "--derived-cache",
            str(moved),
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()
    canonical = torch.load(tmp_path / "uncached/last.pt", weights_only=True)
    cached = torch.load(checkpoint, weights_only=True)
    assert cached["epoch"] == 2
    assert cached["derived_cache"]["fingerprint"] == report["cache_fingerprint"]
    assert str(moved) not in json.dumps(cached["config"])
    for key, value in canonical["state_dict"].items():
        assert torch.equal(value, cached["state_dict"][key])


def test_public_cache_requires_external_pins_without_creating_output(tmp_path):
    manifest, pin = corpus(tmp_path / "canonical")
    report = build_cache(manifest, tmp_path / "cache")
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train-unified",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--expected-content-sha256",
                pin,
                "--derived-cache",
                report["root"],
                "--device",
                "cpu",
            ]
        )
    assert exc.value.code == 2 and not output.exists()
