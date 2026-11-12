"""Toy CPU checkpoint export/load/forward parity; no real models, datasets or GPU."""

import hashlib
import io
import json
from dataclasses import asdict

import pytest
import torch

from deepretina.inference_export import export_unified_inference
from deepretina.unified import CHANNELS, UnifiedConfig, build_unified, load_unified


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("CPU export must not query or initialize CUDA")

    for name in ("init", "_lazy_init", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, blocked)
    torch.set_num_threads(1)
    config = UnifiedConfig(
        base_channels=2,
        depth=1,
        norm="batch",
        epochs=3,
        epoch_samples=2,
        batch_size=1,
        patch_size=32,
        device="cuda:0",
        mixed_precision="bf16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = build_unified(config)
    payload = {
        "format_version": 1,
        "family": "unified",
        "channels": list(CHANNELS),
        "config": asdict(config),
        "normalization": "RGB/255-0.5",
        "preprocessing": {"profile": "legacy25_raw", "pixels_per_degree": 25, "normalized": False},
        "architecture": model.architecture_spec,
        "lineage": model.lineage,
        "state_dict": model.state_dict(),
        "supervised_channels": [True] * 5,
        "epoch": 3,
        "steps": 6,
        "training_source_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "prepared_content_sha256": "3" * 64,
        "coverage": {"seen_photos": 2},
        "validation_metrics": {"mean_dice": 0.2},
        "training_state": {
            "optimizer": {
                "state": {
                    0: {
                        "exp_avg": torch.zeros(2),
                        "exp_avg_sq": torch.zeros(2),
                        "step": torch.tensor(6),
                    }
                }
            },
            "rng": {"torch": torch.get_rng_state()},
            "scheduler": {"T_max": 3},
        },
    }
    path = tmp_path / "original.pt"
    torch.save(payload, path)
    return path, payload


def test_actual_toy_export_safe_CPU_load_model_parity_and_provenance(
    checkpoint, tmp_path, monkeypatch
):
    source, payload = checkpoint
    original_bytes = source.read_bytes()
    output = tmp_path / "new" / "inference.pt"
    calls = []
    actual_load = torch.load

    def safe_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    rng = torch.get_rng_state().clone()
    with torch.device("meta"):
        proof = export_unified_inference(source, output, expected_original_sha256=digest(source))
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(calls) >= 3
    # No map_location is needed: a CUDA-tagged saved storage would hit blocked init.
    saved = actual_load(output, weights_only=True)
    assert saved["inference_only"] is True and saved["config"] == payload["config"]
    assert not any(key in saved for key in ["training_state", "optimizer", "rng", "scheduler"])
    assert (
        saved["preprocessing"] == payload["preprocessing"]
        and saved["architecture"] == payload["architecture"]
    )
    assert saved["training_source_sha256"] == payload["training_source_sha256"]
    assert all(t.device.type == "cpu" for t in saved["state_dict"].values())
    for name, tensor in payload["state_dict"].items():
        actual = saved["state_dict"][name]
        assert (
            actual.dtype == tensor.dtype
            and actual.shape == tensor.shape
            and torch.equal(actual, tensor)
        )
    with torch.random.fork_rng(devices=[]):
        left, _, _ = load_unified(io.BytesIO(original_bytes), device="cpu")
        right, _, _ = load_unified(output, device="cpu")
    image = torch.linspace(-0.5, 0.5, 3 * 9 * 11).reshape(1, 3, 9, 11)
    with torch.inference_mode():
        assert torch.equal(left(image), right(image))
    assert source.read_bytes() == original_bytes
    assert proof["artifact_sha256"] == digest(output) != digest(source)
    assert proof["original_checkpoint_sha256"] == digest(source)
    assert proof["model_state_sha256"] == saved["export_provenance"]["model_state_sha256"]
    assert proof["removed_training_fields"] == ["training_state"]
    assert json.loads(output.with_name(output.name + ".export.json").read_text()) == proof
    assert proof["NN_forward_or_TEST_performed"] is False  # exporter itself has no forward


@pytest.mark.parametrize("which", ["output", "sidecar"])
def test_existing_outputs_preserved_before_any_source_read(tmp_path, which):
    output = tmp_path / "model.pt"
    existing = output if which == "output" else output.with_name(output.name + ".export.json")
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        export_unified_inference(tmp_path / "missing.pt", output, expected_original_sha256="1" * 64)
    assert existing.read_bytes() == b"preserve"


@pytest.mark.parametrize("pin", ["", "bad", "F" * 64, "0" * 64])
def test_bad_or_mismatched_original_pin_fails_before_output(checkpoint, tmp_path, pin):
    source, _ = checkpoint
    out = tmp_path / "inference.pt"
    with pytest.raises(ValueError):
        export_unified_inference(source, out, expected_original_sha256=pin)
    assert not out.exists()


@pytest.mark.parametrize(
    "bad", ["empty", "non_tensor", "nonfinite", "shape", "family", "hidden_tensor", "nonraw_config"]
)
def test_invalid_payload_does_not_publish(checkpoint, tmp_path, bad):
    source, payload = checkpoint
    if bad == "empty":
        payload["state_dict"] = {}
    if bad == "non_tensor":
        payload["state_dict"]["extra"] = 1
    if bad == "nonfinite":
        payload["state_dict"][next(iter(payload["state_dict"]))].flatten()[0] = float("nan")
    if bad == "shape":
        payload["state_dict"][next(iter(payload["state_dict"]))] = torch.ones(1)
    if bad == "family":
        payload["family"] = "legacy_single_net"
    if bad == "hidden_tensor":
        payload["metadata"] = torch.ones(1)
    if bad == "nonraw_config":
        payload["config"]["photometric_profile"] = "unsupported"
    torch.save(payload, source)
    out = tmp_path / "inference.pt"
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        export_unified_inference(source, out, expected_original_sha256=digest(source))
    assert not out.exists()


def test_reexport_refused_and_all_training_aliases_removed(checkpoint, tmp_path):
    source, payload = checkpoint
    for key in (
        "optimizer",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scheduler",
        "rng",
        "rng_state",
    ):
        payload[key] = {"state": torch.ones(1)}
    torch.save(payload, source)
    out = tmp_path / "model.pt"
    proof = export_unified_inference(source, out, expected_original_sha256=digest(source))
    saved = torch.load(out, weights_only=True)
    assert all(k not in saved for k in proof["removed_training_fields"])
    with pytest.raises(ValueError, match="previous export"):
        export_unified_inference(out, tmp_path / "another.pt", expected_original_sha256=digest(out))
