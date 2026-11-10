"""Original A/V topology, safe conversion, decoding and reference-numeric checks."""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from deepretina.legacy.av import (
    REFERENCE_SOURCE_SHA256,
    LegacyAVNet,
    center_crop,
    checkpoint_payload,
    convert_av,
    load_av,
    predict_av_input,
    prepare_input,
    restore_classes,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def sample_input(height=63, width=77):
    generator = torch.Generator().manual_seed(20261110)
    inputs = torch.randn(1, 4, height, width, generator=generator)
    inputs[:, :3] = inputs[:, :3] * 35 + 128
    inputs[:, 3] = (inputs[:, 3] > 0.8).float()
    return inputs


def test_state_dictionary_roundtrip_preserves_logits(tmp_path):
    torch.manual_seed(42)
    model = LegacyAVNet().eval()
    path = tmp_path / "safe_av.pt"
    torch.save(checkpoint_payload(model), path)
    loaded, payload = load_av(path)
    with torch.inference_mode():
        expected = model(sample_input())
        actual = loaded(sample_input())
    assert torch.equal(actual, expected)
    assert payload["family"] == "legacy_av"
    assert sum(p.numel() for p in model.parameters()) == 2_030_379


def test_stacking_preserves_stored_dicom_values_and_channel_order():
    rgb = np.arange(16 * 17 * 3, dtype=np.uint16).reshape(16, 17, 3) * 4
    vessel = np.zeros((16, 17), np.uint16)
    vessel[2:6] = 1
    result = prepare_input(rgb, vessel)
    assert result.dtype == torch.float32
    for channel in range(3):
        np.testing.assert_array_equal(result[0, channel].numpy(), rgb[..., channel])
    np.testing.assert_array_equal(result[0, 3].numpy(), vessel)


def test_odd_center_crop_uses_round_half_to_even():
    tensor = torch.arange(9 * 11).reshape(1, 9, 11)
    assert torch.equal(center_crop(tensor, (6, 8)), tensor[:, 2:8, 2:10])
    padded = center_crop(torch.ones(1, 2, 3), (5, 6))
    assert padded.shape == (1, 5, 6)
    assert padded[:, 1:3, 1:4].all()
    assert padded.sum() == 6


def test_nearest_class_restoration_matches_original_opencv():
    cv2 = pytest.importorskip("cv2")
    source = np.arange(21, dtype=np.uint8).reshape(3, 7) % 3
    for shape in ((5, 11), (2, 4), (6, 14)):
        expected = cv2.resize(source, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(restore_classes(source, shape), expected)


def test_decoding_ignores_background_logit_preserves_ties_and_vessel_gate():
    class FixedModel(nn.Module):
        def forward(self, inputs):
            logits = inputs.new_zeros(1, 3, *inputs.shape[-2:])
            logits[:, 0] = 1000  # The original loss never uses this logit.
            logits[:, 1, 0, :4] = inputs.new_tensor([3, 2, 1, 3])
            logits[:, 2, 0, :4] = inputs.new_tensor([2, 2, 3, 1])
            return logits

    inputs = sample_input(16, 17)
    inputs[:, 3] = 0
    inputs[:, 3, 0, :3] = 1
    result = predict_av_input(FixedModel(), inputs)
    np.testing.assert_array_equal(result["classes"][0, :4], [1, 0, 2, 0])
    np.testing.assert_allclose(result["prepared_conditional_probabilities"].sum(axis=0), 1)
    assert result["prepared_conditional_probabilities"][0, 0, 0] > 0.5


def test_constant_channel_is_reported_without_altering_original_math():
    model = LegacyAVNet().eval()
    inputs = sample_input(16, 17)
    inputs[:, 3] = 0
    with pytest.raises(ValueError, match="constant channel"):
        predict_av_input(model, inputs)
    with torch.inference_mode():
        assert not torch.isfinite(model(inputs)).all()


def test_affine_rgb_rescaling_is_invariant_without_quantization():
    model = LegacyAVNet().eval()
    inputs = sample_input()
    scaled = inputs.clone()
    scaled[:, :3] *= 16  # Exact power-of-two rescale, not new quantization.
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), model(scaled), atol=0, rtol=0)


def _forbidden_pickle_call(path):
    Path(path).write_text("should never execute")


class _ForbiddenPayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _forbidden_pickle_call, (self.marker,)


def test_conversion_never_executes_unlisted_pickle_globals(tmp_path):
    source = tmp_path / "unlisted.pt"
    marker = tmp_path / "executed"
    torch.save(_ForbiddenPayload(marker), source)
    with pytest.raises(pickle.UnpicklingError):
        convert_av(source, tmp_path / "converted.pt", require_known=False)
    assert not marker.exists()


@pytest.fixture
def actual_v9(tmp_path):
    source = Path(
        os.environ.get(
            "DEEPRETINA_AV_CHECKPOINT",
            "/var/home/tom/ownCloud/fundus_databases/models/Ves_Class_AV_UNet/net_v0_0_9.pt",
        )
    )
    if not source.is_file():
        pytest.skip("Original owned v9 checkpoint not supplied; set DEEPRETINA_AV_CHECKPOINT")
    path = tmp_path / "v9_safe.pt"
    payload = convert_av(source, path)
    model, _ = load_av(path)
    return source, model, payload


def test_actual_v9_safe_conversion_and_geometry(actual_v9):
    source, model, payload = actual_v9
    assert payload["provenance"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["provenance"]["source_version"] == "net_v0_0_9"
    assert payload["architecture"]["encoder_channels"] == [4, 32, 64, 128, 256]
    assert sum(p.numel() for p in model.parameters()) == 2_030_379
    result = predict_av_input(model, sample_input(), original_size=(126, 154))
    assert result["prepared_logits"].shape == (3, 63, 77)
    assert result["classes"].shape == (126, 154)


class _ReferenceCenterCrop:
    """Independent test adapter verified against actual torchvision0.21 CenterCrop."""

    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, tensor):
        target_h, target_w = self.size
        difference_h = max(0, target_h - tensor.shape[-2])
        difference_w = max(0, target_w - tensor.shape[-1])
        tensor = F.pad(
            tensor,
            (
                difference_w // 2,
                (difference_w + 1) // 2,
                difference_h // 2,
                (difference_h + 1) // 2,
            ),
        )
        top = int(np.rint((tensor.shape[-2] - target_h) / 2))
        left = int(np.rint((tensor.shape[-1] - target_w) / 2))
        return tensor.narrow(-2, top, target_h).narrow(-1, left, target_w)


def reference_classes():
    source = Path(__file__).parent / "reference/av/Network.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == REFERENCE_SOURCE_SHA256
    tree = ast.parse(source.read_text())
    names = {"Net", "Encoder", "Decoder", "Block", "BottleNeck"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in classes} == names
    namespace = {
        "__name__": "reference_av",
        "torch": torch,
        "nn": nn,
        "F": F,
        "torchvision": SimpleNamespace(transforms=SimpleNamespace(CenterCrop=_ReferenceCenterCrop)),
    }
    exec(  # noqa: S102 - pinned original Git classes, isolated reference namespace
        compile(ast.Module(body=classes, type_ignores=[]), "original/Ves_Class/Network.py", "exec"),
        namespace,
    )
    return namespace


def test_actual_v9_forward_equals_original_git_source(actual_v9):
    source, model, _ = actual_v9
    namespace = reference_classes()
    aliases = [
        (namespace[name], "Network." + name)
        for name in ("Net", "Encoder", "Decoder", "Block", "BottleNeck")
    ]
    layers = [
        nn.ReLU,
        nn.BatchNorm2d,
        nn.ModuleList,
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.MaxPool2d,
        set,
    ]
    with torch.serialization.safe_globals([*aliases, *layers]):
        reference = torch.load(source, map_location="cpu", weights_only=True).eval()
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name])
    with torch.inference_mode():
        for shape in ((31, 35), (63, 77), (64, 80), (129, 173)):
            inputs = sample_input(*shape)
            expected, actual = reference(inputs), model(inputs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for source_shape, crop_shape in [((9, 11), (6, 8)), ((2, 3), (5, 6)), ((31, 35), (24, 32))]:
        tensor = torch.arange(source_shape[0] * source_shape[1]).reshape(1, *source_shape)
        assert torch.equal(
            center_crop(tensor, crop_shape), _ReferenceCenterCrop(crop_shape)(tensor)
        )
