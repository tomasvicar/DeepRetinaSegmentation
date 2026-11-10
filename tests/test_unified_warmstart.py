"""Original-head fidelity, rectangular geometry, and explicit exposure safeguards."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from deepretina.legacy.single_net import Unet
from deepretina.unified import (
    CHANNELS,
    UnifiedConfig,
    build_unified,
    evaluate_unified_model,
    load_unified,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def teacher():
    checkpoint = (
        Path(__file__).parents[1] / "src/deepretina/pretrained/legacy-single-net-universal.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Unet(**payload["architecture"]).eval()
    model.load_state_dict(payload["state_dict"])
    return checkpoint, model


def config(checkpoint):
    return UnifiedConfig(
        architecture="legacy_universal",
        legacy_initialization=str(checkpoint),
        allow_historical_exposure=True,
    )


def test_warmstart_needs_explicit_ancestry_permission(teacher):
    checkpoint, _ = teacher
    with pytest.raises(ValueError, match="allow_historical_exposure"):
        build_unified(replace(config(checkpoint), allow_historical_exposure=False))
    with pytest.raises(ValueError, match="requires legacy_universal"):
        build_unified(replace(config(checkpoint), architecture="compact_unet"))


@pytest.mark.parametrize("shape", [(32, 48), (288, 288)])
def test_actual_original_three_logits_are_bitwise_identical_and_av_neutral(teacher, shape):
    checkpoint, original = teacher
    student = build_unified(config(checkpoint)).eval()
    torch.manual_seed(17)
    inputs = torch.rand(1, 3, *shape) - 0.5
    with torch.inference_mode():
        expected, actual = original(inputs), student(inputs)
    assert torch.equal(actual[:, [0, 3, 4]], expected)
    assert torch.equal(student.final.weight[[0, 3, 4]], original.final.weight)
    assert torch.equal(student.final.bias[[0, 3, 4]], original.final.bias)
    assert torch.count_nonzero(actual[:, 1:3]) == 0
    assert torch.equal(actual[:, 1:3].softmax(1), torch.full_like(actual[:, 1:3], 0.5))
    assert sum(p.numel() for p in student.parameters()) == 1304421
    assert student.lineage["independent_model_eligible"] is False
    assert len(student.lineage["checkpoint_sha256"]) == 64
    assert student.lineage["original_provenance"]["source_sha256"]


@pytest.mark.parametrize("shape", [(33, 49), (32, 47), (1, 3), (17, 16)])
def test_student_corrects_rectangular_odd_geometry_without_changing_reference(teacher, shape):
    checkpoint, original = teacher
    student = build_unified(config(checkpoint)).eval()
    h, w = shape
    inputs = torch.rand(1, 3, h, w) - 0.5
    ph, pw = (-h) % 16, (-w) % 16
    mode = "reflect" if ph < h and pw < w else "replicate"
    padded = F.pad(inputs, (0, pw, 0, ph), mode=mode) if ph or pw else inputs
    with torch.inference_mode():
        expected = original(padded)[..., :h, :w]
        actual = student(inputs)
    assert actual.shape == (1, 5, h, w)
    # Expanding the final convolution from3 to5 output channels can select a
    # different CPU kernel for tiny images; copied coefficients are exact, while
    # final floating-point summation can differ by a few ulps.
    torch.testing.assert_close(actual[:, [0, 3, 4]], expected, rtol=2e-5, atol=2e-4)


def test_cold_checkpoint_load_never_reopens_teacher_and_restores_ancestry(tmp_path, teacher):
    checkpoint, _ = teacher
    cfg = config(checkpoint)
    model = build_unified(cfg).eval()
    # The recorded path intentionally does not exist on this recipient machine.
    saved_config = asdict(replace(cfg, legacy_initialization=str(tmp_path / "absent-teacher.pt")))
    torch.save(
        {
            "format_version": 1,
            "family": "unified",
            "channels": list(CHANNELS),
            "normalization": "RGB/255-0.5",
            "config": saved_config,
            "state_dict": model.state_dict(),
            "lineage": model.lineage,
        },
        tmp_path / "student.pt",
    )
    loaded, payload, _ = load_unified(tmp_path / "student.pt", "cpu")
    assert loaded.lineage == model.lineage == payload["lineage"]
    assert all(
        torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="exploratory train/val"):
        evaluate_unified_model(loaded, tmp_path / "does-not-exist.json", split="test")


def test_scratch_original_architecture_has_no_teacher_ancestry_and_all_heads_train():
    model = build_unified(UnifiedConfig(architecture="legacy_universal"))
    assert model.lineage["initialization"] == "scratch"
    assert model.lineage["historical_exposure"] is False
    assert model.lineage["independent_model_eligible"] is True
    loss = model(torch.rand(2, 3, 33, 49) - 0.5).square().mean()
    loss.backward()
    assert all(model.final.weight.grad[channel].abs().sum() > 0 for channel in range(5))


def test_wrong_safe_checkpoint_family_is_rejected(tmp_path):
    torch.save({"family": "unified"}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="safe original universal"):
        build_unified(config(tmp_path / "wrong.pt"))
