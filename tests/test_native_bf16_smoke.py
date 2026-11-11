"""CPU-only checks of the explicit GPU smoke harness; no CUDA calls."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def harness():
    path = Path(__file__).parents[1] / "scripts/smoke_native_bf16.py"
    spec = importlib.util.spec_from_file_location("native_bf16_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_cannot_dispatch_gpu_worker(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["smoke", "--worker-source", "/unused", "--output", str(tmp_path / "new.pt")]
    )
    monkeypatch.setattr(harness, "worker", lambda *a: pytest.fail("GPU worker dispatched"))
    with pytest.raises(RuntimeError, match="--execute is required"):
        harness.main()
    assert list(tmp_path.iterdir()) == []


def test_state_comparison_detects_optimizer_or_dtype_drift(harness):
    state = {
        "adam": {"step": torch.tensor(1.0), "exp_avg": torch.ones(2)},
        "weights": [torch.ones(2)],
    }
    assert harness.equal_tree(state, state, torch)
    changed = {
        "adam": {"step": torch.tensor(2.0), "exp_avg": torch.ones(2)},
        "weights": [torch.ones(2)],
    }
    assert not harness.equal_tree(state, changed, torch)
    assert not harness.equal_tree(torch.ones(2), torch.ones(2, dtype=torch.bfloat16), torch)


def test_fp32_master_check_rejects_nan_bf16_and_empty(harness):
    harness.finite_fp32([torch.ones(2)], torch)
    for tensors in ([], [torch.tensor(float("nan"))], [torch.ones(2, dtype=torch.bfloat16)]):
        with pytest.raises(RuntimeError):
            harness.finite_fp32(tensors, torch)
