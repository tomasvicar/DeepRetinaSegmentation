"""Small CPU math/admission tests; no CUDA timings or real data read."""

import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from deepretina.unified import UnifiedConfig

PATH = Path(__file__).parents[1] / "scripts/benchmark_unified_bf16.py"
spec = importlib.util.spec_from_file_location("precision_benchmark", PATH)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("execute", [False, True])
def test_no_execute_or_unsupported_device_never_reads_inputs_or_writes(
    tmp_path, monkeypatch, execute
):
    monkeypatch.setattr(b.socket, "gethostname", lambda: "ada")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = b.parser().parse_args(
        [
            "--parent",
            str(tmp_path / "missing.pt"),
            "--manifest",
            str(tmp_path / "missing.json"),
            "--cache",
            str(tmp_path / "missing-cache"),
            "--output",
            str(tmp_path / "out.json"),
        ]
        + (["--execute"] if execute else [])
    )
    with pytest.raises(ValueError, match="--execute|One visible CUDA"):
        b.benchmark(args)
    assert not (tmp_path / "out.json").exists()


def test_reset_restores_exact_weights_bn_adam_and_python_numpy_torch_rng():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, padding=1), torch.nn.BatchNorm2d(5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = (
        torch.rand(2, 3, 8, 8),
        torch.zeros(2, 5, 8, 8),
        torch.ones(2, 5, 8, 8, dtype=torch.bool),
    )
    batch[1][:, 1, :, ::2] = 1
    batch[1][:, 2, :, 1::2] = 1
    cfg = UnifiedConfig()
    b.forward_backward(model, optimizer, batch, cfg, update=True)
    initial = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": b.capture_rng(device),
    }
    expected = b.state_hash(initial)
    results = []
    for _ in range(2):
        b.reset(model, optimizer, initial, device)
        assert (
            b.state_hash(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": b.capture_rng(device),
                }
            )
            == expected
        )
        diagnostic, vector = b.diagnostic(model, optimizer, batch, cfg)
        results.append((diagnostic, vector, random.random(), np.random.rand(), torch.rand(3)))
        b.forward_backward(model, optimizer, batch, cfg, update=True)
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2:4] == results[1][2:4]
    assert torch.equal(results[0][4], results[1][4])
    assert b.state_hash(initial) == expected  # reset never mutates the CPU parent snapshot.


def test_hash_binds_dtype_shape_order_and_values():
    assert b.state_hash(torch.zeros(2)) != b.state_hash(torch.zeros(1, 2))
    assert b.state_hash(torch.zeros(2)) != b.state_hash(torch.zeros(2, dtype=torch.bfloat16))
    assert b.state_hash([1, 2]) != b.state_hash([2, 1])
    assert b.state_hash({"b": 2, "a": 1}) == b.state_hash({"a": 1, "b": 2})


def test_cpu_timing_is_explicitly_forbidden():
    with pytest.raises(ValueError, match="No CPU or simulated GPU timing"):
        b.run_leg(None, None, None, None, None, torch.device("cpu"), 5, 0)


def test_train_preload_traces_tensor_hash_and_cpu_memory_cap(monkeypatch):
    class TrainOnly:
        photo_ids = ("train-a", "train-b")

        def sampling_weights(self):
            return [1, 1]

        def __getitem__(self, index):
            return (
                torch.full((3, 2, 2), float(index)),
                torch.zeros(5, 2, 2),
                torch.ones(5, 2, 2, dtype=torch.bool),
                index,
                json.dumps(["train-record-" + str(index)]),
            )

    cfg = UnifiedConfig(epoch_samples=4, batch_size=2)
    batches, evidence, size = b.preload(TrainOnly(), cfg, 2, time.monotonic() + 10)
    assert len(batches) == len(evidence) == 2 and size > 0
    assert all(p.startswith("train-") for e in evidence for p in e["photo_ids"])
    assert evidence[0]["tensor_sha256"] == [b.state_hash(t) for t in batches[0]]
    monkeypatch.setattr(b, "CPU_LIMIT", 1)
    with pytest.raises(ValueError, match="exceed2GiB"):
        b.preload(TrainOnly(), cfg, 2, time.monotonic() + 10)


def test_nonfinite_loss_or_zero_gradient_cannot_be_diagnostic():
    model = torch.nn.Conv2d(3, 5, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    batch = (
        torch.full((1, 3, 2, 2), float("nan")),
        torch.zeros(1, 5, 2, 2),
        torch.ones(1, 5, 2, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="nonfinite loss"):
        b.diagnostic(model, optimizer, batch, UnifiedConfig())
