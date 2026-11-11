"""CPU1 tiny prepared-input compatibility check, never GT/quality evaluation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from deepretina import unified as u

    assert Path(u.__file__).resolve().is_relative_to(args.source_root.resolve())
    expected = "20d4caab1648ea6216a9373efc8f826f19d8517373881a20c884443548f474d4"
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == expected
    model, payload, device = u.load_unified(args.checkpoint, "cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    cfg = u.inference_config(payload, tile_size=32, overlap=8)
    image = torch.linspace(0, 1, 3 * 35 * 49, dtype=torch.float32).reshape(3, 35, 49)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for mode in args.modes:
        started = time.monotonic()
        kwargs = (
            {}
            if mode == "none"
            and "tta" not in __import__("inspect").signature(u.predict_probabilities).parameters
            else {"tta": mode}
        )
        result = u.predict_probabilities(model, image, device, cfg, 32, 8, **kwargs)
        assert (
            result.shape == (5, 35, 49) and result.dtype == np.float32 and np.isfinite(result).all()
        )
        np.testing.assert_allclose(result[1:3].sum(axis=0), 1, rtol=1e-6, atol=1e-7)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
        assert all(p.grad is None for p in model.parameters())
        np.save(args.output / (mode + ".npy"), result, allow_pickle=False)
        cases.append(
            {
                "mode": mode,
                "seconds": time.monotonic() - started,
                "probability_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "finite": True,
                "model_weights_and_buffers_unchanged": True,
            }
        )
    assert (
        hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == expected
        and not torch.cuda.is_initialized()
    )
    report = {
        "scope": "tiny_prepared_CPU_forward_compatibility_not_quality_or_latency_benchmark",
        "checkpoint_sha256": expected,
        "source_module_sha256": hashlib.sha256(Path(u.__file__).read_bytes()).hexdigest(),
        "prepared_input_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
        "prepared_shape": list(image.shape),
        "overridden_tile_size": 32,
        "overridden_overlap": 8,
        "original_training_mixed_precision": payload["config"]["mixed_precision"],
        "inference_parameter_dtype": "torch.float32",
        "threads": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "CUDA_initialized": False,
        "GT_read": False,
        "TEST_evaluated": False,
        "cases": cases,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
