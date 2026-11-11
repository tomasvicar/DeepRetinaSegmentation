"""Explicit opt-in synthetic CUDA step; run only after the owner clears the GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PORT = "903c92903742bec80fae71167967fc5d0466c9a4"
REFERENCE = "72fb321187eca517823083aca80cedace9605570"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def source_pin(source, commit):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    root = source.parent
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], env=env, text=True
    ).strip()
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, head], env=env, check=True
    )
    require(
        not subprocess.check_output(["git", "-C", str(root), "diff", commit, "--", "src"], env=env),
        "Product source differs from pinned commit",
    )
    relative = "src/deepretina/unified.py"
    original = subprocess.check_output(
        ["git", "-C", str(root), "show", commit + ":" + relative], env=env
    )
    require(
        (root / relative).read_bytes() == original, "Training source differs from pinned commit"
    )
    return hashlib.sha256(original).hexdigest()


def finite_fp32(tensors, torch):
    tensors = list(tensors)
    require(bool(tensors), "No tensors inspected")
    for value in tensors:
        require(
            value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
            "Non-finite/non-FP32 master tensor",
        )


def worker(source, output):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import numpy as np
    import torch

    from deepretina import unified as u

    require(Path(u.__file__).resolve() == source / "deepretina/unified.py", "Wrong imported source")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.device_count() == 1, "Expose exactly one physical GPU")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = u.UnifiedConfig(
        mixed_precision="bf16",
        base_channels=8,
        depth=3,
        patch_size=32,
        batch_size=2,
        tile_size=32,
        overlap=8,
    )
    u._validate_training_precision(cfg, device)
    # A successful feature check alone is insufficient: actual kernels below must run.
    model = u.build_unified(cfg).to(device).train()
    image = torch.rand(2, 3, 32, 32, device=device)
    target = torch.zeros(2, 5, 32, 32, device=device)
    target[:, 0, 8:24] = 1
    target[:, 1, 8:16] = 1
    target[:, 2, 16:24] = 1
    target[:, 3, 4:28, 4:28] = 1
    target[:, 4, 12:20, 12:20] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    with u._precision_context(cfg, device, training=True):
        logits = model(image - 0.5)
        loss, components = u.unified_loss(logits.float(), target, valid)
    require(
        logits.dtype == torch.bfloat16 and loss.dtype == torch.float32, "Wrong training arithmetic"
    )
    require(
        set(components) == {"vessel", "disc", "cup", "av"},
        "Not all five output supervision exercised",
    )
    require(bool(torch.isfinite(loss)), "Non-finite loss")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite_fp32(gradients, torch)
    require(any(bool(g.abs().sum() > 0) for g in gradients), "All parameter gradients are zero")
    optimizer.step()
    torch.cuda.synchronize()
    finite_fp32(model.parameters(), torch)
    states = [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    finite_fp32(states, torch)
    require(
        all(int(state["step"].item()) == 1 for state in optimizer.state.values()),
        "AdamW did not step",
    )
    require(
        any(not torch.equal(before[k], p) for k, p in model.named_parameters()),
        "No parameter update",
    )
    forward_dtypes = []
    hook = model.register_forward_hook(lambda _m, _i, y: forward_dtypes.append(str(y.dtype)))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = u.predict_probabilities(
            model, image[0].cpu(), device, cfg, tile_size=32, overlap=8
        )
    hook.remove()
    require(
        forward_dtypes and set(forward_dtypes) == {"torch.float32"},
        "Inference failed to disable outer autocast",
    )
    require(
        prediction.dtype == np.float32 and np.isfinite(prediction).all(), "Invalid FP32 inference"
    )
    payload = {
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "loss": loss.detach().cpu(),
        "prediction": torch.from_numpy(prediction),
        "cuda_rng": torch.cuda.get_rng_state().cpu(),
    }
    # Optimizer state is copied explicitly; this synthetic artifact is never a released model.
    for state in payload["optimizer"]["state"].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cpu()
    torch.save(payload, output)
    props = torch.cuda.get_device_properties(0)
    return {
        "status": "PASS",
        "device_name": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "loss": float(loss.detach()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_logits_dtype": str(logits.dtype),
        "prediction_dtype": str(prediction.dtype),
        "all_parameter_optimizer_gradient_tensors_finite_fp32": True,
    }


def equal_tree(a, b, torch):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal_tree(x, y, torch) for x, y in zip(a, b))
        )
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port-source", type=Path)
    p.add_argument("--reference-source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-gpu-uuid")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Explicit authorization after an independent idle-device check",
    )
    p.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    require(args.execute, "Preparation only: --execute is required; no GPU action performed")
    if args.worker_source:
        report = worker(args.worker_source.resolve(), args.output)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        return
    require(args.execute, "Preparation only: --execute is required; no GPU action performed")
    require(
        args.expected_gpu_uuid and os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu_uuid,
        "Physical GPU UUID visibility must be explicit",
    )
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "Deterministic CUBLAS environment required before import",
    )
    require(not args.output.exists(), "Use fresh output; never overwrite evidence")
    pins = {
        "port_unified_sha256": source_pin(args.port_source.resolve(), PORT),
        "reference_unified_sha256": source_pin(args.reference_source.resolve(), REFERENCE),
    }
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", args.expected_gpu_uuid, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    require(uuid == args.expected_gpu_uuid, "Observed physical GPU identity differs")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    for label, source in (("reference", args.reference_source), ("port", args.port_source)):
        env = dict(os.environ, PYTHONPATH=str(source.resolve()), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--execute",
                "--worker-source",
                str(source.resolve()),
                "--output",
                str(args.output / (label + ".pt")),
            ],
            env=env,
            check=True,
            timeout=35,
        )
    import torch

    reference = torch.load(args.output / "reference.pt", map_location="cpu", weights_only=True)
    port = torch.load(args.output / "port.pt", map_location="cpu", weights_only=True)
    parity = {k: equal_tree(reference[k], port[k], torch) for k in reference}
    report = {
        "status": "PASS" if all(parity.values()) else "FAIL",
        "source_port": PORT,
        "source_reference": REFERENCE,
        "physical_gpu_uuid": uuid,
        "exact_parity": parity,
        "elapsed_seconds": time.monotonic() - started,
        "synthetic_only": True,
        "GT_TEST_or_model_checkpoint_read": False,
        **pins,
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    require(
        all(parity.values()),
        "Source72/port synthetic step or FP32 prediction differs; retain FAIL evidence",
    )


if __name__ == "__main__":
    main()
