"""Explicit future CUDA TRAIN core-step diagnostic; never VAL/TEST or a training run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import socket
import subprocess
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from deepretina import unified as u
from deepretina.derived_cache import VerifiedCache

PARENT = "10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
POLICY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
GPU = "GPU-d0675605-e6c2-1c85-9799-08b279aab0ff"
CPU_LIMIT = 2 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def state_hash(value):
    """Deterministic exact tensor/metadata hash, independent of pickle storage IDs."""
    h = hashlib.sha256()

    def visit(item):
        h.update(type(item).__name__.encode() + b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), item.shape)).encode() + item.tobytes())
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def capture_rng(device):
    return copy.deepcopy(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
    )


def reset(model, optimizer, initial, device):
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    model.load_state_dict(initial["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    rng = initial["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], device)
    model.train()


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return True


def forward_backward(model, optimizer, batch, config, *, update):
    image, target, valid = batch
    optimizer.zero_grad(set_to_none=True)
    with u._precision_context(config, image.device, training=True):
        logits = model(image - 0.5)
        loss, components = u.unified_loss(
            logits.float() if config.mixed_precision == "bf16" else logits,
            target,
            valid,
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            av_weight=config.av_weight,
            vessel_loss_weight=config.vessel_loss_weight,
            vessel_pos_weight=config.vessel_pos_weight,
        )
    require(
        bool(components) and bool(torch.isfinite(loss)), "Missing supervision or nonfinite loss"
    )
    loss.backward()
    if update:
        optimizer.step()
    return loss, components


def diagnostic(model, optimizer, batch, config):
    loss, components = forward_backward(model, optimizer, batch, config, update=False)
    gradients = [
        p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None
    ]
    vector = torch.cat(gradients)
    require(
        bool(torch.isfinite(vector).all()) and bool(torch.count_nonzero(vector)),
        "Invalid/zero gradients",
    )
    require(
        finite_state(model.state_dict()) and finite_state(optimizer.state_dict()), "Nonfinite state"
    )
    return {
        "loss": float(loss.detach()),
        "components": {k: float(v) for k, v in components.items()},
        "gradient_norm": float(torch.linalg.vector_norm(vector)),
        "gradient_nonzero": int(torch.count_nonzero(vector)),
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "gradient_dtypes": sorted(
            {str(p.grad.dtype) for p in model.parameters() if p.grad is not None}
        ),
    }, vector


def admission(args):
    require(args.execute, "Dry by default: --execute requires prior exclusive-GPU review by root")
    require(socket.gethostname().split(".")[0] == "ada", "Own Ada only")
    require(
        32 <= args.batches <= 64 and 5 <= args.warmup <= 10, "Require32–64 batches and5–10 warmup"
    )
    require(0 < args.max_seconds <= 1200, "Bound must be <=1200seconds")
    require(
        args.output.is_absolute() and not args.output.exists(), "New absolute report path required"
    )
    device = torch.device("cuda:0")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "One visible CUDA device required",
    )
    u._validate_training_precision(u.UnifiedConfig(mixed_precision="bf16"), device)
    actual = str(torch.cuda.get_device_properties(device).uuid)
    require(
        actual.lower().removeprefix("gpu-") == args.expected_gpu_uuid.lower().removeprefix("gpu-"),
        "GPU UUID differs",
    )
    return device, actual


def preload(dataset, config, count, end):
    generator = torch.Generator().manual_seed(config.seed)
    sampler = iter(
        u.CoverageSampler(
            dataset.sampling_weights(), config.epoch_samples, generator, config.coverage_first
        )
    )
    batches, evidence, used = [], [], 0
    for _ in range(count):
        require(time.monotonic() < end, "Benchmark deadline during TRAIN preload")
        samples = [dataset[next(sampler)] for _ in range(config.batch_size)]
        batch = tuple(torch.stack([sample[c] for sample in samples]) for c in range(3))
        used += sum(t.numel() * t.element_size() for t in batch)
        require(used <= CPU_LIMIT, "Preloaded CPU tensors exceed2GiB")
        batches.append(batch)
        evidence.append(
            {
                "photo_ids": [dataset.photo_ids[int(sample[3])] for sample in samples],
                "annotation_ids": [json.loads(sample[4]) for sample in samples],
                "tensor_sha256": [state_hash(t) for t in batch],
            }
        )
    return batches, evidence, used


def run_leg(model, optimizer, initial, batches, config, device, warmup, end):
    require(device.type == "cuda", "No CPU or simulated GPU timing")
    reset(model, optimizer, initial, device)
    for i in range(warmup):
        require(time.monotonic() < end, "Benchmark deadline during warmup")
        forward_backward(
            model, optimizer, tuple(t.to(device) for t in batches[i]), config, update=True
        )
    reset(model, optimizer, initial, device)
    start_hash = state_hash(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(device),
        }
    )
    first, vector = diagnostic(model, optimizer, tuple(t.to(device) for t in batches[0]), config)
    reset(model, optimizer, initial, device)  # Undo diagnostic BN/RNG and gradients too.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    timings, copies, losses = [], [], []
    for batch in batches:
        require(time.monotonic() < end, "Benchmark deadline during measured steps")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        gpu_batch = tuple(t.to(device) for t in batch)
        torch.cuda.synchronize(device)
        copies.append(time.perf_counter() - started)
        started = time.perf_counter()
        loss, _ = forward_backward(model, optimizer, gpu_batch, config, update=True)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
        del gpu_batch, loss
    peak_allocated, peak_reserved = (
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )
    last, _ = diagnostic(model, optimizer, tuple(t.to(device) for t in batches[-1]), config)
    return {
        "precision": config.mixed_precision,
        "initial_state_sha256": start_hash,
        "steps": len(batches),
        "batch_size": config.batch_size,
        "seconds": timings,
        "h2d_seconds": copies,
        "losses": losses,
        "median_step_seconds": float(np.median(timings)),
        "p10_p90_seconds": np.percentile(timings, [10, 90]).tolist(),
        "samples_per_compute_second": len(batches) * config.batch_size / sum(timings),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_extra_allocated_bytes": peak_allocated - baseline_allocated,
        "peak_reserved_bytes": peak_reserved,
        "first_diagnostic": first,
        "last_diagnostic": last,
    }, vector


def benchmark(args):
    device, gpu = admission(args)  # No report or inputs read on unsupported admission.
    torch.set_num_threads(20)
    torch.set_num_interop_threads(20)
    started, end = time.monotonic(), time.monotonic() + args.max_seconds
    root = Path(__file__).resolve().parents[1]
    require(
        Path(u.__file__).resolve() == root / "src/deepretina/unified.py",
        "Imported training package differs from actual source",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        not subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
        ),
        "Clean Git source required",
    )
    require(
        not any(
            args.output.resolve().is_relative_to(p.resolve())
            for p in [root, args.manifest.parent, args.cache]
        ),
        "Report outside source/data/cache required",
    )
    raw = args.parent.read_bytes()
    require(digest(raw) == PARENT, "Exact immutable terminal60 parent bytes required")
    parent = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(
        parent["family"] == "unified" and parent["epoch"] == 60 and parent["steps"] == 122880,
        "Terminal60 parent metadata differs",
    )
    require(
        parent["manifest_sha256"] == MANIFEST
        and parent["prepared_content_sha256"] == CONTENT
        and parent["geometry_policy"]["sha256"] == POLICY,
        "Parent canonical/policy differs",
    )
    require(digest(args.manifest.read_bytes()) == MANIFEST, "Manifest differs")
    config = u.UnifiedConfig(
        **tomllib.loads((root / "configs/unified-v2-vascular2.toml").read_text())
    )
    config = replace(config, device="cuda:0", workers=0, mixed_precision="none")
    cache = VerifiedCache(
        args.cache,
        args.manifest,
        expected_cache_fingerprint=CACHE,
        expected_canonical_fingerprint=CONTENT,
    )
    rows = u.read_unified_manifest(args.manifest, "train")
    require(
        len(rows) == 5700 and len({r["dataset"] for r in rows}) == 21, "TRAIN inventory differs"
    )
    cache.require_coverage(rows)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = u.UnifiedDataset(
        args.manifest, "train", config, return_trace=True, derived_cache=cache
    )
    require(len(dataset) == 2529, "TRAIN photo inventory differs")
    batches, samples, cpu_bytes = preload(dataset, config, args.batches, end)
    del dataset, cache
    model = u.build_unified(config, initialize=False).to(device)
    model.load_state_dict(parent["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    initial = {
        "model": copy.deepcopy(parent["state_dict"]),
        "optimizer": copy.deepcopy(parent["training_state"]["optimizer"]),
        "rng": capture_rng(device),
    }
    require(finite_state(initial), "Nonfinite initial model/Adam state")
    tensors = list(initial["model"].values()) + [
        v
        for state in initial["optimizer"]["state"].values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    ]
    require(
        all(not t.is_floating_point() or t.dtype == torch.float32 for t in tensors),
        "Parent model/Adam must be float32",
    )
    initial_hash = state_hash(initial)
    executable = hashlib.sha256()
    for path in sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    ):
        executable.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    legs, vectors = [], []
    for precision in ("none", "bf16", "bf16", "none"):
        leg, vector = run_leg(
            model,
            optimizer,
            initial,
            batches,
            replace(config, mixed_precision=precision),
            device,
            args.warmup,
            end,
        )
        require(leg["initial_state_sha256"] == initial_hash, "Leg initial state differs")
        legs.append(leg)
        vectors.append(vector)
    reference, difference = vectors[0].double(), vectors[1].double() - vectors[0].double()
    report = {
        "schema_version": 1,
        "kind": "train_core_step_precision_benchmark",
        "completed": True,
        "source_commit": commit,
        "execution_code_sha256": executable.hexdigest(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "training_source_sha256": u._training_source_sha256(),
        "parent_sha256": PARENT,
        "parent_epoch": parent["epoch"],
        "parent_training_source_sha256": parent["training_source_sha256"],
        "manifest_sha256": MANIFEST,
        "canonical_fingerprint": CONTENT,
        "geometry_policy_sha256": POLICY,
        "cache_fingerprint": CACHE,
        "config": asdict(config),
        "initial_state_sha256": initial_hash,
        "parent_optimizer_param_groups": initial["optimizer"]["param_groups"],
        "cpu_batch_tensor_bytes": cpu_bytes,
        "batches": samples,
        "legs": legs,
        "repeated_precision_loss_max_absolute_difference": {
            "none": max(abs(a - b) for a, b in zip(legs[0]["losses"], legs[3]["losses"])),
            "bf16": max(abs(a - b) for a, b in zip(legs[1]["losses"], legs[2]["losses"])),
        },
        "first_gradient_difference": {
            "relative_l2": float(
                torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
            ),
            "max_absolute": float(difference.abs().max()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(reference, vectors[1].double(), dim=0)
            ),
        },
        "runtime": {
            "gpu_uuid": gpu,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "elapsed_seconds": time.monotonic() - started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "math_thread_environment": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "scope": "TRAIN core steps only; excludes preload, DataLoader, VAL and evidence/coverage accounting; no full-epoch throughput or quality claim",
        "test_evaluation_performed": False,
        "validation_evaluation_performed": False,
        "model_saved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-gpu-uuid", default=GPU)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-seconds", type=int, default=900)
    return p


if __name__ == "__main__":
    benchmark(parser().parse_args())
