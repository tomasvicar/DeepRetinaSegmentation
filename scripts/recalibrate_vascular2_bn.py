"""EXPERIMENT: explicit vascular2 BEST E59 TRAIN-only BN recalibration, CUDA FP32.

Distinct0b/81113 parent; original6b CPU-only helper is unchanged. No TEST/resume.
"""

import argparse
import copy
import hashlib
import inspect
import io
import json
import os
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import update_bn

SOURCE = "0b8762e667c997a2dfd129dc65988c1f26f6b2ad"
EXECUTION = "3ffe949e787045546a9470750b1aedc27c8ae9bee3603db568f719e4b407f24e"
TRAINING = "85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
PARENT_SHA = "81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b"
UPDATER_SHA = "4d67111e1e01591e8b3475e0e6353eb92e349907a6f02d06b827bcec3998c6ee"
RECIPE = {
    "samples": 8192,
    "batch_size": 4,
    "seed": 42,
    "coverage_first": True,
    "photos": 2529,
    "sampling": "dataset_task_balanced",
    "patch_size": 384,
    "foreground_probability": 0.5,
    "augmentation": "basic",
    "workers": 2,
    "normalization": "RGB/255-0.5",
    "gradient_updates": 0,
    "vascular_sampling_weight": 2.0,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_bytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def bn_keys(model):
    return {
        f"{name}.{key}" if name else key
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for key in ("running_mean", "running_var", "num_batches_tracked")
        if getattr(module, key) is not None
    }


def recalibrate(model, batches):
    """Use actual PyTorch update_bn; prove every non-BN state byte unchanged."""
    if sha(inspect.getsource(update_bn).encode()) != UPDATER_SHA:
        raise ValueError("Pinned update_bn implementation differs")
    if str(torch.__version__) != "2.6.0+cu124":
        raise ValueError("Pinned PyTorch2.6.0+cu124 required")
    device = next(model.parameters()).device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Explicit CPU test or CUDA FP32 calibration only")
    allowed = bn_keys(model)
    if not allowed:
        raise ValueError("No tracked BatchNorm buffers")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    modes = {m: m.training for m in model.modules()}
    momenta = {
        m: m.momentum
        for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    }
    try:
        model.eval()
        update_bn(batches, model, device=device)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, mode in modes.items():
            module.training = mode
    if model.state_dict().keys() != before.keys():
        raise ValueError("Model state structure changed")
    changed = []
    for key, value in model.state_dict().items():
        if tensor_bytes(value) != tensor_bytes(before[key]):
            if key not in allowed:
                raise ValueError("Non-BN state changed: " + key)
            changed.append(key)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite recalibrated state")
    return {
        "changed_bn_keys": changed,
        "non_bn_state_bitidentical": True,
        "before_bn_sha256": {k: sha(tensor_bytes(before[k])) for k in sorted(allowed)},
        "after_bn_sha256": {k: sha(tensor_bytes(model.state_dict()[k])) for k in sorted(allowed)},
        "update_bn_implementation_sha256": sha(inspect.getsource(update_bn).encode()),
        "gradient_updates": 0,
    }


class CenteredTrainingBatches:
    def __init__(self, loader, photo_ids):
        self.loader = loader
        self.photo_ids = photo_ids
        self.visits = Counter()
        self.delivered = 0

    def __iter__(self):
        for image, _, _, indices, _ in self.loader:
            if not image.is_floating_point() or image.min() < 0 or image.max() > 1:
                raise ValueError("Expected prepared RGB[0,1]")
            self.delivered += len(image)
            self.visits.update(self.photo_ids[int(i)] for i in indices)
            if self.delivered % 1024 == 0:
                print("BN TRAIN samples delivered", self.delivered, flush=True)
            yield image - 0.5


def derived_payload(parent, model, *, parent_sha, report):
    result = copy.deepcopy(parent)
    for key in list(result):
        if key.startswith("val_") or key in {
            "validation_metrics",
            "training_state",
            "checkpoint_sha256",
        }:
            result.pop(key)
    result["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result["accuracy_claim"] = False
    result["lineage"] = {
        **parent["lineage"],
        "derivation": "TRAIN_only_post_training_BN_recalibration",
        "parent_checkpoint_sha256": parent_sha,
        "gradient_training_epochs": parent["epoch"],
        "completed_parent_run_epochs": 60,
        "calibration_gradient_updates": 0,
        "predict_only": True,
        "training_resume_eligible": False,
    }
    result["bn_recalibration"] = {
        "parent_checkpoint_sha256": parent_sha,
        "parent_gradient_epoch": parent["epoch"],
        "parent_gradient_steps": parent["steps"],
        "recipe": RECIPE,
        "report": report,
        "qualification": "Derived BN buffers, no fresh validation score yet; parent original QA does not certify derived bytes",
    }
    return result


def source_digest(root):
    h = hashlib.sha256()
    for p in sorted(
        [
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    ):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "checkpoint", "manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--parent-completion-qa", type=Path, required=True)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--execute-train-calibration", action="store_true", required=True)
    a = p.parse_args()
    if any(k == "deepretina" or k.startswith("deepretina.") for k in sys.modules):
        raise ValueError("Must start isolated with frozen source import")
    if source_digest(a.source_root) != EXECUTION:
        raise ValueError("Frozen source bytes differ")
    if sha(a.manifest.read_bytes()) != MANIFEST:
        raise ValueError("Canonical manifest differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Explicit local CUDA0 calibration required")
    raw = a.checkpoint.read_bytes()
    if sha(raw) != a.checkpoint_sha256 or a.checkpoint_sha256 != PARENT_SHA:
        raise ValueError("Parent original snapshot SHA differs")
    if a.output.exists():
        raise FileExistsError("Preserve all calibration outputs")
    if a.cpu_threads != 2:
        raise ValueError("Exactly2 CPU threads required")
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.source_root / "src"))
    from deepretina.unified import CoverageSampler, UnifiedConfig, UnifiedDataset, load_unified

    qa_raw = a.parent_completion_qa.read_bytes()
    qa = json.loads(qa_raw)
    coverage = qa["best_checkpoint_coverage"]
    if (
        sha(qa_raw) != "ad76ad53e7906b2fba0fe20264c1f44a96634a3787dff2d35a95ee237bfe431f"
        or qa["kind"] != "standalone_unified_completion_qa"
        or qa["checkpoint_sha256"]["best.pt"] != PARENT_SHA
        or qa["last_epoch"] != 60
        or coverage["epoch"] != 59
        or coverage["contributing_annotation_records"] != 5700
        or coverage["seen_photos"] != 2529
    ):
        raise ValueError("Require actual typed completed parent/own BEST59 full coverage")
    model, parent, _ = load_unified(io.BytesIO(raw), device="cuda:0")
    if (
        parent["epoch"] != 59
        or parent["training_source_sha256"] != TRAINING
        or parent["prepared_content_sha256"] != CONTENT
        or parent["config"]["norm"] != "batch"
        or parent["config"]["architecture"] != "compact_unet"
        or parent["config"]["base_channels"] != 12
    ):
        raise ValueError("Wrong parent lineage/architecture/data")
    cfg = replace(UnifiedConfig(**parent["config"]), augmentation="basic")
    if any(
        getattr(cfg, k) != v
        for k, v in {
            "epoch_samples": 8192,
            "batch_size": 4,
            "seed": 42,
            "patch_size": 384,
            "foreground_probability": 0.5,
            "workers": 2,
            "coverage_first": True,
            "sampling": "dataset_task_balanced",
            "vascular_sampling_weight": 2.0,
        }.items()
    ):
        raise ValueError("Parent recipe differs")
    # Old host has no derived cache. Read immutable canonical TRAIN records directly.
    # This is explicit canonical-only I/O, never an unreported cache fallback.
    dataset = UnifiedDataset(a.manifest, "train", cfg, return_trace=True, derived_cache=None)
    if len(dataset) != 2529:
        raise ValueError("Wrong eligible TRAIN photo count")
    random.seed(42)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    generator = torch.Generator().manual_seed(42)
    sampler = CoverageSampler(dataset.sampling_weights(), 8192, generator, True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, sampler=sampler, num_workers=2, generator=generator
    )
    batches = CenteredTrainingBatches(loader, dataset.photo_ids)
    report = recalibrate(model, batches)
    if len(report["changed_bn_keys"]) != 54 or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Expected exactly54 BN-only changed buffers and zero gradients")
    if batches.delivered != 8192 or set(batches.visits) != set(dataset.photo_ids):
        raise ValueError("Missing delivered TRAIN coverage")
    if a.checkpoint.read_bytes() != raw or source_digest(a.source_root) != EXECUTION:
        raise ValueError("Parent/source changed")
    report.update(
        recipe=RECIPE,
        delivered_samples=batches.delivered,
        photo_visits=dict(sorted(batches.visits.items())),
        training_photo_coverage=2529,
        parent_sha256=a.checkpoint_sha256,
        source_commit=SOURCE,
        execution_code_sha256=EXECUTION,
        manifest_sha256=MANIFEST,
        canonical_sha256=CONTENT,
        cache_sha256=None,
        canonical_only_io=True,
        parent_training_cache_sha256=CACHE,
        original_parent_completion_qa_sha256=sha(qa_raw),
        parent_gradient_epoch=59,
        parent_completed_training_epochs=60,
        helper_sha256=sha(Path(__file__).read_bytes()),
        torch=str(torch.__version__),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        cuda_initialized=torch.cuda.is_initialized(),
        device=str(next(model.parameters()).device),
        quality_improvement_measured=False,
    )
    a.output.mkdir(parents=True, exist_ok=False)
    payload = derived_payload(parent, model, parent_sha=a.checkpoint_sha256, report=report)
    with (a.output / "derived-predict-only.pt").open("xb") as f:
        torch.save(payload, f)
    report["derived_sha256"] = sha((a.output / "derived-predict-only.pt").read_bytes())
    with (a.output / "calibration.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
