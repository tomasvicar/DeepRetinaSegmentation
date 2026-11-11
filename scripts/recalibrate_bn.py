"""EXPERIMENT: TRAIN-only BN recalibration of fixed parent last60, CPU only.

Prep-only until root review. No quality claim/selection/TEST/resume route.
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

SOURCE = "6b3dfad605a256e904a877efcdcc1f3cc6135dd0"
EXECUTION = "cb5c4f98e0181a13b166c128a632faf502de1a3bf770fbee4a5b7fa649e651cd"
TRAINING = "9075246883f938ba80b4fe282b365da10ddb8c834bd3106d846465ac79ce9473"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
PARENT_SHA = "10670cffa9c82305558d1f3ab1275b532a287d5c5016546179416b81dc32fab2"
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
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("CPU only")
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
        update_bn(batches, model, device=torch.device("cpu"))
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
        "gradient_training_epochs": 60,
        "calibration_gradient_updates": 0,
        "predict_only": True,
        "training_resume_eligible": False,
    }
    result["bn_recalibration"] = {
        "parent_checkpoint_sha256": parent_sha,
        "parent_gradient_epoch": 60,
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
    for key in ("source-root", "checkpoint", "manifest", "cache", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--execute-train-calibration", action="store_true", required=True)
    a = p.parse_args()
    if any(k == "deepretina" or k.startswith("deepretina.") for k in sys.modules):
        raise ValueError("Must start isolated with frozen source import")
    if source_digest(a.source_root) != EXECUTION:
        raise ValueError("Frozen source bytes differ")
    if sha(a.manifest.read_bytes()) != MANIFEST:
        raise ValueError("Canonical manifest differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CUDA-hidden CPU calibration required")
    raw = a.checkpoint.read_bytes()
    if sha(raw) != a.checkpoint_sha256 or a.checkpoint_sha256 != PARENT_SHA:
        raise ValueError("Parent original snapshot SHA differs")
    if a.output.exists():
        raise FileExistsError("Preserve all calibration outputs")
    if not 1 <= a.cpu_threads <= 4:
        raise ValueError("CPU thread bound1..4 required")
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.source_root / "src"))
    from deepretina.derived_cache import VerifiedCache
    from deepretina.unified import CoverageSampler, UnifiedConfig, UnifiedDataset, load_unified

    model, parent, _ = load_unified(io.BytesIO(raw), device="cpu")
    if (
        parent["epoch"] != 60
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
        }.items()
    ):
        raise ValueError("Parent recipe differs")
    cache = VerifiedCache(
        a.cache,
        a.manifest,
        expected_cache_fingerprint=CACHE,
        expected_canonical_fingerprint=CONTENT,
        allow_canonical_fallback=False,
    )
    dataset = UnifiedDataset(a.manifest, "train", cfg, return_trace=True, derived_cache=cache)
    cache.require_coverage(dataset.rows)
    if len(dataset) != 2529:
        raise ValueError("Wrong eligible TRAIN photo count")
    random.seed(42)
    np.random.seed(42)
    torch.random.default_generator.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    sampler = CoverageSampler(dataset.sampling_weights(), 8192, generator, True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, sampler=sampler, num_workers=2, generator=generator
    )
    batches = CenteredTrainingBatches(loader, dataset.photo_ids)
    report = recalibrate(model, batches)
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
        cache_sha256=CACHE,
        helper_sha256=sha(Path(__file__).read_bytes()),
        torch=str(torch.__version__),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        cuda_initialized=torch.cuda.is_initialized(),
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
