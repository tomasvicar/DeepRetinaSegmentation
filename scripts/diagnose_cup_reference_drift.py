"""One preselected failed VAL photo only; CPU native cup numerical diagnosis."""

import argparse
import datetime
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import UnifiedConfig, UnifiedDataset

PHOTO = "sha256:402f3347e4c847e0beade05125d2c98d73751d2ddd0e8352da5d2002afec23b2"
EXPECTED = {"tp": 4125, "fp": 19777, "fn": 0, "valid_pixels": 885195}
MODEL = "9fa4f9ffddb2fa39828f32bde85299ffaff74b7b8817a70c70243fa11362569f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def counts(pred, target, valid):
    return {
        "tp": int((pred & target & valid).sum()),
        "fp": int((pred & ~target & valid).sum()),
        "fn": int((~pred & target & valid).sum()),
        "valid_pixels": int(valid.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, choices=[1, 2, 4, 8], required=True)
    p.add_argument("--input-only", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    dataset = UnifiedDataset(a.manifest, "val", UnifiedConfig())
    index = dataset.photo_ids.index(PHOTO)
    image, target, valid, trace = dataset.sample(index, random_annotator=False, return_trace=True)
    image = image - 0.5
    t = target[4].numpy() >= 0.5
    v = valid[4].numpy()
    result = {
        "photo_id": PHOTO,
        "source": "g1020",
        "annotation": "g1020_image_532_disc_cup",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "image_tensor_sha256": sha(image.numpy().tobytes()),
        "cup_target_sha256": sha(t.tobytes()),
        "cup_valid_sha256": sha(v.tobytes()),
        "shape": list(image.shape),
        "trace": trace,
        "manifest_sha256": sha(a.manifest.read_bytes()),
        "reference_native_cup": EXPECTED,
        "unified_file": str(Path(__import__("deepretina.unified", fromlist=["__file__"]).__file__)),
        "torch": str(torch.__version__),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
        "input_only": a.input_only,
        "no_gpu": True,
        "no_test": True,
    }
    if not a.input_only:
        assert sha(a.checkpoint.read_bytes()) == MODEL
        model, payload = load_single_net(a.checkpoint, "cpu")
        before = time.perf_counter()
        with torch.inference_mode():
            logits = tiled_logits(
                model, image, "cpu", payload["tiling"]["crop_size"], payload["tiling"]["border"]
            )[2]
        assert torch.isfinite(logits).all()
        array = logits.numpy()
        np.save(a.output / "cup-logits.npy", array)
        np.savez_compressed(a.output / "cup-gt.npz", target=t, valid=v)
        result.update(
            checkpoint_sha256=MODEL,
            tiling=payload["tiling"],
            logits_dtype=str(logits.dtype),
            logits_sha256=sha(array.tobytes()),
            inference_seconds=time.perf_counter() - before,
            native_counts=counts(array > 0, t, v),
            sigmoid_float64_strict_counts=counts(torch.sigmoid(logits).numpy() > 0.5, t, v),
            sigmoid_float32_strict_counts=counts(torch.sigmoid(logits.float()).numpy() > 0.5, t, v),
            legacy_forward_file_sha256=sha(
                Path(
                    __import__("deepretina.legacy.single_net", fromlist=["__file__"]).__file__
                ).read_bytes()
            ),
        )
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
