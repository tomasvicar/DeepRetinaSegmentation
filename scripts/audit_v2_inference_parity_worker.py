"""Read-only-source CPU parity worker; only writes into its new audit output root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source, checkpoint, image, output):
    import deepretina
    from deepretina.unified import (
        inference_config,
        load_unified,
        predict_probabilities,
        predict_unified,
    )

    def require(ok):
        if not ok:
            raise ValueError("Audit package/input identity differs")

    require(Path(deepretina.__file__).resolve().parent == Path(source).resolve() / "src/deepretina")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def blocked(*args, **kwargs):
        raise AssertionError("CUDA initialization forbidden during CPU parity audit")

    torch.cuda.init = blocked
    torch.cuda._lazy_init = blocked
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_before, image_before = digest(checkpoint), digest(image)
    model, payload, _device = load_unified(checkpoint, device="cpu")
    require(payload["config"]["architecture"] == "compact_unet")
    config = inference_config(payload)
    arrays = {}
    for height, width in [(64, 64), (37, 53), (17, 19)]:
        generator = torch.Generator().manual_seed(518 + height + width)
        values = torch.rand((3, height, width), generator=generator)
        with torch.inference_mode():
            arrays[f"logits_{height}x{width}"] = model((values - 0.5)[None])[0].numpy()
            arrays[f"probability_{height}x{width}"] = predict_probabilities(
                model, values, "cpu", config, tile_size=32, overlap=8
            )
    np.savez_compressed(output / "synthetic-intermediates.npz", **arrays)
    actual = predict_unified(
        checkpoint, image, output / "actual-drions", device="cpu", dataset="drions"
    )
    synthetic = np.random.default_rng(717).integers(0, 256, (67, 91, 3), dtype=np.uint8)
    synthetic_path = output / "synthetic.png"
    Image.fromarray(synthetic).save(synthetic_path)
    synthetic_report = predict_unified(
        checkpoint,
        synthetic_path,
        output / "synthetic-drions",
        device="cpu",
        dataset="drions",
        fov=np.ones((67, 91), bool),
    )
    actual = json.loads(Path(actual["metadata"]).read_text())
    synthetic_report = json.loads(Path(synthetic_report["metadata"]).read_text())
    # This changed MATLAB helper is measured separately; compact prediction above
    # never imports/calls it (its preparation uses unchanged skimage resize).
    from deepretina.legacy.matlab_resize import imresize_nearest_mask

    nearest_mask = ((np.arange(1152)[:, None] + np.arange(11)[None, :]) % 2) == 0
    np.savez_compressed(
        output / "nearest-helper.npz", mask=imresize_nearest_mask(nearest_mask, (1005, 11))
    )
    require(digest(checkpoint) == checkpoint_before and digest(image) == image_before)
    result = {
        "source_root": str(source),
        "package_path": str(deepretina.__file__),
        "checkpoint_sha256": checkpoint_before,
        "image_sha256": image_before,
        "architecture": payload["architecture"],
        "configuration": payload["config"],
        "checkpoint_training_source_sha256": payload["training_source_sha256"],
        "geometry_policy_sha256": payload["geometry_policy"]["sha256"],
        "checkpoint_epoch": payload["epoch"],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_init_blocked": True,
        "real_case_scope": "Existing TRAIN DRIONS input only; no labels opened",
        "actual_preprocessing": actual["preprocessing"],
        "synthetic_preprocessing": synthetic_report["preprocessing"],
        "synthetic_generation": "NumPy default_rng717 uint8[0,256),67x91 RGB; provided all-ones synthetic FOV",
        "source_and_input_unchanged": True,
    }
    (output / "worker.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "checkpoint", "image", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.checkpoint, args.image, args.output)
