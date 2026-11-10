"""CPU-only numerical sensitivity probe; goldens and source vessels stay external."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

from deepretina.legacy.av import load_av, predict_av_input, prepare_input


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {"mae": float(d.mean()), "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    before = digest(args.checkpoint)
    model, _ = load_av(args.checkpoint, "cpu")
    rows = []
    for name, dataset, resource in [
        ("drive", "DRIVE", "training/1st_manual/21_manual1.gif"),
        ("hrf", "HRF", "manual1/01_g.tif"),
    ]:
        start = time.monotonic()
        reference = loadmat(args.golden_dir / f"{name}_golden.mat")
        with np.load(args.golden_dir / f"{name}_python_profile.npz") as cache:
            rgb = cache["stored_rgb"]
            assert np.array_equal(cache["fov"], reference["prepared_fov"])
        with zipfile.ZipFile(args.database_dir / f"{dataset}.zip") as archive:
            source_bytes = archive.read(resource)
        original = np.asarray(Image.open(io.BytesIO(source_bytes)).convert("L")) > 127
        h, w = rgb.shape[:2]
        # MATLAB nearest resize to explicit output size, independently tested by profile fixtures.
        iy = np.minimum(
            ((np.arange(h) + 0.5) * original.shape[0] / h).astype(int), original.shape[0] - 1
        )
        ix = np.minimum(
            ((np.arange(w) + 0.5) * original.shape[1] / w).astype(int), original.shape[1] - 1
        )
        vessel = original[iy[:, None], ix[None, :]]
        assert 0 < vessel.sum() < vessel.size
        a, b = prepare_input(reference["stored"], vessel), prepare_input(rgb, vessel)

        def standardize(x):
            return (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), keepdim=True)

        standard_error = error(standardize(a).numpy(), standardize(b).numpy())
        ref, pred = predict_av_input(model, a), predict_av_input(model, b)
        logits_a, logits_b = ref["prepared_logits"], pred["prepared_logits"]
        ca, cb = logits_a[1:].argmax(0), logits_b[1:].argmax(0)
        fov = reference["prepared_fov"].astype(bool)
        rows.append(
            {
                "name": name,
                "shape": list(rgb.shape),
                "vessel_source": resource,
                "vessel_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "vessel_prepared_sha256": hashlib.sha256(vessel.tobytes()).hexdigest(),
                "vessel_pixels": int(vessel.sum()),
                "same_vessel_for_both_inputs": True,
                "standardized_input_error": standard_error,
                "logits_all": error(logits_a, logits_b),
                "logits_fov": error(logits_a[:, fov], logits_b[:, fov]),
                "logits_by_channel": [error(a, b) for a, b in zip(logits_a, logits_b)],
                "conditional_probability_error": error(
                    ref["prepared_conditional_probabilities"],
                    pred["prepared_conditional_probabilities"],
                ),
                "conditional_argmax_disagreements_all": int((ca != cb).sum()),
                "conditional_argmax_disagreements_fov": int(((ca != cb) & fov).sum()),
                "conditional_argmax_disagreements_vessels": int(((ca != cb) & vessel).sum()),
                "gated_class_disagreements": int(
                    (ref["prepared_classes"] != pred["prepared_classes"]).sum()
                ),
                "wall_seconds": time.monotonic() - start,
            }
        )
        print(name, json.dumps(rows[-1]), flush=True)
    after = digest(args.checkpoint)
    assert before == after
    report = {
        "scope": "Numerical preprocessing sensitivity, not accuracy benchmark or universal parity certificate",
        "device": "cpu",
        "threads": 4,
        "torch_version": torch.__version__,
        "checkpoint_sha256": before,
        "checkpoint_unchanged": True,
        "nonconstant_reference_GT_vessel_input": True,
        "historical_training_file_provenance_verified": False,
        "production_activation": False,
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
