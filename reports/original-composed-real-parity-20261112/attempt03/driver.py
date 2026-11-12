"""One authorized TRAIN photo, two composed operations, no scoring or GT reads."""

import ast
import hashlib
import json
import os
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert sorted(os.sched_getaffinity(0)) == [4]
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from deepretina import original as api

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    manifest = json.loads((ROOT / "stage-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        assert sha(ROOT / relative) == expected, relative
    data = Path(
        "/var/home/tom/code/DeepRetinaSegmentation_proposed_storage/corpusv2_2591_fa1cf03/manifest.json"
    ).read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
    )
    rows = [r for r in json.loads(data) if r["id"] == "drive_21_training_vessels"]
    assert len(rows) == 1 and rows[0]["split"] == "train"
    assert rows[0]["metadata"]["source_image"] == "training/images/21_training.tif"
    with Image.open(ROOT / "inputs/image.tif") as im:
        rgb = np.array(im.convert("RGB"))
    with Image.open(ROOT / "inputs/fov.gif") as im:
        fov = np.array(im.convert("L")) > 0
    assert rgb.shape == (584, 565, 3) and fov.shape == rgb.shape[:2]
    photo_hash = hashlib.sha256()
    photo_hash.update(np.asarray(rgb.shape, dtype="<i8").tobytes())
    photo_hash.update(np.ascontiguousarray(rgb).tobytes())
    digest = photo_hash.hexdigest()
    assert (
        digest
        == rows[0]["metadata"]["raw_rgb_sha256"]
        == "77d1a4436a1cac7f3397c9a2f2e84bf1e2d2b13647a833a3d7559aa6aa0d6925"
    )
    assert rows[0]["photo_id"] == "sha256:" + digest
    assert (
        sha(ROOT / "benchmark-original.py")
        == "3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae"
    )
    tree = ast.parse((ROOT / "benchmark-original.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "original_operation"
    )
    namespace = {
        k: getattr(api, k)
        for k in (
            "np",
            "torch",
            "prepare_single_net",
            "tiled_logits",
            "prepare_vesclass_matlab_calibrated",
            "predict_av_input",
        )
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "benchmark-original.py", "exec"),
        namespace,
    )
    load = api._load_components
    captured = []

    def capture(device):
        value = load(device)
        captured.append(value)
        return value

    api._load_components = capture
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        candidate = api.predict_original(
            ROOT / "inputs/image.tif",
            fov=ROOT / "inputs/fov.gif",
            dataset="drive",
            device="cpu",
            output=ROOT / "prediction",
        )
        api._load_components = load
        candidate_seconds = time.monotonic() - started
        models, av, identities = captured[0]
        # Exact benchmark main loader normalization; preserve API full provenance.
        models = {
            role: (
                model,
                {
                    **payload,
                    "tiling": {key: payload["tiling"][key] for key in ("crop_size", "border")},
                },
            )
            for role, (model, payload) in models.items()
        }
        started = time.monotonic()
        baseline = namespace["original_operation"](
            rgb,
            fov,
            models,
            av,
            torch.device("cpu"),
            {"dataset": "drive", "pixels_per_degree": 25},
            cup_policy="inside-disc",
        )
        baseline_seconds = time.monotonic() - started
    differences = np.count_nonzero(candidate.masks != baseline, axis=(1, 2)).tolist()
    assert differences == [0] * 5, differences
    assert np.isfinite(candidate.probabilities).all()
    assert candidate.masks.shape == baseline.shape == (5, 584, 565)
    assert {k: v["sha256"] for k, v in identities.items()} == {
        k: v[1] for k, v in api.COMPONENTS.items()
    }
    for relative, expected in manifest["files"].items():
        assert sha(ROOT / relative) == expected
    assert not torch.cuda.is_initialized()
    report = {
        "status": "passed",
        "scope": "one raw TRAIN DRIVE21 photo; exact masks, no GT/VAL/TEST/quality claim",
        "source_commit": manifest["source_commit"],
        "staged_files_unchanged": True,
        "manifest_sha256": hashlib.sha256(data).hexdigest(),
        "train_record": rows[0],
        "input_sha256": sha(ROOT / "inputs/image.tif"),
        "fov_sha256": sha(ROOT / "inputs/fov.gif"),
        "benchmark_script_sha256": sha(ROOT / "benchmark-original.py"),
        "quality_policy_script_sha256": candidate.metadata["quality_policy_script_sha256"],
        "components": identities,
        "channels": list(api.CHANNELS),
        "shape": list(baseline.shape),
        "mask_differences": differences,
        "positive_pixels": candidate.masks.sum(axis=(1, 2)).tolist(),
        "candidate_mask_sha256": hashlib.sha256(candidate.masks.tobytes()).hexdigest(),
        "baseline_mask_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "finite_probabilities": True,
        "probability_shape": list(candidate.probabilities.shape),
        "candidate_seconds_including_load_export": candidate_seconds,
        "baseline_seconds_reusing_models": baseline_seconds,
        "timing_not_a_speed_comparison": True,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "loader_instrumentation": "capture actual API loader return once; same loaded immutable models reused by literal benchmark function",
    }
    (ROOT / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
