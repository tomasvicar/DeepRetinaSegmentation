"""Summarize saved CPU parity arrays and source bytes; no model or GT loading."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compare_arrays(left, right):
    result = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if a.files != b.files:
            raise ValueError("Parity array inventories differ")
        for key in a.files:
            x, y = a[key], b[key]
            result[key] = {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "array_equal": bool(np.array_equal(x, y)),
                "left_sha256": sha(x.tobytes()),
                "right_sha256": sha(y.tobytes()),
            }
            if np.issubdtype(x.dtype, np.number):
                result[key]["maximum_absolute_difference"] = float(
                    np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))
                )
    return result


def summarize(left, right, output):
    root = Path(__file__).parents[1]
    before = "6b3dfad605a256e904a877efcdcc1f3cc6135dd0"
    after = subprocess.check_output(["git", "rev-parse", "f1c4c99"], text=True, cwd=root).strip()
    files = [
        "src/deepretina/" + name
        for name in [
            "model.py",
            "unified.py",
            "geometry_policy.py",
            "dataset.py",
            "inference.py",
            "legacy/preprocessing.py",
            "legacy/single_net.py",
            "legacy/matlab_resize.py",
        ]
    ]
    source = {}
    texts = {}
    for name in files:
        blobs = [
            subprocess.check_output(["git", "show", ref + ":" + name], cwd=root)
            for ref in (before, after)
        ]
        source[name] = {
            "frozen_sha256": sha(blobs[0]),
            "stable_sha256": sha(blobs[1]),
            "byte_identical": blobs[0] == blobs[1],
        }
        texts[name] = blobs
    function_equality = {}
    functions = [
        "_activation",
        "_conditional_probabilities",
        "_joint_probabilities",
        "restore_probabilities",
        "predict_probabilities",
        "decode_probabilities",
        "evaluate_unified_model",
        "load_unified",
        "inference_config",
        "predict_unified",
    ]
    trees = [ast.parse(raw) for raw in texts["src/deepretina/unified.py"]]
    for name in functions:
        functions_ast = [
            next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for t in trees
        ]
        function_equality[name] = ast.dump(functions_ast[0], include_attributes=False) == ast.dump(
            functions_ast[1], include_attributes=False
        )
    cases = {
        name: compare_arrays(left / path, right / path)
        for name, path in [
            ("synthetic_intermediates", "synthetic-intermediates.npz"),
            ("actual_train_drions", "actual-drions/probabilities.npz"),
            ("synthetic_raw_drions", "synthetic-drions/probabilities.npz"),
        ]
    }
    masks = {}
    for case in ["actual-drions", "synthetic-drions"]:
        masks[case] = {}
        for name in ["vessel", "artery", "vein", "disc", "cup", "av", "overlay"]:
            x, y = left / case / (name + ".png"), right / case / (name + ".png")
            masks[case][name] = {
                "pixel_equal": bool(
                    np.array_equal(np.asarray(Image.open(x)), np.asarray(Image.open(y)))
                ),
                "png_bytes_equal": x.read_bytes() == y.read_bytes(),
                "sha256": sha(x.read_bytes()),
            }
    with np.load(left / "nearest-helper.npz") as a, np.load(right / "nearest-helper.npz") as b:
        changed = a["mask"] != b["mask"]
        nearest = {
            "synthetic_only": True,
            "input_shape": [1152, 11],
            "output_shape": [1005, 11],
            "mask_rule": "(row+column)%2==0",
            "changed_pixels": int(changed.sum()),
            "changed_zero_based_rows": np.unique(np.where(changed)[0]).tolist(),
            "compact_inference_dependency": False,
        }
    retained = Path(
        "/var/home/tom/code/DeepRetinaSegmentation_model_backups/v2-control-e002-functional-smoke/prediction/probabilities.npz"
    )
    retained_comparison = compare_arrays(retained, right / "actual-drions/probabilities.npz")
    workers = [json.loads((directory / "worker.json").read_text()) for directory in (left, right)]
    equal = all(v["array_equal"] for case in cases.values() for v in case.values()) and all(
        v["pixel_equal"] and v["png_bytes_equal"] for case in masks.values() for v in case.values()
    )
    if not equal:
        raise ValueError("Observed compact inference parity failed")
    result = {
        "frozen_source": before,
        "stable_source": after,
        "scope": "E2 compact_unet on CPU, same pinned environment; TRAIN raw image plus synthetic inputs only",
        "checkpoint_sha256": workers[0]["checkpoint_sha256"],
        "image_sha256": workers[0]["image_sha256"],
        "all_observed_compact_outputs_bitwise_equal": equal,
        "source_files": source,
        "unified_function_ast_identical": function_equality,
        "numeric_cases": cases,
        "output_images": masks,
        "retained_e2_smoke_comparison": retained_comparison,
        "nearest_counterexample_outside_compact_route": nearest,
        "worker_receipts": workers,
        "qualifications": [
            "Not GPU/backend/version-wide proof or proof for all future checkpoints",
            "No validation/test GT or quality metric read/calculated",
            "Inference compatibility does not permit cross-source training resume; frozen training hash remains authoritative",
            "Modified nearest helper affects separately selected original MATLAB mask routes; no such route used by compact inference",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("left", "right", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.left, args.right, args.output)
