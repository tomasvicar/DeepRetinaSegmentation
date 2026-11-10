import json
import math
from pathlib import Path

import numpy as np

root = (
    Path.home() / "code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z/forensic"
)
arrays = {
    name: np.load(root / name / "cup-logits.npy", allow_pickle=False)
    for name in ["cpu-1", "cpu-2", "cpu-4", "cpu-8", "original87-wrapper"]
}
with np.load(root / "cpu-2/cup-gt.npz", allow_pickle=False) as f:
    valid = f["valid"]
    target = f["target"]
base = arrays["cpu-2"]
comparisons = {}
for name, a in arrays.items():
    y, x = np.where(((base > 0) != (a > 0)) & valid)
    comparisons[name] = {
        "max_abs_logit_difference": float(np.max(np.abs(a - base))),
        "valid_binary_disagreement_pixels": len(y),
        "first_disagreements": [
            {
                "y": int(i),
                "x": int(j),
                "target_cup": bool(target[i, j]),
                "cpu2_logit": float(base[i, j]),
                "compared_logit": float(a[i, j]),
            }
            for i, j in list(zip(y, x))[:20]
        ],
    }
flat = np.flatnonzero(valid)
candidate = flat[np.argsort(np.abs(base.ravel()[flat]))[:12]]
near = []
for index in candidate:
    y, x = np.unravel_index(index, base.shape)
    near.append(
        {
            "y": int(y),
            "x": int(x),
            "valid": True,
            "target_cup": bool(target[y, x]),
            "logits_by_variant": {name: float(a[y, x]) for name, a in arrays.items()},
            "sigmoid_float64_by_variant": {
                name: float(1 / (1 + math.exp(-float(a[y, x])))) for name, a in arrays.items()
            },
            "classification": bool(base[y, x] > 0),
            "status": "nearest-zero diagnostic candidate, NOT identified historical mismatching pixel",
        }
    )
result = {
    "photo_id": "sha256:402f3347e4c847e0beade05125d2c98d73751d2ddd0e8352da5d2002afec23b2",
    "source": "g1020",
    "record_id": "g1020_image_532_disc_cup",
    "prepared_shape": list(base.shape),
    "historical_native_cup_counts": {"tp": 4125, "fp": 19777, "fn": 0, "valid_pixels": 885195},
    "current_all_variants_native_counts": {
        "tp": 4125,
        "fp": 19778,
        "fn": 0,
        "valid_pixels": 885195,
    },
    "source87_and_b413_input_gt_valid_hashes_identical": True,
    "legacy_forward_source_identical": True,
    "comparisons_against_cpu2": comparisons,
    "nearest_zero_valid_pixels": near,
    "historical_exact_pixel_location": "Unknown: historical per-pixel logits/masks were not saved; aggregate FP counts alone do not identify a coordinate",
    "conclusion": "Current CPU1/2/4/8 and original87 wrapper do not reproduce historical1FP difference. Thread count and strictsigmoid-vs-logit theories are not supported by these bounded runs. Underlying historical numerical-state difference remains unresolved.",
    "no_full_restart": True,
    "no_gate_relaxation": True,
    "no_quality_improvement_claim": True,
    "no_test_evaluation": True,
}
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"comparisons": comparisons, "closest": near[0]}))
