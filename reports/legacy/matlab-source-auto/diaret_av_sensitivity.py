"""One paired CPU numerical sensitivity study; never a quality metric."""

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from deepretina.legacy.av import load_av, predict_av, prepare_input
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.matlab_resize import imresize_nearest_mask
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.legacy.single_net import tiled_logits

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["vessels", "av"])
parser.add_argument("--oracle-dir", type=Path, required=True)
parser.add_argument("--model-root", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
models = args.model_root
torch.set_num_threads(2)
mode = args.mode
data = np.load(root / "diaret_sensitivity_inputs.npz", allow_pickle=False)


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


if mode == "vessels":
    snapshot = (models / "legacy-single-net-vessels.pt").read_bytes()
    model, payload = load_single_net(io.BytesIO(snapshot), "cpu")
    raw = data["raw_rgb"]
    p = prepare_single_net(
        raw, dataset="diaretdb", field_of_view_degrees=50, pixels_per_degree=25, normalized=False
    )
    with torch.inference_mode():
        logits = tiled_logits(
            model,
            p.tensor(),
            "cpu",
            crop_size=payload["tiling"]["crop_size"],
            border=payload["tiling"]["border"],
        )
    index = payload["channels"].index("vessel")
    native = p.geometry.restore(logits[index].numpy() > 0, order=0)
    target = imresize_nearest_mask(native, data["matlab_rgb"].shape[:2])
    if not target.any() or target.all():
        raise ValueError("Actual predicted support is constant; no fallback mask")
    np.savez_compressed(root / "diaret_same_predicted_vessel.npz", native=native, prepared=target)
    info = {
        "weight_sha256": hashlib.sha256(snapshot).hexdigest(),
        "original_weight_provenance": payload["provenance"],
        "raw_rgb_sha256": digest(raw),
        "native_vessel_sha256": digest(native),
        "prepared_vessel_sha256": digest(target),
        "native_shape": list(native.shape),
        "prepared_shape": list(target.shape),
        "native_positive": int(native.sum()),
        "prepared_positive": int(target.sum()),
        "threshold": "strict prepared logit>0, nearest restore then source-nearest to AV shape",
        "normalized": False,
        "field_angle": 50,
        "pixels_per_degree": 25,
    }
    (root / "diaret_vessel_support.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )
    print("vessel support", int(target.sum()), flush=True)
else:
    support = np.load(root / "diaret_same_predicted_vessel.npz", allow_pickle=False)
    vessel = support["prepared"]
    snapshot = (models / "legacy-av-v9.pt").read_bytes()
    model, payload = load_av(io.BytesIO(snapshot), "cpu")
    a, b = data["matlab_rgb"], data["python_rgb"]
    ia, ib = prepare_input(a, vessel), prepare_input(b, vessel)

    def normalize(t):
        return (t - t.mean((2, 3), keepdim=True)) / t.std((2, 3), keepdim=True)

    na, nb = normalize(ia), normalize(ib)
    d = (na - nb).abs()
    norm_error = {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "per_channel_max": [float(d[:, i].max()) for i in range(4)],
    }
    del na, nb, d
    gc.collect()
    print("source AV forward", flush=True)
    first = predict_av(model, a, vessel, "cpu", original_size=data["raw_rgb"].shape[:2])
    print("Python AV forward", flush=True)
    second = predict_av(model, b, vessel, "cpu", original_size=data["raw_rgb"].shape[:2])

    def difference(x, y):
        d = np.abs(x - y)
        return {
            "max": float(d.max()),
            "mean": float(d.mean()),
            "unequal": int(np.count_nonzero(x != y)),
        }

    def counts(x):
        return {str(i): int(np.count_nonzero(x == i)) for i in range(3)}

    report = {
        "scope": "One paired preprocessing numerical sensitivity measurement, not quality/GT/heldout evaluation",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "weight_sha256": hashlib.sha256(snapshot).hexdigest(),
        "weight_provenance": payload["provenance"],
        "input_npz_sha256": hashlib.sha256(
            (root / "diaret_sensitivity_inputs.npz").read_bytes()
        ).hexdigest(),
        "source_stored_rgb_sha256": digest(a),
        "python_stored_rgb_sha256": digest(b),
        "same_prepared_vessel_sha256": digest(vessel),
        "support": json.loads((root / "diaret_vessel_support.json").read_text()),
        "input_standardization_error": norm_error,
        "logit_difference": difference(first["prepared_logits"], second["prepared_logits"]),
        "conditional_probability_difference": difference(
            first["prepared_conditional_probabilities"],
            second["prepared_conditional_probabilities"],
        ),
        "prepared_gated_classes_disagree": int(
            np.count_nonzero(first["prepared_classes"] != second["prepared_classes"])
        ),
        "prepared_class_counts": {
            "matlab": counts(first["prepared_classes"]),
            "python": counts(second["prepared_classes"]),
        },
        "native_gated_classes_disagree": int(
            np.count_nonzero(first["classes"] != second["classes"])
        ),
        "native_class_counts": {
            "matlab": counts(first["classes"]),
            "python": counts(second["classes"]),
        },
    }
    (root / "diaret-av-sensitivity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        "done",
        report["prepared_gated_classes_disagree"],
        report["native_gated_classes_disagree"],
        flush=True,
    )
