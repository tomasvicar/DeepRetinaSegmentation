"""Verify actual short CPU runs; never imply all-data training or quality gains."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from deepretina.integrity import verify_prepared_content
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy_training import LegacyTrainingConfig, build_fresh, require_finite
from deepretina.unified import read_unified_sample


def verify(base, data):
    base, data = Path(base), Path(data)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = json.loads((data / "manifest.json").read_text())
    reports = {}
    for case, task in (
        ("vessel", "vessel"),
        ("disc", "disc"),
        ("disc256", "disc"),
        ("universal", "universal"),
    ):
        exit_proof = json.loads((base / (case + "-observed-exit.json")).read_text())
        if exit_proof["observed_returncode"] is not True or exit_proof["returncode"] != 0:
            raise ValueError("Actual unsuccessful training child: " + case)
        model, payload = load_single_net(base / case / "last.pt", "cpu")
        require_finite(payload["state_dict"])
        require_finite(payload["training_state"]["optimizer"])
        recipe = payload["training_recipe"]
        if (
            not recipe["fresh_initialization"]
            or not recipe["architecture_matches_original_default"]
        ):
            raise ValueError("Require fresh exact production architecture")
        row = next(
            r
            for r in rows
            if r["split"] == "train" and r["task"] == ("av" if task == "vessel" else "disc_cup")
        )
        image, _, _ = read_unified_sample(data, row)
        with torch.inference_mode():
            out = model((image[:, :64, :64] - 0.5)[None])
        if not torch.isfinite(out).all():
            raise ValueError("Nonfinite reloaded native forward")
        torch.manual_seed(42)
        initial = build_fresh(LegacyTrainingConfig(task=task)).state_dict()
        names = set(dict(model.named_parameters()))
        changed = [k for k in names if not torch.equal(initial[k], payload["state_dict"][k])]
        if not changed:
            raise ValueError("No actual parameter update relative to same-seed initialization")
        coverage = json.loads((base / case / "coverage.json").read_text())
        summary = json.loads((base / case / "training_summary.json").read_text())
        if not (
            coverage["all_eligible_annotations_contributed"]
            and coverage["all_eligible_photos_seen"]
            and summary["completed_epochs"] == 1
            and summary["steps"] > 0
            and summary["test_evaluation_performed"] is False
        ):
            raise ValueError("Incomplete smoke coverage/budget or unexpected TEST")
        reports[case] = {
            "architecture": payload["architecture"],
            "channels": payload["channels"],
            "steps": summary["steps"],
            "coverage": coverage,
            "changed_parameter_tensors_vs_same_seed_fresh": len(changed),
            "reloaded_real_TRAIN_crop_finite_shape": list(out.shape),
            "last_sha256": hashlib.sha256((base / case / "last.pt").read_bytes()).hexdigest(),
            "finite_model_and_optimizer": True,
            "actual_exit": exit_proof,
        }
        del model, payload, initial, out
    proof = {
        "kind": "four_original_production_single_net_real_smoke_verification",
        "scope": "one-epoch functional only; no new quality or all-data training claim",
        "cases": reports,
        "subset_fingerprint_after": verify_prepared_content(data / "manifest.json"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "TEST_evaluation": False,
    }
    if proof["cuda_initialized"]:
        raise ValueError("CPU smoke unexpectedly initialized CUDA")
    with (base / "root-four-native-verification.json").open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    verify(args.runs, args.data)
