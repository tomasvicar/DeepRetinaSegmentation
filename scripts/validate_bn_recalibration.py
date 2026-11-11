"""Research-only fresh VAL for a checked derived BN checkpoint; no TEST route.

No stale training score is added. This does not mint standard selection evidence.
Root review required before first actual evaluation of this new report semantics.
"""

import argparse
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import recalibrate_bn as recipe

CALIBRATOR_SHA = "8825ebf4b229e11ea9adf9a64c80c28fb67dbd6c009271ff23d4401ac94db26d"


def validate_derivation(model, parent, derived, report, *, parent_sha, derived_sha):
    if (
        parent_sha != recipe.PARENT_SHA
        or report["parent_sha256"] != parent_sha
        or report["derived_sha256"] != derived_sha
    ):
        raise ValueError("Original parent/derived/calibration snapshot differs")
    if (
        derived["config"] != parent["config"]
        or derived["steps"] != parent["steps"]
        or report["recipe"] != recipe.RECIPE
        or report["helper_sha256"] != CALIBRATOR_SHA
    ):
        raise ValueError("Parent gradient history/config/calibration recipe differs")
    if derived["epoch"] != 60 or parent["epoch"] != 60:
        raise ValueError("Require parent gradient60")
    if (
        report["delivered_samples"] != 8192
        or len(report["photo_visits"]) != 2529
        or sum(report["photo_visits"].values()) != 8192
        or min(report["photo_visits"].values()) < 1
        or report["gradient_updates"] != 0
        or report["non_bn_state_bitidentical"] is not True
    ):
        raise ValueError("Calibration delivered coverage/zero-gradient proof differs")
    if (
        derived["lineage"].get("parent_checkpoint_sha256") != parent_sha
        or derived["lineage"].get("predict_only") is not True
        or derived["lineage"].get("calibration_gradient_updates") != 0
        or "training_state" in derived
        or "validation_metrics" in derived
        or any(k.startswith("val_") for k in derived)
    ):
        raise ValueError("Derived lineage/stale score differs")
    inner = derived["bn_recalibration"]["report"]
    if inner != {k: v for k, v in report.items() if k != "derived_sha256"}:
        raise ValueError("Original sidecar/calibration payload differs")
    allowed = recipe.bn_keys(model)
    state = derived["state_dict"]
    if state.keys() != parent["state_dict"].keys():
        raise ValueError("Model state keys differ")
    for key, value in state.items():
        if key not in allowed and recipe.tensor_bytes(value) != recipe.tensor_bytes(
            parent["state_dict"][key]
        ):
            raise ValueError("Non-BN parent bytes differ: " + key)
        if (
            key in allowed
            and recipe.sha(recipe.tensor_bytes(value)) != report["after_bn_sha256"][key]
        ):
            raise ValueError("Recalibrated BN bytes differ")
    return {
        "independently_checked_non_bn_bitidentical": True,
        "parent_gradient_epoch": 60,
        "calibration_gradient_updates": 0,
        "selection_or_test_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source-root",
        "checkpoint",
        "parent",
        "calibration",
        "manifest",
        "cache",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("checkpoint-sha256", "calibration-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--cpu-threads", type=int, default=4)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.manifest.parent.resolve()):
        raise ValueError("Write research report outside canonical corpus")
    if a.output.exists() or a.output.is_symlink():
        raise FileExistsError("Preserve research VAL report")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only CUDA-hidden route")
    if any(k == "deepretina" or k.startswith("deepretina.") for k in sys.modules):
        raise ValueError("Start fresh with frozen source")
    if (
        recipe.source_digest(a.source_root) != recipe.EXECUTION
        or recipe.sha(Path(recipe.__file__).read_bytes()) != CALIBRATOR_SHA
    ):
        raise ValueError("Frozen evaluator/calibrator bytes differ")
    blobs = {k: Path(getattr(a, k)).read_bytes() for k in ("checkpoint", "parent", "calibration")}
    if (
        recipe.sha(blobs["checkpoint"]) != a.checkpoint_sha256
        or recipe.sha(blobs["parent"]) != recipe.PARENT_SHA
        or recipe.sha(blobs["calibration"]) != a.calibration_sha256
    ):
        raise ValueError("External original byte pins differ")
    if not 1 <= a.cpu_threads <= 4:
        raise ValueError("CPU1..4 only")
    import torch

    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(a.source_root / "src"))
    from deepretina import selection as gate
    from deepretina.derived_cache import VerifiedCache
    from deepretina.unified import evaluate_unified_model, inference_config, load_unified
    from deepretina.validation_evidence import evaluation_runtime

    parent = torch.load(io.BytesIO(blobs["parent"]), weights_only=True, map_location="cpu")
    model, derived, device = load_unified(io.BytesIO(blobs["checkpoint"]), device="cpu")
    report = json.loads(blobs["calibration"])
    derivation = validate_derivation(
        model,
        parent,
        derived,
        report,
        parent_sha=recipe.PARENT_SHA,
        derived_sha=a.checkpoint_sha256,
    )
    canonical_before = gate._canonical_metadata(a.manifest)
    data, groups = gate.data_identity(a.manifest, derived)
    if data["eligible_photo_counts"]["val"] != 770:
        raise ValueError("Require exact770 VAL")
    source = gate.evaluation_source_identity()
    helper_sha = recipe.sha(Path(__file__).read_bytes())
    config = inference_config(derived)
    protocol = gate.inference_protocol(derived, threshold=0.5)
    cache = VerifiedCache(
        a.cache,
        a.manifest,
        expected_cache_fingerprint=recipe.CACHE,
        expected_canonical_fingerprint=recipe.CONTENT,
        allow_canonical_fallback=False,
    )
    cache.require_coverage([r for rows in groups["val"].values() for r in rows])
    started = datetime.now(UTC).isoformat()
    metrics = evaluate_unified_model(
        model,
        a.manifest,
        split="val",
        device=device,
        config=config,
        threshold=0.5,
        derived_cache=cache,
    )
    if metrics["photos"] != 770 or {p["photo_id"] for p in metrics["per_photo"]} != set(
        groups["val"]
    ):
        raise ValueError("Fresh research VAL photo inventory differs")
    gate._finish_canonical_read(a.manifest, recipe.CONTENT, canonical_before)
    if (
        gate.evaluation_source_identity() != source
        or recipe.source_digest(a.source_root) != recipe.EXECUTION
        or any(Path(getattr(a, k)).read_bytes() != v for k, v in blobs.items())
        or recipe.sha(Path(__file__).read_bytes()) != helper_sha
    ):
        raise ValueError("Source/input changed during research VAL")
    result = {
        "schema_version": 1,
        "kind": "bn_recalibration_research_validation",
        "split": "val",
        "metrics": metrics,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started,
        "data": data,
        "checkpoint_sha256": a.checkpoint_sha256,
        "parent_checkpoint_sha256": recipe.PARENT_SHA,
        "calibration_sha256": a.calibration_sha256,
        "derivation": derivation,
        "configuration": derived["config"],
        "calibration_recipe": report["recipe"],
        "source": source,
        "research_helper_sha256": helper_sha,
        "runtime": evaluation_runtime(device),
        "inference_protocol": protocol,
        "selection_authorized": False,
        "test_evaluation_enabled": False,
        "training_selection_score": None,
        "qualification": "Fresh measured derived BN VAL; no inherited validation score, no standard selection receipt or clinical claim",
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
