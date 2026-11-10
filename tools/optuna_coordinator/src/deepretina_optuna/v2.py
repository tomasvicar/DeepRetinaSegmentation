"""Finite corpus-v2 control/ablations; no random extra trials or ML imports."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

KIND = "finite_v2_controlled"
NAMES = (
    "control",
    "vessel_loss2",
    "vessel_pos2",
    "dice_balance",
    "geometry256b8",
    "basic",
    "strong",
)
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
POLICY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
CONTROL = {
    "architecture": "compact_unet",
    "legacy_initialization": None,
    "allow_historical_exposure": False,
    "preprocessing_profile": "legacy25_raw",
    "pixels_per_degree": 25.0,
    "epochs": 60,
    "epoch_samples": 8192,
    "coverage_first": True,
    "batch_size": 4,
    "patch_size": 384,
    "base_channels": 12,
    "depth": 4,
    "norm": "batch",
    "lr": 0.0003,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "min_lr_ratio": 0.05,
    "seed": 42,
    "workers": 2,
    "device": "cuda:0",
    "sampling": "dataset_task_balanced",
    "train_view": "patch",
    "view_size": 512,
    "foreground_probability": 0.5,
    "augmentation": "mild",
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "av_weight": 1.0,
    "vessel_loss_weight": 1.0,
    "vessel_pos_weight": 1.0,
    "tile_size": 512,
    "overlap": 64,
    "require_all_heads": True,
    "selection_metric": "macro_dataset_mean_dice",
    "purpose": "full_data",
    "max_steps": None,
}


def is_finite(protocol):
    return protocol.get("study_kind") == KIND


def configurations():
    result = {name: copy.deepcopy(CONTROL) for name in NAMES}
    result["vessel_loss2"]["vessel_loss_weight"] = 2.0
    result["vessel_pos2"]["vessel_pos_weight"] = 2.0
    result["dice_balance"].update(bce_weight=0.5, dice_weight=1.5)
    result["geometry256b8"].update(patch_size=256, batch_size=8)
    result["basic"]["augmentation"] = "basic"
    result["strong"]["augmentation"] = "strong"
    return result


def validate_protocol(protocol):
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    require(
        protocol.get("schema_version") == 2 and protocol.get("max_new_configs") == 7,
        "Finite v2 requires schema2 and exactly7 candidates",
    )
    require(
        protocol.get("control_worker_id") == "ada",
        "Finite v2 control is assigned to Ada",
    )
    require(
        protocol.get("search_space")
        == {"candidate": {"kind": "categorical", "choices": list(NAMES)}},
        "Finite candidate choice inventory differs",
    )
    require(
        json.dumps(protocol.get("candidate_configs"), sort_keys=True, allow_nan=False)
        == json.dumps(configurations(), sort_keys=True, allow_nan=False)
        and json.dumps(protocol.get("config_template"), sort_keys=True, allow_nan=False)
        == json.dumps(CONTROL, sort_keys=True, allow_nan=False),
        "Finite v2 controlled configurations differ",
    )
    require(
        "manual_baseline" not in protocol,
        "V2 must not import the v1 baseline as a same-data control",
    )
    plan = protocol["plan_template"]
    require(
        plan["manifest_sha256"] == MANIFEST
        and plan["prepared_content_sha256"] == CONTENT
        and plan.get("geometry_policy_sha256") == POLICY,
        "Finite v2 corpus/geometry pins differ",
    )
    require(
        plan["training_photos"] == 2529
        and plan["validation_photos"] == 770
        and plan["test_photos"] == 856
        and plan["training_annotation_records"] == 5700,
        "Finite v2 corpus support counts differ",
    )
    require(
        plan.get("derived_cache") is None,
        "Initial finite v2 protocol requires canonical inputs, no v1 cache",
    )
    require(
        bool(re.fullmatch("[0-9a-f]{40}", plan["source_commit"]))
        and all(
            re.fullmatch("[0-9a-f]{64}", plan[k])
            for k in ("execution_code_sha256", "training_source_sha256")
        ),
        "Final source pins must be full SHA values",
    )
    require(
        len(plan["datasets"]) == 21 and len(set(plan["datasets"])) == 21,
        "Expected21 v2 source names",
    )
    require(
        all(not p["cache"] for p in protocol["profiles"].values()),
        "No unprepared v2 cache may be implied by a worker profile",
    )


def make_protocol(source_commit, execution_code_sha256, training_source_sha256):
    old = json.loads((Path(__file__).parents[2] / "protocol.json").read_text())
    result = copy.deepcopy(old)
    for key in ("manual_baseline", "seed_candidate", "seed_candidate_profile"):
        result.pop(key, None)
    result.update(
        schema_version=2,
        study_kind=KIND,
        study_id="leipzig-v2-controlled-20261110",
        max_new_configs=7,
        control_worker_id="ada",
        candidate_configs=configurations(),
        config_template=copy.deepcopy(CONTROL),
        search_space={"candidate": {"kind": "categorical", "choices": list(NAMES)}},
        control_policy="Actual trial0 always continues to60 unless failure/deadline; native ASHA rung values registered",
        sampler={
            "name": "TPESampler_fixed_queue_only",
            "seed": 42,
            "constant_liar": True,
            "multivariate": True,
        },
        source_freeze_status="Concrete reviewed-parent source pins; root must regenerate for its final source freeze",
        no_random_extra_trials=True,
        maximum_candidate_epochs=420,
        qualifications=[
            "Seven fixed measured candidates; no imported v1 same-data control or synthetic quality value",
            "All candidates scheduled60 with actual3/9/27/60 ASHA checkpoints; no random extras",
            "Actual control0 always eligible to60 except deadline/failure; noncontrols may be pruned",
            "Meta execution remains conditional on reviewed staging/allocation; protocol submits no jobs",
            "No TEST or historical teacher initialization; repeated VAL selection remains exploratory",
        ],
    )
    plan = result["plan_template"]
    plan.update(
        source_commit=source_commit,
        execution_code_sha256=execution_code_sha256,
        training_source_sha256=training_source_sha256,
        manifest_sha256=MANIFEST,
        prepared_content_sha256=CONTENT,
        geometry_policy_sha256=POLICY,
        derived_cache=None,
        training_photos=2529,
        training_annotation_records=5700,
        validation_photos=770,
        test_photos=856,
    )
    plan["datasets"] = sorted(
        [*plan["datasets"], "drions", "gamma", "idrid_disc", "biomisa"]
    )
    for profile in result["profiles"].values():
        profile["cache"] = False
    validate_protocol(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source-commit",
        "execution-code-sha256",
        "training-source-sha256",
        "output",
    ):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    value = make_protocol(
        args.source_commit, args.execution_code_sha256, args.training_source_sha256
    )
    with Path(args.output).open("x") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
