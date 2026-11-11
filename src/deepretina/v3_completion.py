"""Genuine V3 pool final60 inputs; never relabel a pool receipt as standalone."""

import math
from pathlib import Path

from . import selection as g
from ._frozen_v3 import v3

CORE = "8777d6cca60f7ced08798ad2af7fabf626421d05"
CORE_FILES = {
    "v2.py": "a0e8df7e9bc86038c549523a0c84211e6892299b73246e0f12490f22d7110d6d",
    "v3.py": "234f64ba7871f17c71f07d179e43ce1a0d7427020a800acdcdddc3bfa2480ced",
    "v3_reference.json": "fc7284a0de7dd55251bcde79b128532d9221cecdbe98748dc2ae2f35866ac558",
}


def verify_original_pool(receipt, index, best, last, snapshots, *, return_snapshots=False):
    """Validate original immutable records plus actual independent milestone bytes.

    `snapshots` contains already checked BEST/LAST/CSV/summary raw bytes. Index
    paths locate preserved copies; their hashes and the external index pin bind
    the exact originals, without publishing private operational paths.
    """
    root = Path(v3.__file__).parent
    g.require(
        all(g.digest((root / name).read_bytes()) == sha for name, sha in CORE_FILES.items()),
        "Native8777 implementation/reference bytes changed",
    )
    g.require(
        index.get("schema_version") == 1
        and index.get("kind") == "original_v3_pool_evidence_index"
        and index.get("core_source_commit") == CORE,
        "Require explicit native8777 V3 input index",
    )
    g.require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("actual_epoch") == receipt.get("target_epoch") == 60,
        "Original V3 pool receipt must actually complete60; partial rung is ineligible",
    )
    files = index["files"]
    g.exact(files, ("task", "protocol", "envelope", "profile"), "V3 original input index")
    loaded, originals = {}, []
    for name, spec in files.items():
        raw, value = g.json_snapshot(spec["path"])
        g.require(g.digest(raw) == spec["sha256"], "Original V3 input bytes differ: " + name)
        loaded[name] = value
        originals.append((Path(spec["path"]), raw))
    task, protocol, envelope, profile = (
        loaded[k] for k in ("task", "protocol", "envelope", "profile")
    )
    v3.validate_protocol(protocol)
    digest = v3.digest  # Native compact sorted JSON + newline convention.
    g.require(
        envelope["v3_protocol"] == protocol
        and envelope["protocol_sha256"] == task["protocol_sha256"] == digest(protocol)
        and envelope["execution_authorized"] is True,
        "Original V3 protocol/envelope pins differ",
    )
    g.require(
        task["study_kind"] == receipt["study_kind"] == v3.KIND
        and task["study_id"] == receipt["study_id"] == protocol["study_id"]
        and task["target_epoch"] == 60
        and task["resume"] is True,
        "Require genuine final V3 pool task and study",
    )
    number = task["trial_number"]
    g.require(
        type(number) is int
        and 0 <= number <= protocol["max_new_configs"]
        and task["trial"] == f"trial{number:04d}"
        and task["task_id"] == f"trial{number:04d}-e060"
        and isinstance(task["lease"], str)
        and bool(task["lease"])
        and task["previous_epoch"] == 27
        and task["is_control"] is (number == 0)
        and task["candidate_role"] == ("seed_control" if number == 0 else "adaptive_candidate"),
        "Native V3 trial/rung/control identity differs",
    )
    if number == 0:
        g.require(
            task["worker_id"] == protocol["control_worker_id"]
            and task["suggested_params"] == protocol["seed_candidate"],
            "Original native V3 seed control differs",
        )
    for key in (
        "worker_id",
        "trial",
        "trial_number",
        "task_id",
        "lease",
        "profile",
        "profile_sha256",
        "gpu_uuid",
        "plan_sha256",
        "config_sha256",
        "protocol_sha256",
        "search_space_sha256",
    ):
        g.require(task[key] == receipt[key], "Original V3 task/receipt differs: " + key)
    g.require(
        digest(profile) == task["profile_sha256"]
        and profile["worker_id"] == task["worker_id"]
        and profile["source_commit"] == v3.SOURCE
        and profile["execution_code_sha256"] == v3.EXECUTION
        and profile["training_source_sha256"] == v3.TRAINING,
        "Actual original V3 worker profile differs",
    )
    binding = envelope["profiles"][task["worker_id"]]
    g.require(
        binding["profile_sha256"] == task["profile_sha256"]
        and binding["profile"] == task["profile"]
        and task["search_space_sha256"] == envelope["search_space_sha256"]
        and task["deadline_utc"] == protocol["deadline_utc"] == envelope["deadline_utc"],
        "Original worker/search/deadline binding differs",
    )
    plan = task["plan"]
    config = plan["candidates"][task["trial"]]
    g.require(
        digest(plan) == task["plan_sha256"]
        and digest(config) == task["config_sha256"]
        and config == best["config"] == last["config"]
        and task["source_commit"] == plan["source_commit"] == v3.SOURCE
        and task["execution_code_sha256"] == plan["execution_code_sha256"] == v3.EXECUTION
        and plan["training_source_sha256"] == v3.TRAINING,
        "Original V3 source/config/plan differs",
    )
    for key, value in (
        ("source_commit", v3.SOURCE),
        ("execution_code_sha256", v3.EXECUTION),
        ("training_source_sha256", v3.TRAINING),
        ("manifest_sha256", plan["manifest_sha256"]),
        ("canonical_sha256", plan["prepared_content_sha256"]),
        ("geometry_policy_sha256", plan["geometry_policy_sha256"]),
    ):
        g.require(envelope[key] == value, "Original V3 envelope source/data differs: " + key)
    g.require(
        task["canonical_sha256"] == plan["prepared_content_sha256"]
        and task["cache_sha256"] == plan["derived_cache"]["fingerprint"],
        "V3 task actual data/cache differs",
    )
    params = task["suggested_params"]
    g.require(set(params) == set(protocol["search_space"]), "V3 search inventory differs")
    for key, spec in protocol["search_space"].items():
        val = params[key]
        if spec["kind"] == "categorical":
            g.require(
                any(type(val) is type(x) and val == x for x in spec["choices"]),
                "V3 category outside bounds",
            )
        else:
            g.require(
                type(val) is float and math.isfinite(val) and spec["low"] <= val <= spec["high"],
                "V3 float outside bounds",
            )
    expected = {
        **protocol["config_template"],
        **params,
        **profile["config_fixed"],
        "device": "cuda:0",
    }
    expected_plan = {
        **protocol["plan_template"],
        "manifest": "/data/manifest.json",
        "screen_epochs": 3,
        "total_epochs": 60,
        "candidates": {task["trial"]: expected},
    }
    g.require(
        plan == expected_plan and profile["config_fixed"] == {"workers": 8},
        "Exact native V3 recipe differs",
    )
    run = receipt["run_result"]
    g.require(
        type(run.get("returncode")) is int
        and run["returncode"] == 0
        and run.get("test_evaluation_performed") is False
        and run["target_epoch"] == 60,
        "Require original successful real trainer return code; no TEST",
    )
    runtime = run["runtime"]
    g.require(
        runtime["gpu_uuid"] == runtime["expected_gpu_uuid"] == task["gpu_uuid"]
        and runtime["cuda_visible_devices"] == task["gpu_uuid"]
        and runtime["runtime_installs"] is False
        and isinstance(runtime["pbs_job_id"], str)
        and runtime["pbs_job_id"]
        and runtime["intraop_threads"] == profile["runtime"]["intraop_threads"]
        and runtime["interop_threads"] == profile["runtime"]["interop_threads"],
        "Original physical GPU/PBS/runtime differs",
    )
    g.require(
        runtime["environment"] == profile["runtime"]["environment"],
        "Original runtime environment differs",
    )
    worker = run["worker"]
    g.require(
        worker["source_commit"] == v3.SOURCE
        and worker["worker_id"] == task["worker_id"]
        and worker["profile_sha256"] == task["profile_sha256"]
        and worker["plan_sha256"] == task["plan_sha256"]
        and worker["gpu_uuid"] == runtime["gpu_uuid"]
        and worker["pbs_job_id"] == runtime["pbs_job_id"],
        "Original V3 actual worker identity differs",
    )
    qa = receipt["checkpoint_qa"]
    g.require(
        qa["status"] == "PASS"
        and qa["completed_budget"] is True
        and qa["actual_epoch"] == 60
        and qa["finite_weights"] is True
        and qa["finite_optimizer"] is True
        and qa["checkpoint_sha256"] == qa["last_checkpoint_sha256"] == last["checkpoint_sha256"]
        and qa["best_checkpoint_sha256"] == best["checkpoint_sha256"]
        and qa["last_validation_metrics"] == last["validation_metrics"]
        and qa["coverage"] == last["coverage"]
        and qa["csv_sha256"] == g.digest(snapshots["metrics.csv"])
        and qa["csv_text"].encode() == snapshots["metrics.csv"],
        "Native V3 QA differs from actual saved bytes",
    )
    for key, value in (
        ("source_commit", v3.SOURCE),
        ("execution_code_sha256", v3.EXECUTION),
        ("training_source_sha256", v3.TRAINING),
        ("plan_sha256", task["plan_sha256"]),
        ("config_sha256", task["config_sha256"]),
        ("profile_sha256", task["profile_sha256"]),
        ("gpu_uuid", task["gpu_uuid"]),
        ("worker_id", task["worker_id"]),
    ):
        g.require(qa[key] == value, "V3 raw QA pin differs: " + key)
    g.require(
        qa["steps"] == last["steps"]
        and qa["scheduler_T_max"] == qa["scheduler_last_epoch"] == 60
        and qa["delivered_samples"] == 491520
        and qa["eligible_photos"] == qa["seen_photos"] == 2529,
        "Native V3 budget/scheduler headers differ",
    )
    objective = v3.validate_rung(qa, task, best["val_selection_score"], best["epoch"])
    g.require(receipt["metric"] == objective, "Original study LAST objective differs")
    milestone = qa["independent_milestone"]
    g.require(
        qa["host_independent_milestone_sha256_verified"] is True
        and milestone["status"] == "independent_copy_sha_verified_readonly"
        and milestone["hardlinks_used"] is False,
        "Require original host independent milestone proof",
    )
    directory = Path(index["milestone_directory"])
    expected_files = {
        "best.pt",
        "last.pt",
        "metrics.csv",
        "training_summary.json",
        "coverage.json",
        "validation.json",
        "run.json",
        "search_trial.json",
    }
    g.require(
        set(milestone["sha256"]) == expected_files, "Incomplete original V3 milestone inventory"
    )
    sidecars = {}
    for name in sorted(expected_files):
        path = directory / name
        raw = path.read_bytes()
        g.require(
            not path.is_symlink()
            and path.stat().st_nlink == 1
            and path.stat().st_mode & 0o222 == 0
            and g.digest(raw) == milestone["sha256"][name],
            "Actual independent milestone bytes/mode differ: " + name,
        )
        if name in snapshots:
            g.require(
                raw == snapshots[name],
                "Supplied raw checkpoint/sidecar is not original milestone: " + name,
            )
        else:
            sidecars[name] = g.parse_json(raw)
        originals.append((path, raw))
    marker = {
        "plan_sha256": task["plan_sha256"],
        "trial": task["trial"],
        "execution_code_sha256": v3.EXECUTION,
        "manifest_sha256": plan["manifest_sha256"],
    }
    run_metadata = sidecars["run.json"]
    g.require(
        run_metadata["config"] == config
        and run_metadata["training_source_sha256"] == v3.TRAINING
        and run_metadata["manifest_sha256"] == plan["manifest_sha256"]
        and run_metadata["prepared_content_sha256"] == plan["prepared_content_sha256"]
        and run_metadata["derived_cache"] == plan["derived_cache"],
        "Independent run.json differs from training recipe",
    )
    g.require(
        sidecars["search_trial.json"] == marker
        and sidecars["coverage.json"] == last["coverage"]
        and sidecars["validation.json"] == best["validation_metrics"],
        "Milestone runner/coverage/VAL sidecar differs",
    )
    g.require(
        all(path.read_bytes() == raw for path, raw in originals),
        "Original V3 inputs changed during verification",
    )
    proof = {
        "kind": "original_v3_pool_COMPLETE_receipt",
        "original_task_sha256": files["task"]["sha256"],
        "reviewed_protocol_sha256": files["protocol"]["sha256"],
        "envelope_sha256": files["envelope"]["sha256"],
        "profile_sha256": files["profile"]["sha256"],
        "core_source_commit": CORE,
        "milestone_inventory_sha256": g.digest(g.canonical(milestone["sha256"])),
        "study_terminal_objective": objective,
        "study_objective_role": "fixed_terminal60_worst6_ratio",
        "macro_best_epoch": best["epoch"],
        "macro_best_score": best["val_selection_score"],
    }

    return (proof, originals) if return_snapshots else proof
