"""Proposal only: native TPE on completed E27 observations, never a parent-state migration."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CORE = "8777d6cca60f7ced08798ad2af7fabf626421d05"
CONTRACT = "332a8e4f0fc82d096d1a4fb61baf7d0bd36ca5e3d2d4247b324b02b7363805c2"
PARENT = "053f05c094d052f98069358effc56522b743b8a25c85c3afcb9a6a75aef79200"
DECISION = datetime(2026, 11, 11, 10, 35, tzinfo=UTC)
SAMPLER = {
    "seed": 42,
    "constant_liar": True,
    "multivariate": True,
    "n_startup_trials": 4,
    "n_ei_candidates": 24,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(entry, parent):
    path = parent / entry["path"]
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 * 1024**2,
        "Require small original JSON",
    )
    raw = path.read_bytes()
    require(digest(raw) == entry["sha256"], "Input byte SHA differs")
    return json.loads(raw)


def protocol(api):
    p = api.v3.make_protocol(
        "2026-11-11T14:30:00Z", "meta01", {"meta": {"cache": True, "workers": 8}}
    )
    p["study_id"] = "leipzig-v3-adaptive64-20261111"
    api.v3.validate_protocol(p)
    require(api.v3.digest(p) == PARENT, "Native parent factory differs")
    return p


def qualify(entry, parent, api, p):
    task, result = checked(entry["task"], parent), checked(entry["receipt"], parent)
    require(
        task.get("schema_version") == result.get("schema_version") == 1
        and task.get("study_kind") == result.get("study_kind") == api.v3.KIND,
        "Original typed V3 task and receipt required",
    )
    require(
        result["status"] == "COMPLETE"
        and task["target_epoch"] == result["actual_epoch"] == result["target_epoch"] == 27,
        "Actual completed E27 receipt required",
    )
    require(
        task["previous_epoch"] == 9 and task["resume"] is True,
        "Original 60-scheduler E27 lineage required",
    )
    require(
        entry["parent_status"] in {"RUNNING", "COMPLETE", "PRUNED", "FAIL"},
        "Record actual parent status separately",
    )
    for key in (
        "study_id",
        "trial_number",
        "trial",
        "task_id",
        "lease",
        "worker_id",
        "plan_sha256",
        "config_sha256",
        "protocol_sha256",
    ):
        require(task[key] == result[key], "Original task/receipt identity differs: " + key)
    require(
        task["protocol_sha256"] == PARENT and task["study_id"] == p["study_id"],
        "Wrong original protocol",
    )
    require(
        type(task["trial_number"]) is int
        and 0 <= task["trial_number"] < 24
        and task["trial"] == f"trial{task['trial_number']:04d}",
        "Wrong original parent trial",
    )
    params = task["suggested_params"]
    config = api.config_for(params, "meta", p)
    expected = dict(
        api.v3.PLAN,
        manifest="/data/manifest.json",
        screen_epochs=3,
        total_epochs=60,
        candidates={task["trial"]: config},
    )
    require(
        task["plan"] == expected
        and api.v3.digest(expected) == task["plan_sha256"]
        and api.v3.digest(config) == task["config_sha256"],
        "Exact source72 factory plan required",
    )
    require(
        task["is_control"] is (task["trial_number"] == 0)
        and (task["trial_number"] != 0 or params == p["seed_candidate"]),
        "Original protected control differs",
    )
    q = result["checkpoint_qa"]
    bindings = {
        "source_commit": api.v3.SOURCE,
        "execution_code_sha256": api.v3.EXECUTION,
        "training_source_sha256": api.v3.TRAINING,
        "manifest_sha256": api.v3.PLAN["manifest_sha256"],
        "canonical_sha256": api.v3.PLAN["prepared_content_sha256"],
        "cache_sha256": api.v3.CACHE,
        "geometry_policy_sha256": api.v3.PLAN["geometry_policy_sha256"],
    }
    for key, value in bindings.items():
        require(q[key] == value, "Wrong observation binding: " + key)
    require(
        task["source_commit"] == api.v3.SOURCE
        and task["execution_code_sha256"] == api.v3.EXECUTION,
        "Wrong task source",
    )
    for key, value in {
        "actual_epoch": 27,
        "steps": 27 * 2048,
        "delivered_samples": 27 * 8192,
        "scheduler_T_max": 60,
        "scheduler_last_epoch": 27,
        "eligible_photos": 2529,
        "seen_photos": 2529,
    }.items():
        require(type(q[key]) is int and q[key] == value, "Wrong fidelity/budget: " + key)
    require(
        q["status"] == "PASS"
        and q["finite_weights"] is True
        and q["finite_optimizer"] is True
        and q["completed_budget"] is False
        and q["minimum_photo_visits"] >= 27,
        "Incomplete E27 execution proof",
    )
    for key in ("plan_sha256", "config_sha256", "worker_id", "profile_sha256"):
        require(q[key] == task[key], "QA/task binding differs: " + key)
    proof = q["independent_milestone"]
    require(
        q.get("host_independent_milestone_sha256_verified") is True
        and proof["status"] == "independent_copy_sha_verified_readonly"
        and proof["hardlinks_used"] is False,
        "Original host-verified independent milestone required",
    )
    expected_destination = (
        Path(task["output"]).parent
        / ".pool-milestones"
        / (Path(task["output"]).name + "-e027-" + task["lease"])
    )
    require(
        proof["destination"] == str(expected_destination), "Original milestone destination differs"
    )
    require(
        proof["sha256"]["best.pt"] == q["best_checkpoint_sha256"], "Original BEST/LAST roles differ"
    )
    require(
        proof["source_run"] == task["output"]
        and proof["sha256"]["last.pt"] == q["checkpoint_sha256"],
        "Milestone LAST SHA differs",
    )
    require(
        digest(q["csv_text"].encode()) == q["csv_sha256"] == proof["sha256"]["metrics.csv"],
        "Original CSV differs",
    )
    values, high, epoch = api.csv_scores(q["csv_text"], 27, config)
    require(
        q["last_validation_metrics"]["macro_dataset_mean_dice"] == values[-1],
        "Actual LAST macro differs from final CSV row",
    )
    value = api.v3.validate_rung(q, task, high, epoch)
    require(result["metric"] == value, "Outer objective differs")
    worker = result["run_result"]["worker"]
    require(
        worker["host"]
        and worker["pbs_job_id"]
        and worker["gpu_uuid"] == result["gpu_uuid"] == q["gpu_uuid"],
        "Node/GPU provenance missing",
    )
    return {
        "params": params,
        "value": value,
        "parent_trial": task["trial_number"],
        "parent_study_id": task["study_id"],
        "parent_status": entry["parent_status"],
        "task_sha256": entry["task"]["sha256"],
        "receipt_sha256": entry["receipt"]["sha256"],
        "checkpoint_sha256": q["checkpoint_sha256"],
        "source_commit": api.v3.SOURCE,
        "training_source_sha256": api.v3.TRAINING,
        "execution_code_sha256": api.v3.EXECUTION,
        "parent_protocol_sha256": PARENT,
        "profile_sha256": task["profile_sha256"],
        "config_sha256": task["config_sha256"],
        "plan_sha256": task["plan_sha256"],
        "node": worker["host"],
        "gpu_uuid": worker["gpu_uuid"],
        "pbs_job_id": worker["pbs_job_id"],
        "parent_horizon": 60,
        "fidelity_epoch": 27,
        "steps": q["steps"],
        "delivered_samples": q["delivered_samples"],
    }


def propose(path, expected_sha, output, api, *, synthetic=False, now=None):
    raw = path.read_bytes()
    require(digest(raw) == expected_sha, "Input manifest SHA differs")
    manifest = json.loads(raw)
    now = now if synthetic else datetime.now(UTC)
    require(now is not None and now.tzinfo is not None, "UTC decision time required")
    require(
        manifest["schema_version"] == 1
        and manifest["inventory_complete"] is True
        and manifest["available_rung27_count"] == len(manifest["observations"]),
        "Complete decision-time inventory required",
    )
    if synthetic:
        require(
            manifest.get("kind") == "synthetic_fixed27_test"
            and output.resolve().is_relative_to(Path("/tmp")),
            "Synthetic tests must use temporary outputs",
        )
    else:
        require(manifest.get("kind") == "fixed27_proposal_inputs", "Typed proposal inputs required")
    require(
        type(manifest["own_ada_free"]) is bool, "Explicit Ada availability observation required"
    )
    observed = datetime.fromisoformat(manifest["observed_at_utc"])
    require(observed.tzinfo is not None, "UTC inventory time required")
    require(
        -30 <= (now - observed).total_seconds() <= 300,
        "Refresh ALL-available inventory at decision time",
    )
    require(len(manifest["observations"]) <= 24, "Unexpected parent trial inventory size")
    p = protocol(api)
    observations = [qualify(e, path.parent, api, p) for e in manifest["observations"]]
    require(
        len({o["parent_trial"] for o in observations}) == len(observations),
        "Repeated parent trial entry",
    )
    unique = {}
    for obs in sorted(observations, key=lambda o: o["parent_trial"]):
        unique.setdefault(api.v3.digest(obs["params"]), obs)
    selected = sorted(unique.values(), key=lambda o: o["parent_trial"])
    require(
        len(selected) >= 4, "At least4 unique qualified configurations required; no study created"
    )
    require(
        synthetic or (manifest["own_ada_free"] is True and now >= DECISION),
        "Decision gate: free Ada and >=4 unique observations, not before10:35UTC; no study created",
    )
    require(not output.exists(), "Use a fresh output, never an active database")
    import optuna

    require(optuna.__version__ == "5.0.0", "Reviewed Optuna5.0.0 required")
    api.local_filesystem(output.parent)  # Before mkdir/SQLite; never NFS/ownCloud.
    distributions = {
        k: (
            optuna.distributions.CategoricalDistribution(s["choices"])
            if s["kind"] == "categorical"
            else optuna.distributions.FloatDistribution(s["low"], s["high"], log=s["log"])
        )
        for k, s in api.v3.SEARCH.items()
    }
    output.mkdir()
    (output / "input-manifest.json").write_bytes(raw)
    sampler = optuna.samplers.TPESampler(**SAMPLER)
    study = optuna.create_study(
        study_name="fixed27-proposal-synthetic"
        if synthetic
        else "leipzig-fixed27-proposal-20261111",
        storage="sqlite:///" + str((output / "proposal.sqlite3").resolve()),
        direction="maximize",
        sampler=sampler,
    )
    for obs in selected:
        attrs = {k: v for k, v in obs.items() if k not in ("params", "value")}
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=obs["value"],
                params=obs["params"],
                distributions=distributions,
                user_attrs=attrs,
            )
        )
    before = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    trial = study.ask(
        fixed_distributions=distributions
    )  # One native TPE ask, no manual alterations.
    trial.set_user_attr("proposal_only_no_evaluation", True)
    config = api.config_for(trial.params, "meta", p)
    state = pickle.dumps(sampler, protocol=5)
    result = {
        "kind": "synthetic_fixed27_test_result" if synthetic else "fixed27_tpe_proposal_only",
        "native_proposal_trial_number": trial.number,
        "proposal_study_name": study.study_name,
        "proposed_params": trial.params,
        "full_future60_config": config,
        "full_future60_plan": dict(
            api.v3.PLAN,
            manifest="/data/manifest.json",
            screen_epochs=3,
            total_epochs=60,
            candidates={"fixed27_tpe_proposal": config},
        ),
        "helper_sha256": digest(Path(__file__).read_bytes()),
        "search_space_sha256": api.v3.digest(api.v3.SEARCH),
        "future_training_deadline_utc": "2026-11-11T18:30:00Z",
        "observations": observations,
        "imported_unique_observations": selected,
        "imported_unique_count": before,
        "duplicate_count": len(observations) - before,
        "sampler_kwargs": SAMPLER,
        "optuna_version": optuna.__version__,
        "sampler_state_sha256": digest(state),
        "input_manifest_sha256": expected_sha,
        "parent_protocol_sha256": PARENT,
        "public_core_commit": CORE,
        "native_sampler": "TPESampler_post_startup",
        "new_trial_state": "RUNNING_UNEVALUATED_PROPOSAL",
        "own_study_fidelity": 27,
        "actual_decision_utc": now.isoformat(),
        "parent_states_modified": False,
        "no_model_or_GT_read": True,
        "no_job_or_training_side_effect": True,
        "final_candidate_requires": "fresh full60/all5700/freshVAL/final freeze/TEST; E27proxy is not final evidence",
    }
    (output / "proposal.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "sampler-state.pkl").write_bytes(state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-manifest", "input-sha256", "output", "core-source"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path(args.core_source).resolve()
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    require(
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, env=git_env
        ).strip()
        == CORE,
        "Require frozen8777 core",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True, env=git_env
        ).strip(),
        "Frozen core is dirty",
    )
    contract = source.parent
    h = hashlib.sha256()
    for f in sorted(
        [*contract.glob("src/**/*.py"), *contract.glob("src/**/*.json"), contract / "protocol.json"]
    ):
        h.update(f.relative_to(contract).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    require(h.hexdigest() == CONTRACT, "Frozen core byte fingerprint differs")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from deepretina_optuna import core as api

    require(
        Path(api.__file__).resolve() == source / "deepretina_optuna/core.py", "Wrong imported core"
    )
    propose(Path(args.input_manifest), args.input_sha256, Path(args.output), api)


if __name__ == "__main__":
    main()
