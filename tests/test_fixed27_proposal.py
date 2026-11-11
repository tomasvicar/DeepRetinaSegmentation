"""Synthetic CPU metadata only; native Optuna and native8777 validation, no NN/GT."""

import copy
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import optuna
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("fixed27", ROOT / "scripts/propose_fixed27_tpe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def api(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    package.mkdir()
    for name in ["__init__.py", "core.py", "v2.py", "v3.py", "v3_reference.json"]:
        raw = subprocess.check_output(
            ["git", "show", f"{m.CORE}:tools/optuna_coordinator/src/deepretina_optuna/{name}"],
            cwd=ROOT,
        )
        (package / name).write_bytes(raw)
    prefix = "fixed27_fixture_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        prefix, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, prefix, mod)
    spec.loader.exec_module(mod)
    return importlib.import_module(prefix + ".core")


def metrics(api, tp=6):
    def counts(tp, fp, fn, n):
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "valid_pixels": n,
            "dice": 2 * tp / (2 * tp + fp + fn),
        }

    channels = {
        "vessel": counts(tp, 1, 8 - tp, 16),
        "artery": counts(2, 0, 2, 8),
        "vein": counts(2, 0, 2, 8),
        "disc": counts(4, 0, 1, 16),
        "cup": counts(2, 0, 1, 16),
    }
    photos = [
        {
            "photo_id": f"synthetic{i:04}",
            "record_ids": [f"row{i}"],
            "dataset": "synthetic",
            "channel_datasets": {c: "synthetic" for c in channels},
            "channels": copy.deepcopy(channels),
            "av_conditional_counts": {
                "correct": 6,
                "scored_pixels": 8,
                "eligible_pixels": 8,
                "end_to_end_correct": 4,
            },
        }
        for i in range(770)
    ]
    pooled = {
        c: {**{k: v[k] * 770 for k in ["tp", "fp", "fn", "valid_pixels"]}, "dice": v["dice"]}
        for c, v in channels.items()
    }
    return {
        "split": "val",
        "photos": 770,
        "threshold": 0.5,
        "family": "unified",
        "purpose": "full_data",
        "per_photo": photos,
        "channels": pooled,
        "av_conditional_accuracy": 0.75,
        "macro_dataset_mean_dice": sum(v["dice"] for v in channels.values()) / 5,
    }


@pytest.fixture
def case(tmp_path, monkeypatch, api):
    base = metrics(api)
    monkeypatch.setitem(
        api.v3.REFERENCE,
        "validation_photo_ids_sha256",
        api.v3.digest(sorted(p["photo_id"] for p in base["per_photo"])),
    )
    monkeypatch.setitem(
        api.v3.REFERENCE, "validation_support_sha256", api.v3.digest(api.v3.support_inventory(base))
    )
    p = api.v3.make_protocol(
        "2026-11-11T14:30:00Z", "meta01", {"meta": {"cache": True, "workers": 8}}
    )
    p["study_id"] = "leipzig-v3-adaptive64-20261111"
    monkeypatch.setattr(
        m, "PARENT", api.v3.digest(p)
    )  # Explicit synthetic golden inventory, never a real observation.
    now = datetime(2026, 11, 11, 11, tzinfo=UTC)
    manifest = {
        "schema_version": 1,
        "kind": "synthetic_fixed27_test",
        "inventory_complete": True,
        "observed_at_utc": now.isoformat(),
        "own_ada_free": True,
        "available_rung27_count": 0,
        "observations": [],
    }

    def write(number, lr=None, duplicate=False):
        params = dict(p["seed_candidate"])
        params["lr"] = float(lr if lr is not None else 0.0003 + number * 0.00005)
        if number and not duplicate:
            params["base_channels"] = 16 if number % 2 else 12
        config = api.config_for(params, "meta", p)
        plan = dict(
            api.v3.PLAN,
            manifest="/data/manifest.json",
            screen_epochs=3,
            total_epochs=60,
            candidates={f"trial{number:04}": config},
        )
        task = {
            "schema_version": 1,
            "study_kind": api.v3.KIND,
            "target_epoch": 27,
            "previous_epoch": 9,
            "resume": True,
            "trial_number": number,
            "trial": f"trial{number:04}",
            "task_id": f"trial{number:04}-e027",
            "lease": f"synthetic{number}-e027",
            "worker_id": "meta01",
            "profile_sha256": "1" * 64,
            "plan": plan,
            "plan_sha256": api.v3.digest(plan),
            "config_sha256": api.v3.digest(config),
            "protocol_sha256": m.PARENT,
            "source_commit": api.v3.SOURCE,
            "execution_code_sha256": api.v3.EXECUTION,
            "study_id": p["study_id"],
            "suggested_params": params,
            "is_control": number == 0,
            "output": f"/synthetic/trial{number:04}",
        }
        stats = metrics(api, 4 + min(number, 3))
        macro = stats["macro_dataset_mean_dice"]
        csv = "epoch,steps,val_selection_score,val_macro_dataset_mean_dice\n" + "".join(
            f"{i},{i * 2048},{macro},{macro}\n" for i in range(1, 28)
        )
        csvsha = m.digest(csv.encode())
        sha = f"{number + 10:064x}"
        q = {
            "kind": "v3_rung_evidence",
            "status": "PASS",
            "actual_epoch": 27,
            "steps": 55296,
            "delivered_samples": 221184,
            "scheduler_T_max": 60,
            "scheduler_last_epoch": 27,
            "eligible_photos": 2529,
            "seen_photos": 2529,
            "minimum_photo_visits": 27,
            "finite_weights": True,
            "finite_optimizer": True,
            "completed_budget": False,
            "source_commit": api.v3.SOURCE,
            "execution_code_sha256": api.v3.EXECUTION,
            "training_source_sha256": api.v3.TRAINING,
            "manifest_sha256": api.v3.PLAN["manifest_sha256"],
            "canonical_sha256": api.v3.PLAN["prepared_content_sha256"],
            "cache_sha256": api.v3.CACHE,
            "geometry_policy_sha256": api.v3.PLAN["geometry_policy_sha256"],
            "plan_sha256": task["plan_sha256"],
            "config_sha256": task["config_sha256"],
            "worker_id": task["worker_id"],
            "profile_sha256": task["profile_sha256"],
            "gpu_uuid": "SYNTHETIC-GPU",
            "checkpoint_sha256": sha,
            "last_checkpoint_sha256": sha,
            "best_checkpoint_sha256": "f" * 64,
            "last_validation_metrics": stats,
            "last_validation_metrics_sha256": api.v3.digest(stats),
            "macro_best_score": macro,
            "macro_best_epoch": 1,
            "best_epoch": 1,
            "csv_text": csv,
            "csv_sha256": csvsha,
            "host_independent_milestone_sha256_verified": True,
            "independent_milestone": {
                "status": "independent_copy_sha_verified_readonly",
                "source_run": task["output"],
                "hardlinks_used": False,
                "destination": str(
                    Path(task["output"]).parent
                    / ".pool-milestones"
                    / (Path(task["output"]).name + "-e027-" + task["lease"])
                ),
                "sha256": {"last.pt": sha, "best.pt": "f" * 64, "metrics.csv": csvsha},
            },
        }
        q["metric"] = api.v3.objective_score(stats)
        result = {
            k: task[k]
            for k in [
                "study_id",
                "schema_version",
                "study_kind",
                "trial_number",
                "trial",
                "task_id",
                "lease",
                "worker_id",
                "plan_sha256",
                "config_sha256",
                "protocol_sha256",
                "target_epoch",
            ]
        }
        result.update(
            status="COMPLETE",
            actual_epoch=27,
            checkpoint_qa=q,
            metric=q["metric"],
            gpu_uuid=q["gpu_uuid"],
            run_result={
                "worker": {
                    "host": "synthetic-cpu-fixture",
                    "pbs_job_id": "synthetic",
                    "gpu_uuid": q["gpu_uuid"],
                }
            },
        )
        refs = {}
        for key, value in [("task", task), ("receipt", result)]:
            path = tmp_path / f"{key}-{number}.json"
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            refs[key] = {"path": path.name, "sha256": m.digest(raw)}
        manifest["observations"].append(
            dict(**refs, parent_status="RUNNING" if number == 0 else "PRUNED")
        )
        manifest["available_rung27_count"] = len(manifest["observations"])

    for number in range(4):
        write(number)

    def save():
        path = tmp_path / "inputs.json"
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        return path, m.digest(raw)

    return api, manifest, write, save, now


def test_native_fixed27_import_and_one_poststartup_tpe_ask(case, tmp_path, monkeypatch):
    api, _manifest, _, save, now = case
    path, sha = save()
    before = {p.name: m.digest(p.read_bytes()) for p in tmp_path.glob("*.json")}
    monkeypatch.setattr(
        optuna.samplers.RandomSampler,
        "sample_independent",
        lambda *a, **k: pytest.fail("Random startup fallback instead of native TPE"),
    )
    result = m.propose(path, sha, tmp_path / "proposal", api, synthetic=True, now=now)
    assert (
        result["kind"] == "synthetic_fixed27_test_result" and result["imported_unique_count"] == 4
    )
    assert result["sampler_kwargs"]["n_startup_trials"] == 4 and set(
        result["proposed_params"]
    ) == set(api.v3.SEARCH)
    assert (
        result["full_future60_config"]["epochs"] == 60
        and result["full_future60_config"]["workers"] == 8
        and result["full_future60_config"]["mixed_precision"] == "bf16"
    )
    study = optuna.load_study(
        study_name="fixed27-proposal-synthetic",
        storage="sqlite:///" + str(tmp_path / "proposal/proposal.sqlite3"),
    )
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE] * 4 + [
        optuna.trial.TrialState.RUNNING
    ]
    assert (
        study.trials[0].user_attrs["parent_status"] == "RUNNING"
        and study.trials[0].user_attrs["fidelity_epoch"] == 27
    )
    assert study.trials[0].value == api.v3.objective_score(
        json.loads((tmp_path / "receipt-0.json").read_text())["checkpoint_qa"][
            "last_validation_metrics"
        ]
    )
    assert study.trials[-1].distributions["base_channels"].choices == (12, 16)
    assert all(m.digest((tmp_path / name).read_bytes()) == value for name, value in before.items())


def test_duplicate_config_uses_earliest_parent_and_retains_all_origins(case, tmp_path):
    api, _, write, save, now = case
    write(7, lr=0.0003, duplicate=True)
    path, sha = save()
    result = m.propose(path, sha, tmp_path / "dedup", api, synthetic=True, now=now)
    assert (
        result["duplicate_count"] == 1
        and result["imported_unique_count"] == 4
        and len(result["observations"]) == 5
    )
    assert [o["parent_trial"] for o in result["imported_unique_observations"]] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "fault", ["fidelity", "source", "budget", "host", "macro_as_objective", "input_sha"]
)
def test_reject_bad_original_observation_before_sqlite(case, tmp_path, fault):
    api, manifest, _, save, now = case
    entry = manifest["observations"][0]
    path = tmp_path / entry["receipt"]["path"]
    r = json.loads(path.read_text())
    q = r["checkpoint_qa"]
    if fault == "fidelity":
        q["actual_epoch"] = 9
    if fault == "source":
        q["source_commit"] = "wrong"
    if fault == "budget":
        q["scheduler_T_max"] = 27
    if fault == "host":
        q["host_independent_milestone_sha256_verified"] = False
    if fault == "macro_as_objective":
        q["metric"] = q["macro_best_score"]
        r["metric"] = q["metric"]
    raw = json.dumps(r).encode()
    path.write_bytes(raw)
    if fault != "input_sha":
        entry["receipt"]["sha256"] = m.digest(raw)
    else:
        path.write_bytes(raw + b" ")
    path, sha = save()
    with pytest.raises(ValueError):
        m.propose(path, sha, tmp_path / "bad", api, synthetic=True, now=now)
    assert not (tmp_path / "bad").exists()


def test_under_four_does_not_create_study(case, tmp_path, monkeypatch):
    api, manifest, _, save, now = case
    manifest["observations"].pop()
    manifest["available_rung27_count"] = 3
    path, sha = save()
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: pytest.fail("No SQLite with<4"))
    with pytest.raises(ValueError, match="At least4"):
        m.propose(path, sha, tmp_path / "early", api, synthetic=True, now=now)
    assert not (tmp_path / "early").exists()


def test_real_time_gate_blocks_four_before1035_without_creating_sqlite(case, tmp_path, monkeypatch):
    api, manifest, _, save, now = case
    manifest["kind"] = "fixed27_proposal_inputs"
    now = datetime(2026, 11, 11, 10, tzinfo=UTC)
    manifest["observed_at_utc"] = now.isoformat()
    path, sha = save()

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return now

    monkeypatch.setattr(m, "datetime", Clock)
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: pytest.fail("No early SQLite"))
    with pytest.raises(ValueError, match="Decision gate"):
        m.propose(path, sha, tmp_path / "early", api)
    assert not (tmp_path / "early").exists()


def test_busy_ada_blocks_even_after_earliest_time(case, tmp_path, monkeypatch):
    api, manifest, _, save, now = case
    manifest["kind"] = "fixed27_proposal_inputs"
    manifest["own_ada_free"] = False
    path, sha = save()

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return now

    monkeypatch.setattr(m, "datetime", Clock)
    monkeypatch.setattr(
        optuna, "create_study", lambda **kwargs: pytest.fail("No study on busy Ada")
    )
    with pytest.raises(ValueError, match="Decision gate"):
        m.propose(path, sha, tmp_path / "busy", api)
    assert not (tmp_path / "busy").exists()
