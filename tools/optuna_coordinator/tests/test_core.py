import copy
import datetime
import hashlib
import json
from pathlib import Path

import pytest
from deepretina_optuna.core import Coordinator, config_for

ROOT = Path(__file__).parents[1]
NOW = datetime.datetime(2026, 11, 10, 22, tzinfo=datetime.UTC)


@pytest.fixture
def protocol():
    return json.loads((ROOT / "protocol.json").read_text())


@pytest.fixture
def profiles():
    return {
        "meta1": {"profile": "meta", "profile_sha256": "a" * 64, "output_root": "/artifacts/night"},
        "ada": {"profile": "ada", "profile_sha256": "b" * 64, "output_root": "/artifacts/local"},
    }


def baseline(c):
    c.import_baseline(
        (ROOT / "tests/fixtures/baseline-epochs-001-060.csv").read_text(),
        (ROOT / "tests/fixtures/baseline-qa.json").read_bytes(),
        c.protocol["manual_baseline"]["qa_sha256"],
    )


def receipt(task, score=0.7):
    epoch = task["target_epoch"]
    cfg = task["plan"]["candidates"][task["trial"]]
    header = "epoch,steps,lr,train_loss,val_mean_dice,val_macro_dataset_mean_dice,val_selection_score,train_seconds,validation_seconds\n"
    csv = header + "".join(
        f"{e},{e * 8192 // cfg['batch_size']},0.0003,0.4,{score},{score},{score},1,1\n"
        for e in range(1, epoch + 1)
    )
    q = {
        "status": "PASS",
        "actual_epoch": epoch,
        "metric": float(score),
        "best_epoch": 1,
        "checkpoint_sha256": "c" * 64,
        "best_checkpoint_sha256": "d" * 64,
        "csv_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "csv_text": csv,
        "finite_weights": True,
        "finite_optimizer": True,
        "scheduler_T_max": 60,
        "scheduler_last_epoch": epoch,
        "steps": epoch * 8192 // cfg["batch_size"],
        "delivered_samples": epoch * 8192,
        "eligible_photos": 2221,
        "seen_photos": 2221,
        "minimum_photo_visits": epoch,
        "completed_budget": epoch == 60,
        "plan_sha256": task["plan_sha256"],
        "config_sha256": task["config_sha256"],
        "source_commit": task["source_commit"],
        "execution_code_sha256": task["execution_code_sha256"],
        "training_source_sha256": task["plan"]["training_source_sha256"],
        "manifest_sha256": task["plan"]["manifest_sha256"],
        "canonical_sha256": task["canonical_sha256"],
        "cache_sha256": task["cache_sha256"],
        "profile_sha256": task["profile_sha256"],
        "gpu_uuid": "GPU-test",
        "worker_id": task["worker_id"],
        "output": task["output"],
    }
    r = {
        k: task[k]
        for k in [
            "schema_version",
            "study_id",
            "task_id",
            "trial_number",
            "trial",
            "lease",
            "worker_id",
            "profile_sha256",
            "plan_sha256",
            "config_sha256",
            "protocol_sha256",
            "search_space_sha256",
            "target_epoch",
        ]
    }
    r.update(
        status="COMPLETE",
        actual_epoch=epoch,
        metric=float(score),
        checkpoint_qa=q,
        gpu_uuid="GPU-test",
        completed_utc=NOW.isoformat(),
    )
    return r


def test_full_ask_report_rungs_and_actual60_only(tmp_path, protocol, profiles):
    with Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        task = c.ask("meta1", "GPU-test")
        assert (
            task["trial_number"] == 1
            and task["plan"]["candidates"][task["trial"]]["av_weight"] == 2
        )
        plan_sha = task["plan_sha256"]
        for epoch in [3, 9, 27]:
            assert task["target_epoch"] == epoch
            r = receipt(task)
            decision = c.report(r)
            assert decision["action"] == "continue"
            assert c.report(r) == decision
            task = decision["next_task"]
            assert task["plan_sha256"] == plan_sha and task["gpu_uuid"] == "GPU-test"
        assert c.status()["trials"][1]["state"] == "RUNNING"
        result = c.report(receipt(task))
        assert result["action"] == "finished"
        assert c.status()["trials"][1]["state"] == "COMPLETE"


def test_missing_fullqa_gpu_or_lease_change_rejected(tmp_path, protocol, profiles):
    with Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        task = c.ask("meta1", "GPU-test")
        for field in ["lease", "gpu_uuid"]:
            r = receipt(task)
            r[field] = "wrong"
            with pytest.raises(ValueError):
                c.report(r)
        r = receipt(task)
        r["checkpoint_qa"]["completed_budget"] = True
        with pytest.raises(ValueError):
            c.report(r)


def test_pruner_and_deadline_do_not_create_facade_complete(tmp_path, protocol, profiles):
    with Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        task = c.ask("meta1", "GPU-test")
        c.clock = lambda: datetime.datetime(2026, 11, 11, 8, tzinfo=datetime.UTC)
        decision = c.report(receipt(task))
        assert decision["action"] == "pruned"
        assert c.status()["trials"][1]["state"] == "PRUNED"
        assert c.ask("ada") is None


def test_config_bounds_geometry_bool_and_fixed_worker_count(protocol):
    p = copy.deepcopy(protocol["seed_candidate"])
    cfg = config_for(p, "meta", protocol)
    assert cfg["workers"] == 8 and cfg["device"] == "cuda:0"
    for key, value in [
        ("lr", float("nan")),
        ("av_weight", True),
        ("geometry", "384x8"),
        ("depth", 5),
        ("architecture", "legacy"),
    ]:
        bad = dict(p)
        bad[key] = value
        with pytest.raises(ValueError):
            config_for(bad, "meta", protocol)


def test_local_singlewriter_and_no_torch(tmp_path, protocol, profiles):
    import sys

    assert "torch" not in sys.modules
    c = Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW)
    try:
        with pytest.raises(BlockingIOError):
            Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW)
    finally:
        c.close()


def test_restart_preserves_same_lease_and_plan(tmp_path, protocol, profiles):
    path = tmp_path / "study.sqlite3"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        task = c.ask("meta1", "GPU-test")
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        assert c.ask("meta1", "GPU-test") == task
        assert c.report(receipt(task))["action"] == "continue"


def test_asha_prunes_weaker_peer_and_new_config_cap(tmp_path, protocol, profiles):
    protocol["max_new_configs"] = 2
    with Coordinator(tmp_path / "study.sqlite3", protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        first = c.ask("meta1", "GPU-test")
        assert c.report(receipt(first, 0.8))["action"] == "continue"
        with pytest.raises(ValueError, match="migration"):
            c.ask("meta1", "GPU-other")
        second = c.ask("ada", "GPU-second")
        r = receipt(second, 0.1)
        r["gpu_uuid"] = "GPU-second"
        r["checkpoint_qa"]["gpu_uuid"] = "GPU-second"
        d = c.report(r)
        assert d["action"] == "pruned" and d["trial_state"] == "PRUNED"
        assert c.ask("ada", "GPU-third") is None
