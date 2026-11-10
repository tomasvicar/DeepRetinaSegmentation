"""Real pinned Optuna5/SQLite tests; no model, data reader, transport or GPU."""

import copy
import json
import sqlite3

import optuna
import pytest
import test_core
from deepretina_optuna.core import Coordinator
from test_core import NOW, baseline, receipt


@pytest.fixture
def protocol():
    return test_core.protocol.__wrapped__()


@pytest.fixture
def profiles():
    return test_core.profiles.__wrapped__()


def finish_failed(c, task):
    result = {
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
    result.update(status="FAILED", gpu_uuid="GPU-test", completed_utc=NOW.isoformat())
    c.report(result)


def warm_tpe(c):
    first = c.study.get_trials()[0]
    for i in range(10):
        params = dict(first.params, lr=0.00011 + i * 0.00001)
        c.study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=first.distributions,
                value=0.4 + i * 0.01,
                user_attrs={"origin": "synthetic_tpe_history"},
            )
        )
    c._persist_sampler_state()


@pytest.mark.parametrize("tpe", [False, True])
def test_restart_next_parameters_equal_uninterrupted(tmp_path, protocol, profiles, tpe):
    uninterrupted = []
    for variant in ["continuous", "restart"]:
        path = tmp_path / f"{variant}.sqlite3"
        c = Coordinator(path, protocol, profiles, clock=lambda: NOW)
        baseline(c)
        if tpe:
            warm_tpe(c)
        sampled = []
        try:
            for index in range(4):
                task = c.ask("ada", "GPU-test")
                sampled.append(c._trial(task["trial_number"]).params)
                finish_failed(c, task)
                if variant == "restart":
                    c.close()
                    c = Coordinator(path, protocol, profiles, clock=lambda: NOW)
                    assert c.sampler_restoration == "exact_saved_streams"
            uninterrupted.append(sampled)
        finally:
            c.close()
    assert uninterrupted[0] == uninterrupted[1]
    assert len({json.dumps(p, sort_keys=True) for p in uninterrupted[0]}) == 4


def test_pending_ask_and_promotion_do_not_advance_streams(tmp_path, protocol, profiles):
    path = tmp_path / "db"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        task = c.ask("ada", "GPU-test")
        saved = copy.deepcopy(c.study.user_attrs["sampler_state"])
        assert c.ask("ada", "GPU-test") == task
        assert c.study.user_attrs["sampler_state"] == saved
        next_task = c.report(receipt(task, score=0.9))["next_task"]
        assert c.study.user_attrs["sampler_state"] == saved
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        assert c.ask("ada", "GPU-test") == next_task
        assert c.study.user_attrs["sampler_state"] == saved


def test_legacy_missing_state_migrates_once_with_recorded_deterministic_seeds(
    tmp_path, protocol, profiles
):
    path = tmp_path / "legacy"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        first = c.ask("ada", "GPU-test")
        old_params = c._trial(first["trial_number"]).params
        finish_failed(c, first)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM study_user_attributes WHERE key='sampler_state'")
        db.commit()
        with sqlite3.connect(tmp_path / "clone") as destination:
            db.backup(destination)
    results = []
    seeds = []
    for dbpath in [path, tmp_path / "clone"]:
        with Coordinator(dbpath, protocol, profiles, clock=lambda: NOW) as c:
            assert c.sampler_restoration == "missing_legacy_state"
            events = c.study.user_attrs["sampler_state_migrations"]
            assert len(events) == 1
            assert events[0]["original_stream_recovery_claim"] is False
            seeds.append(events[0]["derived_seeds"])
            task = c.ask("ada", "GPU-test")
            results.append(c._trial(task["trial_number"]).params)
        with Coordinator(dbpath, protocol, profiles, clock=lambda: NOW) as c:
            assert len(c.study.user_attrs["sampler_state_migrations"]) == 1
            assert c.sampler_restoration == "exact_saved_streams"
    assert results[0] == results[1] and results[0] != old_params
    assert seeds[0] == seeds[1] and all(value != 42 for value in seeds[0].values())


def test_corrupt_sampler_json_fails_closed(tmp_path, protocol, profiles):
    path = tmp_path / "db"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        saved = c.study.user_attrs["sampler_state"]
        saved["streams"]["tpe"]["keys"][0] ^= 1
        c.study.set_user_attr("sampler_state", saved)
    with pytest.raises(ValueError, match="checksum"):
        Coordinator(path, protocol, profiles, clock=lambda: NOW)


def test_unclean_ask_history_mismatch_is_recorded_not_silently_replayed(
    tmp_path, protocol, profiles
):
    path = tmp_path / "db"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        baseline(c)
        c._persist_sampler_state(phase="before_ask")
        # Crash boundary: Optuna allocated a trial after the write-ahead marker.
        c.study.ask()
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        assert c.sampler_restoration == "unclean_ask_history_mismatch"
        assert (
            c.study.user_attrs["sampler_state_migrations"][-1]["reason"]
            == "unclean_ask_history_mismatch"
        )


def test_baseline_import_refreshes_history_without_consuming_seed42(tmp_path, protocol, profiles):
    path = tmp_path / "db"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        initial = copy.deepcopy(c.study.user_attrs["sampler_state"])
        baseline(c)
        updated = c.study.user_attrs["sampler_state"]
        assert initial["streams"] == updated["streams"]
        assert initial["history_sha256"] != updated["history_sha256"]
        assert c.sampler_restoration == "new_study_seed42"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        assert c.sampler_restoration == "exact_saved_streams"
        assert c.study.user_attrs.get("sampler_state_migrations", []) == []
