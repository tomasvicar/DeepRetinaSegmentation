"""Real Optuna/SQLite protocol2, synthetic receipts only; no model/data/GPU."""

import copy
import json
from pathlib import Path

import pytest
from deepretina_optuna.core import Coordinator
from deepretina_optuna.v2 import NAMES, configurations
from test_core import NOW, receipt


@pytest.fixture
def protocol():
    return json.loads((Path(__file__).parents[1] / "protocol_v2.json").read_text())


@pytest.fixture
def profiles():
    return {
        "ada": {"profile": "ada", "profile_sha256": "a" * 64, "output_root": "/v2/ada"},
        "other": {
            "profile": "old",
            "profile_sha256": "b" * 64,
            "output_root": "/v2/other",
        },
    }


def actual(task, score=0.5):
    value = receipt(task, score)
    value["gpu_uuid"] = task.get("gpu_uuid") or "GPU-test"
    value["checkpoint_qa"]["gpu_uuid"] = value["gpu_uuid"]
    value["checkpoint_qa"].update(
        eligible_photos=2529,
        seen_photos=2529,
        geometry_policy_sha256=task["plan"]["geometry_policy_sha256"],
    )
    return value


def failed(task):
    value = actual(task)
    value["status"] = "FAILED"
    value.pop("checkpoint_qa")
    return value


def test_first_control_is_real_trial0_no_completed_baseline_and_queue_is_finite(
    tmp_path, protocol, profiles
):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        before = c.status()["trials"]
        assert len(before) == 7
        assert all(t["state"] == "WAITING" and t["value"] is None for t in before)
        assert c.ask("other", "GPU-test") is None
        seen = []
        for index, name in enumerate(NAMES):
            task = c.ask("ada", "GPU-test")
            assert task["trial_number"] == index
            assert task["controlled_candidate"] == name and task["is_control"] is (
                index == 0
            )
            cfg = task["plan"]["candidates"][task["trial"]]
            assert cfg == configurations()[name]
            assert (
                task["plan"]["training_photos"] == 2529 and task["cache_sha256"] is None
            )
            seen.append(task["config_sha256"])
            c.report(failed(task))
        assert len(set(seen)) == 7 and c.ask("ada", "GPU-test") is None
        assert c.status()["new_configs"] == 7 and len(c._trials()) == 7
        with pytest.raises(ValueError, match="no imported baseline"):
            c.import_baseline("unused", b"{}", "x")


def test_control_overrides_native_prune_but_registers_real_rungs_and_finishes_only60(
    tmp_path, protocol, profiles
):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        control = c.ask("ada", "GPU-test")
        competitor = c.ask("other", "GPU-other")
        for epoch, rung in [(3, 0), (9, 1), (27, 2)]:
            other_decision = c.report(actual(competitor, 0.9))
            assert other_decision["action"] == "continue"
            competitor = other_decision["next_task"]
            result = actual(control, 0.1)
            decision = c.report(result)
            assert (
                decision["action"] == "continue"
                and decision["native_pruner_suggested_prune"] is True
            )
            assert c.report(result) == decision
            assert c.study.get_trials()[0].system_attrs[f"completed_rung_{rung}"] == 0.1
            control = decision["next_task"]
            assert control["trial_number"] == 0
            assert c.status()["trials"][0]["state"] == "RUNNING"
        result = c.report(actual(control, 0.1))
        assert result["action"] == "finished"
        trial = c.study.get_trials()[0]
        assert trial.state.name == "COMPLETE" and trial.value == 0.1
        assert "completed_rung_3" not in trial.system_attrs


def test_v2_restart_restores_pending_and_never_enqueues_extras(
    tmp_path, protocol, profiles
):
    path = tmp_path / "db"
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        control = c.ask("ada", "GPU-test")
        saved = copy.deepcopy(c.study.user_attrs["sampler_state"])
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as c:
        assert c.sampler_restoration == "exact_saved_streams"
        assert c.study.user_attrs.get("sampler_state_migrations", []) == []
        assert c.ask("ada", "GPU-test") == control
        assert c.study.user_attrs["sampler_state"] == saved and len(c._trials()) == 7


def test_dynamic_photo_coverage_and_policy_identity_are_required(
    tmp_path, protocol, profiles
):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        task = c.ask("ada", "GPU-test")
        for field, bad in [
            ("eligible_photos", 2221),
            ("seen_photos", 2221),
            ("geometry_policy_sha256", "0" * 64),
        ]:
            value = actual(task)
            value["checkpoint_qa"][field] = bad
            with pytest.raises(ValueError):
                c.report(value)
        assert c.report(actual(task))["action"] == "continue"


def test_control_failure_or_deadline_is_not_fake_completion(
    tmp_path, protocol, profiles
):
    with Coordinator(tmp_path / "fail", protocol, profiles, clock=lambda: NOW) as c:
        task = c.ask("ada", "GPU-test")
        assert c.report(failed(task))["trial_state"] == "FAIL"
    with Coordinator(tmp_path / "deadline", protocol, profiles, clock=lambda: NOW) as c:
        import datetime

        task = c.ask("ada", "GPU-test")
        c.clock = lambda: datetime.datetime(2026, 11, 11, 8, tzinfo=datetime.UTC)
        result = c.report(actual(task))
        assert (
            result["trial_state"] == "PRUNED"
            and result["reason"] == "deadline_partial_not_final"
        )


@pytest.mark.parametrize(
    "field,value",
    [("max_new_configs", 8), ("manual_baseline", {}), ("control_worker_id", "other")],
)
def test_v2_inventory_cannot_be_extended_or_reuse_v1_baseline(
    tmp_path, protocol, profiles, field, value
):
    protocol[field] = value
    with pytest.raises(ValueError):
        Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW)
