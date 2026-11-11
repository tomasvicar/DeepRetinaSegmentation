"""Adaptive protocol3 with TEMP SQLite and synthetic receipts, no GPU/model/GT."""

import copy
import json
from pathlib import Path

import pytest
from deepretina_optuna import v3
from deepretina_optuna.core import Coordinator, digest, validate_params
from test_core import NOW, receipt

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def protocol():
    return v3.make_protocol(
        "2026-11-12T20:00:00Z", "meta0", {"meta": {"cache": True, "workers": 2}}
    )


@pytest.fixture
def profiles():
    return {
        f"meta{i}": {
            "profile": "meta",
            "profile_sha256": str(i) * 64,
            "output_root": f"/new-v3/meta{i}",
        }
        for i in range(6)
    }


def metrics(scale=1):
    m = json.loads((FIX / "v3-last-metrics.json").read_text())
    if scale != 1:
        for p in m["per_photo"]:
            for c in p["channels"].values():
                old = c["tp"]
                c["tp"] = int(old * scale)
                c["fn"] += old - c["tp"]
                c["dice"] = v3._dice(c)
            c = p["av_conditional_counts"]
            c["correct"] = int(c["correct"] * scale)
            c["end_to_end_correct"] = p["channels"]["artery"]["tp"] + p["channels"]["vein"]["tp"]
        for name, c in m["channels"].items():
            for key in ("tp", "fp", "fn", "valid_pixels"):
                c[key] = sum(p["channels"][name][key] for p in m["per_photo"])
            c["dice"] = v3._dice(c)
        m["av_conditional_accuracy"] = sum(
            p["av_conditional_counts"]["correct"] for p in m["per_photo"]
        ) / sum(p["av_conditional_counts"]["scored_pixels"] for p in m["per_photo"])
    return m


def actual(task, scale=1):
    r = receipt(task, 0.5)
    q = r["checkpoint_qa"]
    m = metrics(scale)
    r["gpu_uuid"] = task["gpu_uuid"]
    q["gpu_uuid"] = task["gpu_uuid"]
    q.update(
        eligible_photos=2529,
        seen_photos=2529,
        geometry_policy_sha256=task["plan"]["geometry_policy_sha256"],
        kind="v3_rung_evidence",
        last_checkpoint_sha256=q["checkpoint_sha256"],
        last_validation_metrics=m,
        last_validation_metrics_sha256=digest(m),
        macro_best_score=0.5,
        macro_best_epoch=1,
    )
    q["metric"] = r["metric"] = v3.objective_score(m)
    if task["target_epoch"] == 60:
        ids = json.loads((FIX / "v3-train-inventory.json").read_text())
        photos = {p: 194 for p in ids["photo_ids"]}
        photos[ids["photo_ids"][0]] += 491520 - sum(photos.values())
        q["coverage"] = {
            "eligible_annotation_records": 5700,
            "contributing_annotation_records": 5700,
            "annotation_visits": dict.fromkeys(ids["annotation_ids"], 1),
            "photo_visits": photos,
        }
    return r


def fail(task):
    r = receipt(task)
    r.update(status="FAILED", gpu_uuid=task["gpu_uuid"])
    r.pop("checkpoint_qa")
    return r


def test_real_seed_control_dynamic_beyond_seven_and_five_gpu_cap(tmp_path, protocol, profiles):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        assert c.ask("meta1", "GPU1") is None
        first = c.ask("meta0", "GPU0")
        assert first["trial_number"] == 0 and first["is_control"] is True
        assert first["suggested_params"] == protocol["seed_candidate"]
        active = [first] + [c.ask(f"meta{i}", f"GPU{i}") for i in range(1, 5)]
        assert all(active) and c.ask("meta5", "GPU5") is None
        for task in active:
            c.report(fail(task))
        for index in range(5, 24):
            task = c.ask("meta0", "GPU0")
            assert task["trial_number"] == index
            validate_params(task["suggested_params"], protocol)
            assert task["candidate_role"] == "adaptive_candidate"
            c.report(fail(task))
        assert c.ask("meta0", "GPU0") is None and len(c._trials()) == 24
        with pytest.raises(ValueError, match="no imported baseline"):
            c.import_baseline("", b"", "x")


def test_control_protected_competitor_delayed_pruning_and_terminal_last(
    tmp_path, protocol, profiles
):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        control = c.ask("meta0", "GPU0")
        other = c.ask("meta1", "GPU1")
        # Force native suggestion;3/9 noncontrols must still be protected.
        from optuna.trial import Trial

        original = Trial.should_prune
        Trial.should_prune = lambda self: True
        try:
            for epoch in (3, 9, 27):
                decision = c.report(actual(control))
                assert decision["action"] == "continue"
                control = decision["next_task"]
                decision = c.report(actual(other, 0.8))
                if epoch < 27:
                    assert decision["action"] == "continue"
                    other = decision["next_task"]
                else:
                    assert decision["trial_state"] == "PRUNED"
            r = actual(control)
            decision = c.report(r)
            assert decision["trial_state"] == "COMPLETE" and decision["value"] == r["metric"] != 0.5
            assert c.report(r) == decision
        finally:
            Trial.should_prune = original


@pytest.mark.parametrize(
    "point",
    [
        "v3_issued_journal",
        "v3_issued_running",
        "v3_parameter_augmentation",
        "v3_parameter_lr",
        "v3_parameter_base_channels",
        "v3_parameter_norm",
        "v3_parameter_vascular_sampling_weight",
        "v3_parameter_weight_decay",
        "v3_issued_origin",
        "v3_issued_meta",
        "v3_issued_pending",
    ],
)
def test_interrupted_issuance_restarts_same_control_and_task(
    tmp_path, protocol, profiles, monkeypatch, point
):
    path = tmp_path / "db"
    c = Coordinator(path, protocol, profiles, clock=lambda: NOW)

    def inject(p):
        if p == point:
            raise RuntimeError("injected")

    monkeypatch.setattr(c, "_transition_point", inject)
    with pytest.raises(RuntimeError):
        c.ask("meta0", "GPU0")
    c.close()
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as restarted:
        task = restarted.ask("meta0", "GPU0")
        assert (
            task["trial_number"] == 0
            and task["is_control"]
            and task["suggested_params"] == protocol["seed_candidate"]
        )
        assert restarted.ask("meta0", "GPU0") == task and len(restarted._trials()) == 1


@pytest.mark.parametrize(
    "point",
    ["v3_reported_journal", "v3_reported_pending", "v3_reported_receipt", "v3_reported_decision"],
)
def test_interrupted_promotion_replays_old_receipt(
    tmp_path, protocol, profiles, monkeypatch, point
):
    path = tmp_path / "db"
    c = Coordinator(path, protocol, profiles, clock=lambda: NOW)
    task = c.ask("meta0", "GPU0")
    r = actual(task)

    def inject(p):
        if p == point:
            raise RuntimeError("injected")

    monkeypatch.setattr(c, "_transition_point", inject)
    with pytest.raises(RuntimeError):
        c.report(r)
    c.close()
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as restarted:
        decision = restarted.report(r)
        assert (
            decision["next_task"]["target_epoch"] == 9
            and restarted.ask("meta0", "GPU0") == decision["next_task"]
        )


def test_six_metrics_artery_regression_and_count_tamper():
    m = metrics()
    values = v3.objective_metrics(m)
    assert set(values) == {"vessel", "artery", "vein", "disc", "cup", "av_conditional_accuracy"}
    lower = metrics(0.7)
    assert v3.objective_score(lower) < v3.objective_score(m)
    m["channels"]["artery"]["tp"] += 1
    with pytest.raises(ValueError, match="count mismatch"):
        v3.objective_score(m)


def test_terminal_rejects_missing_annotation_or_replaced_id(tmp_path, protocol, profiles):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        task = c.ask("meta0", "GPU0")
        for _ in range(3):
            task = c.report(actual(task))["next_task"]
        for change in ("zero", "id", "photo"):
            r = actual(task)
            cover = r["checkpoint_qa"]["coverage"]
            if change == "zero":
                cover["annotation_visits"][next(iter(cover["annotation_visits"]))] = 0
            elif change == "id":
                cover["annotation_visits"]["invented"] = cover["annotation_visits"].pop(
                    next(iter(cover["annotation_visits"]))
                )
            else:
                cover["photo_visits"]["invented"] = cover["photo_visits"].pop(
                    next(iter(cover["photo_visits"]))
                )
            with pytest.raises(ValueError, match="TRAIN"):
                c.report(r)
        assert c.report(actual(task))["trial_state"] == "COMPLETE"


@pytest.mark.parametrize("change", ["source", "bounds", "fixed", "reference", "pruner"])
def test_protocol_admission_fails_before_db(tmp_path, protocol, profiles, change):
    p = copy.deepcopy(protocol)
    if change == "source":
        p["plan_template"]["source_commit"] = "a" * 40
    elif change == "bounds":
        p["search_space"]["lr"]["high"] = 0.01
    elif change == "fixed":
        p["config_template"]["mixed_precision"] = "none"
    elif change == "reference":
        p["objective"]["counts"]["artery"]["tp"] += 1
    else:
        p["pruner"]["min_resource"] = 3
    with pytest.raises(ValueError):
        Coordinator(tmp_path / "db", p, profiles, clock=lambda: NOW)
    assert not (tmp_path / "db").exists()


@pytest.mark.parametrize("point", ["v3_issued_running", "v3_parameter_lr", "v3_issued_meta"])
def test_partial_adaptive_parameters_and_binding_recover_without_new_trial(
    tmp_path, protocol, profiles, monkeypatch, point
):
    path = tmp_path / "db"
    c = Coordinator(path, protocol, profiles, clock=lambda: NOW)
    c.report(fail(c.ask("meta0", "GPU0")))

    def inject(p):
        if p == point:
            raise RuntimeError("injected")

    monkeypatch.setattr(c, "_transition_point", inject)
    with pytest.raises(RuntimeError):
        c.ask("meta1", "GPU1")
    before = copy.deepcopy(c._trials()[1].params)
    c.close()
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as restarted:
        task = restarted.ask("meta1", "GPU1")
        assert (
            task["trial_number"] == 1
            and task["worker_id"] == "meta1"
            and task["gpu_uuid"] == "GPU1"
        )
        assert not task["is_control"] and len(restarted._trials()) == 2
        assert all(task["suggested_params"][k] == v for k, v in before.items())
        validate_params(task["suggested_params"], protocol)
        assert restarted.sampler_restoration in {
            "exact_saved_streams",
            "unclean_ask_history_mismatch",
        }


@pytest.mark.parametrize(
    "point", ["v3_reported_receipt", "v3_reported_decision", "v3_reported_told"]
)
def test_terminal_tell_crash_recovery_is_idempotent(
    tmp_path, protocol, profiles, monkeypatch, point
):
    path = tmp_path / "db"
    c = Coordinator(path, protocol, profiles, clock=lambda: NOW)
    task = c.ask("meta0", "GPU0")
    for _ in range(3):
        task = c.report(actual(task))["next_task"]
    r = actual(task)

    def inject(p):
        if p == point:
            raise RuntimeError("injected")

    monkeypatch.setattr(c, "_transition_point", inject)
    with pytest.raises(RuntimeError):
        c.report(r)
    c.close()
    with Coordinator(path, protocol, profiles, clock=lambda: NOW) as restarted:
        decision = restarted.report(r)
        assert (
            decision["trial_state"] == "COMPLETE"
            and restarted._trials()[0].state.name == "COMPLETE"
        )
        assert (
            restarted.report(r) == decision and restarted.study.user_attrs["v3_transition"] is None
        )


def test_genuine_tpe_sampling_after_eight_completed_observations(
    tmp_path, protocol, profiles, monkeypatch
):
    from optuna.trial import Trial

    monkeypatch.setattr(Trial, "should_prune", lambda self: False)
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        calls = []
        native = c.study.sampler._sample

        def observed(*args, **kwargs):
            calls.append(True)
            return native(*args, **kwargs)

        monkeypatch.setattr(c.study.sampler, "_sample", observed)
        for i in range(8):
            task = c.ask("meta0", "GPU0")
            for _ in range(3):
                task = c.report(actual(task, 0.7 + i * 0.02))["next_task"]
            assert c.report(actual(task, 0.7 + i * 0.02))["trial_state"] == "COMPLETE"
        assert not calls
        task = c.ask("meta0", "GPU0")
        assert task["trial_number"] == 8 and calls
        validate_params(task["suggested_params"], protocol)


def test_artery_alone_can_be_objective_bottleneck():
    m = metrics()
    for p in m["per_photo"]:
        c = p["channels"]["artery"]
        old = c["tp"]
        c["tp"] = int(old * 0.1)
        c["fn"] += old - c["tp"]
        c["dice"] = v3._dice(c)
        p["av_conditional_counts"]["end_to_end_correct"] = c["tp"] + p["channels"]["vein"]["tp"]
    pooled = m["channels"]["artery"]
    for k in ("tp", "fp", "fn", "valid_pixels"):
        pooled[k] = sum(p["channels"]["artery"][k] for p in m["per_photo"])
    pooled["dice"] = v3._dice(pooled)
    values = v3.objective_metrics(m)
    assert v3.objective_score(m) == values["artery"] / v3._dice(v3.REFERENCE["counts"]["artery"])


@pytest.mark.parametrize(
    "change", ["metric_sha", "last_sha", "macro", "gt_support", "av_count", "objective"]
)
def test_actual_last_binding_and_derived_evidence_tampering_rejected(
    tmp_path, protocol, profiles, change
):
    with Coordinator(tmp_path / "db", protocol, profiles, clock=lambda: NOW) as c:
        r = actual(c.ask("meta0", "GPU0"))
        q = r["checkpoint_qa"]
        if change == "metric_sha":
            q["last_validation_metrics_sha256"] = "0" * 64
        elif change == "last_sha":
            q["last_checkpoint_sha256"] = "0" * 64
        elif change == "macro":
            q["macro_best_score"] = 0.8
        elif change == "objective":
            q["metric"] = r["metric"] = 0.5
        else:
            m = q["last_validation_metrics"]
            if change == "gt_support":
                m["per_photo"][0]["record_ids"] = ["invented"]
            else:
                m["av_conditional_accuracy"] = 0.5
            q["last_validation_metrics_sha256"] = digest(m)
        with pytest.raises(ValueError):
            c.report(r)


def test_conditional_av_cannot_disagree_with_actual_class_true_positives():
    m = metrics()
    row = next(p for p in m["per_photo"] if p["av_conditional_counts"]["end_to_end_correct"] > 0)
    row["av_conditional_counts"]["end_to_end_correct"] -= 1
    with pytest.raises(ValueError, match="class-count evidence"):
        v3.objective_score(m)
