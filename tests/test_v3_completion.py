"""Explicit SYNTHETIC final60 pool records; no trained final60 V3 candidate exists.

Native8777 validation uses the real fixed public support/identity metadata. Fake
training records and dummy milestone bytes do not establish an actual completion.
"""

import hashlib
import json
from pathlib import Path

import pytest

from deepretina import selection as g
from deepretina._frozen_v3 import v3
from deepretina.v3_completion import verify_original_pool

ROOT = Path(__file__).parents[1]


@pytest.fixture
def synthetic(tmp_path):
    protocol = v3.make_protocol(
        "2026-11-11T14:30:00Z", "meta01", {"meta": {"cache": True, "workers": 8}}
    )
    protocol["study_id"] = "synthetic-pool-final60"
    profile = {
        "worker_id": "meta01",
        "config_fixed": {"workers": 8},
        "runtime": {
            "intraop_threads": 8,
            "interop_threads": 1,
            "environment": {"OMP_NUM_THREADS": "1"},
        },
    }
    profile.update(
        source_commit=v3.SOURCE,
        execution_code_sha256=v3.EXECUTION,
        training_source_sha256=v3.TRAINING,
    )
    config = {
        **protocol["config_template"],
        **protocol["seed_candidate"],
        "workers": 8,
        "device": "cuda:0",
    }
    plan = {
        **protocol["plan_template"],
        "manifest": "/data/manifest.json",
        "screen_epochs": 3,
        "total_epochs": 60,
        "candidates": {"trial0000": config},
    }
    task = {
        "study_kind": v3.KIND,
        "study_id": protocol["study_id"],
        "worker_id": "meta01",
        "trial": "trial0000",
        "trial_number": 0,
        "task_id": "trial0000-e060",
        "lease": "synthetic-lease",
        "profile": "meta",
        "profile_sha256": v3.digest(profile),
        "gpu_uuid": "GPU-synthetic",
        "plan_sha256": v3.digest(plan),
        "config_sha256": v3.digest(config),
        "protocol_sha256": v3.digest(protocol),
        "search_space_sha256": "a" * 64,
        "deadline_utc": protocol["deadline_utc"],
        "plan": plan,
        "suggested_params": protocol["seed_candidate"],
        "source_commit": v3.SOURCE,
        "execution_code_sha256": v3.EXECUTION,
        "target_epoch": 60,
        "resume": True,
    }
    task.update(previous_epoch=27, is_control=True, candidate_role="seed_control")
    task.update(
        canonical_sha256=plan["prepared_content_sha256"],
        cache_sha256=plan["derived_cache"]["fingerprint"],
    )
    envelope = {
        "v3_protocol": protocol,
        "protocol_sha256": v3.digest(protocol),
        "execution_authorized": True,
        "profiles": {"meta01": {"profile_sha256": task["profile_sha256"], "profile": "meta"}},
        "search_space_sha256": task["search_space_sha256"],
        "deadline_utc": protocol["deadline_utc"],
    }
    envelope.update(
        source_commit=v3.SOURCE,
        execution_code_sha256=v3.EXECUTION,
        training_source_sha256=v3.TRAINING,
        manifest_sha256=plan["manifest_sha256"],
        canonical_sha256=plan["prepared_content_sha256"],
        geometry_policy_sha256=plan["geometry_policy_sha256"],
    )
    index = {
        "schema_version": 1,
        "kind": "original_v3_pool_evidence_index",
        "core_source_commit": "8777d6cca60f7ced08798ad2af7fabf626421d05",
        "files": {},
        "milestone_directory": str(tmp_path / "milestone"),
    }
    for name, value in [
        ("task", task),
        ("protocol", protocol),
        ("envelope", envelope),
        ("profile", profile),
    ]:
        p = tmp_path / (name + ".json")
        raw = g.canonical(value)
        p.write_bytes(raw)
        index["files"][name] = {"path": str(p), "sha256": g.digest(raw)}
    metrics = json.loads((ROOT / "reports/frozen-original-val-nearest/validation.json").read_text())
    metrics.update(family="unified", purpose="full_data", threshold=0.5)
    ids = json.loads((ROOT / "tests/reference/v3_pool/train-identities.json").read_text())
    photos = dict.fromkeys(ids["photos"], 60)
    photos[ids["photos"][0]] += 491520 - sum(photos.values())
    coverage = {
        "photo_visits": photos,
        "annotation_visits": dict.fromkeys(ids["annotations"], 1),
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
    }
    best = {
        "config": config,
        "checkpoint_sha256": g.digest(b"synthetic BEST"),
        "epoch": 59,
        "val_selection_score": 0.8,
        "validation_metrics": {"fixture_best": True},
    }
    last = {
        "config": config,
        "checkpoint_sha256": g.digest(b"synthetic LAST"),
        "epoch": 60,
        "val_selection_score": 0.7,
        "validation_metrics": metrics,
        "coverage": coverage,
    }
    last["steps"] = 122880
    snapshots = {
        "best.pt": b"synthetic BEST",
        "last.pt": b"synthetic LAST",
        "metrics.csv": b"synthetic CSV",
        "training_summary.json": b"{}",
    }
    sidecars = {
        "coverage.json": coverage,
        "validation.json": best["validation_metrics"],
        "run.json": {
            "config": config,
            "training_source_sha256": v3.TRAINING,
            "manifest_sha256": plan["manifest_sha256"],
            "prepared_content_sha256": plan["prepared_content_sha256"],
            "derived_cache": plan["derived_cache"],
        },
        "search_trial.json": {
            "plan_sha256": task["plan_sha256"],
            "trial": task["trial"],
            "execution_code_sha256": v3.EXECUTION,
            "manifest_sha256": plan["manifest_sha256"],
        },
    }
    directory = Path(index["milestone_directory"])
    directory.mkdir()
    hashes = {}
    for name, raw in {**snapshots, **{k: g.canonical(v) for k, v in sidecars.items()}}.items():
        p = directory / name
        p.write_bytes(raw)
        p.chmod(0o444)
        hashes[name] = g.digest(raw)
    qa = {
        "kind": "v3_rung_evidence",
        "status": "PASS",
        "completed_budget": True,
        "actual_epoch": 60,
        "finite_weights": True,
        "finite_optimizer": True,
        "checkpoint_sha256": last["checkpoint_sha256"],
        "last_checkpoint_sha256": last["checkpoint_sha256"],
        "best_checkpoint_sha256": best["checkpoint_sha256"],
        "best_epoch": 59,
        "macro_best_epoch": 59,
        "macro_best_score": 0.8,
        "last_validation_metrics": metrics,
        "last_validation_metrics_sha256": v3.digest(metrics),
        "coverage": coverage,
        "csv_sha256": g.digest(snapshots["metrics.csv"]),
        "csv_text": snapshots["metrics.csv"].decode(),
        "metric": v3.objective_score(metrics),
        "source_commit": v3.SOURCE,
        "execution_code_sha256": v3.EXECUTION,
        "training_source_sha256": v3.TRAINING,
        "plan_sha256": task["plan_sha256"],
        "config_sha256": task["config_sha256"],
        "profile_sha256": task["profile_sha256"],
        "gpu_uuid": task["gpu_uuid"],
        "worker_id": "meta01",
        "host_independent_milestone_sha256_verified": True,
        "independent_milestone": {
            "status": "independent_copy_sha_verified_readonly",
            "hardlinks_used": False,
            "sha256": hashes,
        },
    }
    qa.update(
        steps=122880,
        scheduler_T_max=60,
        scheduler_last_epoch=60,
        delivered_samples=491520,
        eligible_photos=2529,
        seen_photos=2529,
    )
    runtime = {
        "gpu_uuid": task["gpu_uuid"],
        "expected_gpu_uuid": task["gpu_uuid"],
        "cuda_visible_devices": task["gpu_uuid"],
        "runtime_installs": False,
        "pbs_job_id": "synthetic.pbs",
        "intraop_threads": 8,
        "interop_threads": 1,
        "environment": profile["runtime"]["environment"],
    }
    receipt = {
        **{
            k: task[k]
            for k in (
                "study_kind",
                "study_id",
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
            )
        },
        "status": "COMPLETE",
        "actual_epoch": 60,
        "target_epoch": 60,
        "metric": qa["metric"],
        "checkpoint_qa": qa,
        "run_result": {
            "returncode": 0,
            "test_evaluation_performed": False,
            "target_epoch": 60,
            "runtime": runtime,
            "worker": {
                **runtime,
                "source_commit": v3.SOURCE,
                "worker_id": task["worker_id"],
                "profile_sha256": task["profile_sha256"],
                "plan_sha256": task["plan_sha256"],
            },
        },
    }
    return receipt, index, best, last, snapshots


def test_synthetic_native_final60_complete_pool_contract(synthetic):
    result = verify_original_pool(*synthetic)
    assert result["study_terminal_objective"] == 1.0
    assert result["macro_best_epoch"] == 59 and result["macro_best_score"] == 0.8
    assert result["study_objective_role"] == "fixed_terminal60_worst6_ratio"


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "failed",
        "wrong_gpu",
        "wrong_lease",
        "stale_last",
        "objective",
        "missing_annotation",
        "missing_photo",
        "changed_milestone",
        "writable",
        "hardlink",
        "missing_host_proof",
        "retagged",
        "wrong_core",
    ],
)
def test_v3_cannot_promote_partial_or_retagged_evidence(synthetic, case):
    receipt, index, best, last, snapshots = synthetic
    if case == "partial":
        receipt["actual_epoch"] = 27
    if case == "failed":
        receipt["run_result"]["returncode"] = 1
    if case == "wrong_gpu":
        receipt["run_result"]["runtime"]["gpu_uuid"] = "GPU-other"
    if case == "wrong_lease":
        receipt["lease"] = "foreign"
    if case == "stale_last":
        receipt["checkpoint_qa"]["last_checkpoint_sha256"] = "0" * 64
    if case == "objective":
        receipt["metric"] = 0.99
    if case == "missing_annotation":
        receipt["checkpoint_qa"]["coverage"]["annotation_visits"].pop(
            next(iter(last["coverage"]["annotation_visits"]))
        )
    if case == "missing_photo":
        receipt["checkpoint_qa"]["coverage"]["photo_visits"].pop(
            next(iter(last["coverage"]["photo_visits"]))
        )
    if case == "changed_milestone":
        p = Path(index["milestone_directory"]) / "last.pt"
        p.chmod(0o644)
        p.write_bytes(b"changed")
        p.chmod(0o444)
    if case == "writable":
        (Path(index["milestone_directory"]) / "metrics.csv").chmod(0o644)
    if case == "hardlink":
        receipt["checkpoint_qa"]["independent_milestone"]["hardlinks_used"] = True
    if case == "missing_host_proof":
        receipt["checkpoint_qa"]["host_independent_milestone_sha256_verified"] = False
    if case == "retagged":
        receipt["checkpoint_qa"]["kind"] = "standalone_unified_completion_qa"
    if case == "wrong_core":
        index["core_source_commit"] = "0" * 40
    with pytest.raises(ValueError):
        verify_original_pool(receipt, index, best, last, snapshots)


def test_vendored_native_modules_are_byte_exact():
    p = ROOT / "src/deepretina/_frozen_v3"
    provenance = json.loads((p / "provenance.json").read_text())
    for name, sha in provenance["files_sha256"].items():
        assert hashlib.sha256((p / name).read_bytes()).hexdigest() == sha
