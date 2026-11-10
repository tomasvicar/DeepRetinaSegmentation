"""Synthetic terminal metadata only; no weights, model/GT loads or CUDA calls."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "pool_completion_adapter", ROOT / "scripts/adapt_pool_completion_qa.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
spec2 = importlib.util.spec_from_file_location(
    "compare_completed_val", ROOT / "scripts/compare_completed_val.py"
)
comparison = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(comparison)


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Synthetic manifest uses the same identity/cardinality contract. Production
    # has a fixed manifest SHA and provides no command-line override.
    manifest_rows = []
    photo_groups = {}
    next_photo = 0
    for name, count in m.SOURCE_PHOTOS.items():
        if name == "rite":
            ids = photo_groups["drive"]
        else:
            ids = [f"photo{i}" for i in range(next_photo, next_photo + count)]
            next_photo += count
        photo_groups[name] = ids
        for photo in ids:
            manifest_rows.append(
                {
                    "id": f"annotation{len(manifest_rows)}",
                    "photo_id": photo,
                    "dataset": name,
                    "split": "train",
                    "task": "vessels",
                }
            )
    original_rows = list(manifest_rows)
    while len(manifest_rows) < 5700:
        row = dict(original_rows[len(manifest_rows) % len(original_rows)])
        row["id"] = f"annotation{len(manifest_rows)}"
        manifest_rows.append(row)
    manifest_bytes = m.canonical(manifest_rows)
    monkeypatch.setattr(m, "MANIFEST", m.sha(manifest_bytes))
    p = json.loads((ROOT / "tools/optuna_coordinator/protocol_v2.json").read_text())
    plan = p["plan_template"]
    plan["manifest_sha256"] = m.MANIFEST
    config = p["candidate_configs"]["control"]
    trial = "trial0000"
    plan.update(candidates={trial: config}, total_epochs=60, screen_epochs=3)
    plan["derived_cache"] = {
        "schema": 1,
        "fingerprint": m.CACHE,
        "canonical_fingerprint": m.CONTENT,
        "allow_canonical_fallback": False,
    }
    task = {
        "output": "/synthetic/run",
        "study_id": "synthetic-fixture",
        "study_kind": "finite_v2_controlled",
        "trial": trial,
        "trial_number": 0,
        "task_id": "trial0000-e060",
        "lease": "synthetic-e060",
        "worker_id": "ada",
        "profile_sha256": "a" * 64,
        "gpu_uuid": "GPU-synthetic",
        "target_epoch": 60,
        "previous_epoch": 27,
        "resume": True,
        "source_commit": m.SOURCE,
        "execution_code_sha256": m.EXECUTION,
        "canonical_sha256": m.CONTENT,
        "cache_sha256": m.CACHE,
        "plan": plan,
        "plan_sha256": m.sha(m.canonical(plan)),
        "config_sha256": m.sha(m.canonical(config)),
    }
    visits = {f"photo{i}": 60 * 8192 // 2529 + (i < 60 * 8192 % 2529) for i in range(2529)}
    coverage = {
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "photo_visits": visits,
        "annotation_visits": {f"annotation{i}": 100 for i in range(5700)},
        "by_dataset": {
            name: {"eligible_photos": count, "seen_photos": count}
            for name, count in m.SOURCE_PHOTOS.items()
        },
    }
    csv = "epoch,steps,val_selection_score,val_macro_dataset_mean_dice\n" + "".join(
        f"{i},{i * 2048},{0.6 if i == 56 else 0.1},{0.6 if i == 56 else 0.1}\n"
        for i in range(1, 61)
    )
    documents = {
        "coverage.json": m.canonical(coverage),
        "metrics.csv": csv.encode(),
        "run.json": m.canonical(
            {
                "config": config,
                "manifest_sha256": m.MANIFEST,
                "prepared_content_sha256": m.CONTENT,
                "training_source_sha256": m.TRAINING,
                "derived_cache": plan["derived_cache"],
                "geometry_policy": {
                    "format_version": 1,
                    "sha256": m.POLICY,
                    "document_json": (
                        ROOT / "reports/data/corpus-v2-geometry-policy/policy.json"
                    ).read_text(),
                },
            }
        ),
        "training_summary.json": m.canonical(
            {
                "completed_budget": True,
                "steps": 122880,
                "best_validation_score": 0.6,
                "selection_metric": config["selection_metric"],
            }
        ),
        "validation.json": m.canonical({"macro_dataset_mean_dice": 0.6, "selection_score": 0.6}),
        "search_trial.json": m.canonical(
            {
                "plan_sha256": task["plan_sha256"],
                "trial": trial,
                "execution_code_sha256": m.EXECUTION,
                "manifest_sha256": m.MANIFEST,
            }
        ),
    }
    milestone = tmp_path / "immutable"
    milestone.mkdir()
    proof = {
        "source_run": task["output"],
        "destination": str(milestone),
        "status": "independent_copy_sha_verified_readonly",
        "hardlinks_used": False,
        "sha256": {
            "best.pt": "b" * 64,
            "last.pt": "c" * 64,
            **{k: m.sha(v) for k, v in documents.items()},
        },
    }
    documents["copy-verification.json"] = m.canonical(proof)
    for name, raw in documents.items():
        path = milestone / name
        path.write_bytes(raw)
        path.chmod(0o444)
    milestone.chmod(0o500)
    q = {
        "output": task["output"],
        "actual_epoch": 60,
        "status": "PASS",
        "completed_budget": True,
        "finite_weights": True,
        "finite_optimizer": True,
        "scheduler_T_max": 60,
        "scheduler_last_epoch": 60,
        "steps": 122880,
        "delivered_samples": 491520,
        "source_commit": m.SOURCE,
        "execution_code_sha256": m.EXECUTION,
        "training_source_sha256": m.TRAINING,
        "manifest_sha256": m.MANIFEST,
        "canonical_sha256": m.CONTENT,
        "geometry_policy_sha256": m.POLICY,
        "cache_sha256": m.CACHE,
        "independent_milestone": proof,
        "best_checkpoint_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "csv_text": csv,
        "csv_sha256": m.sha(csv.encode()),
        "best_epoch": 56,
        "metric": 0.6,
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "minimum_photo_visits": min(visits.values()),
    }
    for key in ("worker_id", "profile_sha256", "plan_sha256", "config_sha256", "gpu_uuid"):
        q[key] = task[key]
    receipt = {
        **{
            key: task[key]
            for key in (
                "study_id",
                "trial_number",
                "task_id",
                "lease",
                "worker_id",
                "profile_sha256",
                "plan_sha256",
                "config_sha256",
                "gpu_uuid",
            )
        },
        "status": "COMPLETE",
        "actual_epoch": 60,
        "target_epoch": 60,
        "metric": 0.6,
        "checkpoint_qa": q,
        "run_result": {
            "output": task["output"],
            "worker": {
                key: task[key]
                for key in (
                    "source_commit",
                    "profile_sha256",
                    "plan_sha256",
                    "gpu_uuid",
                    "worker_id",
                )
            },
            "returncode": 0,
            "target_epoch": 60,
            "test_evaluation_performed": False,
        },
    }
    decision = {
        **{key: task[key] for key in ("study_id", "trial_number", "task_id", "lease")},
        "action": "finished",
        "trial_state": "COMPLETE",
        "value": 0.6,
        "receipt_sha256": m.sha(m.canonical(receipt)),
    }
    return receipt, task, decision, milestone, manifest_bytes


def run(case):
    r, t, d, path, manifest_bytes = case
    # Original bytes deliberately use different formatting from canonical decision digest.
    raw = json.dumps(r, indent=2).encode()
    return m.adapt(
        raw,
        m.canonical(t),
        m.canonical(d),
        manifest_bytes,
        trusted_receipt_sha256=m.sha(raw),
        trusted_task_sha256=m.sha(m.canonical(t)),
        trusted_decision_sha256=m.sha(m.canonical(d)),
        milestone=path,
    )


def bind(case):
    case[2]["receipt_sha256"] = m.sha(m.canonical(case[0]))


def test_consistent_e060_is_typed_and_satisfies_unchanged_comparison_gate(case):
    qa = run(case)
    assert qa["terminal"] is True and qa["last_epoch"] == 60
    assert json.loads(qa["original_receipt_text"]) == case[0]
    assert qa["original_receipt_sha256"] != case[2]["receipt_sha256"]
    assert not (case[3] / "best.pt").exists()  # Checkpoint files are never opened.
    evidence = {
        "checkpoint": {"sha256": "b" * 64, "epoch": 56},
        "data": {
            "manifest_sha256": m.MANIFEST,
            "prepared_content_sha256": m.CONTENT,
            "eligible_photo_counts": {"train": 2529},
        },
        "source": {"training_source_sha256": m.TRAINING},
    }
    comparison.verify_completion(qa, evidence, expected_train_photo_count=2529)


def test_actual_stripped_e003_rejected_before_any_milestone_read():
    raw = (ROOT / "tests/reference/pool_completion/actual-v2-e003-stripped.json").read_bytes()
    with pytest.raises(ValueError, match="actual terminal E60"):
        m.adapt(
            raw,
            b"",
            b"",
            trusted_receipt_sha256=m.sha(raw),
            trusted_task_sha256="",
            trusted_decision_sha256="",
            milestone="/never-read",
        )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("actual_epoch", 3),
        ("completed_budget", False),
        ("finite_weights", False),
        ("finite_optimizer", False),
        ("steps", 1),
        ("scheduler_last_epoch", 27),
        ("geometry_policy_sha256", "0" * 64),
        ("canonical_sha256", "0" * 64),
        ("cache_sha256", "0" * 64),
        ("best_epoch", 55),
        ("minimum_photo_visits", 59),
    ],
)
def test_inconsistent_or_partial_pool_qa_rejected(case, field, bad):
    case[0]["checkpoint_qa"][field] = bad
    bind(case)
    with pytest.raises(ValueError):
        run(case)


def test_forged_unbound_receipt_and_nonterminal_decision_rejected(case):
    case[0]["metric"] = 0.7
    with pytest.raises(ValueError, match="bound completed"):
        run(case)
    case[0]["metric"] = 0.6
    case[2]["action"] = "continue"
    with pytest.raises(ValueError, match="bound completed"):
        run(case)


def test_original_byte_sha_is_mandatory(case):
    r, t, d, p, _ = case
    with pytest.raises(ValueError, match="Trusted original bytes"):
        m.adapt(
            m.canonical(r),
            m.canonical(t),
            m.canonical(d),
            trusted_receipt_sha256="0" * 64,
            trusted_task_sha256=m.sha(m.canonical(t)),
            trusted_decision_sha256=m.sha(m.canonical(d)),
            milestone=p,
        )


@pytest.mark.parametrize("mode", ["writable", "mutated"])
def test_milestone_must_stay_immutable_and_byte_pinned(case, mode):
    path = case[3] / "coverage.json"
    path.chmod(0o644)
    if mode == "mutated":
        path.write_text("{}")
        path.chmod(0o444)
    with pytest.raises(ValueError, match="metadata"):
        run(case)


def replace_sidecar(case, name, change):
    path = case[3] / name
    value = json.loads(path.read_bytes())
    change(value)
    raw = m.canonical(value)
    path.chmod(0o644)
    path.write_bytes(raw)
    path.chmod(0o444)
    proof = case[0]["checkpoint_qa"]["independent_milestone"]
    proof["sha256"][name] = m.sha(raw)
    copy_path = case[3] / "copy-verification.json"
    copy_path.chmod(0o644)
    copy_path.write_bytes(m.canonical(proof))
    copy_path.chmod(0o444)
    bind(case)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(eligible_annotation_records=5699),
        lambda c: c.update(eligible_photos=2528),
        lambda c: c["by_dataset"]["gamma"].update(eligible_photos=100, seen_photos=100),
        lambda c: c["photo_visits"].update(photo0=59),
        lambda c: c["photo_visits"].update(photo0=1000),
        lambda c: c.update(contributing_annotation_records=5699),
    ],
)
def test_even_rehashed_but_inconsistent_actual_coverage_rejected(case, change):
    replace_sidecar(case, "coverage.json", change)
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize(
    "name,change",
    [
        ("run.json", lambda d: d.update(training_source_sha256="0" * 64)),
        ("run.json", lambda d: d["geometry_policy"].update(document_json="{}")),
        ("training_summary.json", lambda d: d.update(completed_budget=False)),
        ("validation.json", lambda d: d.update(selection_score=0.7)),
        ("search_trial.json", lambda d: d.update(trial="other")),
    ],
)
def test_rehashed_sidecars_cannot_override_bound_source_terminal_or_best_metadata(
    case, name, change
):
    replace_sidecar(case, name, change)
    with pytest.raises(ValueError):
        run(case)


def test_cli_writes_fresh_typed_qa_and_preserves_existing_output(case, tmp_path, monkeypatch):
    import sys

    r, t, d, milestone, manifest_bytes = case
    args = ["adapter"]
    for name, value in [("receipt", r), ("task", t), ("decision", d)]:
        path = tmp_path / (name + ".json")
        raw = json.dumps(value, indent=2).encode()
        path.write_bytes(raw)
        args.extend(["--" + name, str(path), "--trusted-" + name + "-sha256", m.sha(raw)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    args.extend(["--manifest", str(manifest_path)])
    output = tmp_path / "completion.json"
    args.extend(["--milestone", str(milestone), "--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    m.main()
    assert json.loads(output.read_bytes())["last_epoch"] == 60
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["annotation_visits"].pop("annotation0"),
        lambda c: c["annotation_visits"].update(foreign=c["annotation_visits"].pop("annotation0")),
        lambda c: c["annotation_visits"].update(annotation0=0),
        lambda c: c["photo_visits"].update(foreign=c["photo_visits"].pop("photo0")),
    ],
)
def test_exact_manifest_ids_cannot_be_replaced_by_same_count_or_subset(case, change):
    replace_sidecar(case, "coverage.json", change)
    with pytest.raises(ValueError):
        run(case)


def test_manifest_original_bytes_must_match_fixed_corpus_pin(case):
    mutated = (*case[:4], case[4] + b" ")
    with pytest.raises(ValueError, match="manifest bytes"):
        run(mutated)


def test_canonical_worker_completion_is_bound_without_implying_cache(case):
    r, t, _, _, _ = case
    t["plan"]["derived_cache"] = None
    t["cache_sha256"] = None
    t["plan_sha256"] = m.sha(m.canonical(t["plan"]))
    r["plan_sha256"] = r["checkpoint_qa"]["plan_sha256"] = t["plan_sha256"]
    r["checkpoint_qa"]["cache_sha256"] = None
    r["run_result"]["worker"]["plan_sha256"] = t["plan_sha256"]
    replace_sidecar(case, "run.json", lambda d: d.update(derived_cache=None))
    replace_sidecar(case, "search_trial.json", lambda d: d.update(plan_sha256=t["plan_sha256"]))
    result = run(case)
    assert result["derived_cache"] is None
    assert result["all_eligible_annotations_contributed"] is True
