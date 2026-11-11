"""Synthetic770 VAL metadata only; no real photos, GT, weights or TEST reads."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "frozen_val_compare", ROOT / "scripts/compare_frozen_completed_val.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case(tmp_path):
    rows = []
    for split, ids in [
        ("train", ["train"]),
        ("val", [f"val{i:03}" for i in range(770)]),
        ("test", ["test"]),
    ]:
        for pid in ids:
            for task in ["av", "disc_cup"]:
                rows.append(
                    {
                        "id": pid + task,
                        "photo_id": pid,
                        "source_id": pid,
                        "split_group_id": pid,
                        "dataset": "synthetic",
                        "task": task,
                        "split": split,
                        "preprocessing_profile": "legacy25_raw",
                        "pixels_per_degree": 25,
                    }
                )
    raw = m.canonical(rows)
    groups, counts, oi, ui = m.inventory(raw)
    data = {
        "manifest_sha256": m.sha(raw),
        "prepared_content_sha256": "d" * 64,
        "eligible_photo_counts": counts,
        "split_inventory_sha256": ui,
        "fingerprint_schema": 2,
    }
    cfg = {
        "purpose": "full_data",
        "epochs": 60,
        "epoch_samples": 8192,
        "coverage_first": True,
        "legacy_initialization": None,
        "allow_historical_exposure": False,
        "max_steps": None,
        "batch_size": 4,
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25,
    }
    protocol = {
        "threshold": 0.5,
        "geometry": "canonical_prepared",
        "normalization": "RGB/255-0.5",
        "preprocessing_profile": "legacy25_raw",
        "pixels_per_degree": 25,
        "channels": list(m.CHANNELS),
        "metric_support": {"synthetic": "known"},
        "annotation_policy": "first sorted",
    }
    entries = []
    for pid, records in groups["val"].items():
        entries.append(
            {
                "photo_id": pid,
                "record_ids": [r["id"] for r in records],
                "dataset": "synthetic",
                "channel_datasets": dict.fromkeys(m.CHANNELS, "synthetic"),
                "channels": {
                    n: {"tp": 6, "fp": 1, "fn": 4, "valid_pixels": 20} for n in m.CHANNELS
                },
                "av_conditional_counts": {
                    "correct": 16,
                    "scored_pixels": 20,
                    "eligible_pixels": 20,
                    "end_to_end_correct": 12,
                },
            }
        )
    metrics = {
        "split": "val",
        "photos": 770,
        "records": 1540,
        "threshold": 0.5,
        "family": "unified",
        "purpose": "full_data",
        "supervised_channels": [True] * 5,
        "metric_support": protocol["metric_support"],
        "evaluation_policy": protocol["annotation_policy"],
        "per_photo": entries,
    }
    source = {
        "evaluation_source_sha256": "1" * 64,
        "training_source_sha256": "2" * 64,
        "runtime": {"python": "fixture"},
    }
    cp = {"sha256": "e" * 64, "epoch": 56, "steps": 56 * 2048}
    evidence = {
        "schema_version": 1,
        "kind": "unified_validation_evidence",
        "metrics": metrics,
        "data": data,
        "source": source,
        "checkpoint": cp,
        "configuration": {"values": cfg, "sha256": m.sha(m.canonical(cfg))},
        "inference_protocol": protocol,
        "evaluated_at_utc": "2026-01-01T00:00:00Z",
    }
    qa = {
        "terminal": True,
        "last_epoch": 60,
        "finite_best_last_weights_and_last_optimizer": True,
        "checkpoint_sha256": {"best.pt": "e" * 64, "last.pt": "f" * 64},
        "best_epoch": 56,
        "manifest_sha256": data["manifest_sha256"],
        "canonical_fingerprint": data["prepared_content_sha256"],
        "checkpoint_training_source_sha256": source["training_source_sha256"],
        "eligible_photos": 1,
        "minimum_photo_visits": 60,
        "delivered_samples": 60 * 8192,
    }
    # Exact source recipe strings are copied from user-owned evaluator API,
    # without importing its Torch/model/raw-image dependencies.
    import ast

    text = (ROOT / "scripts/evaluate_original_frozen.py").read_text()
    namespace = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"MODEL_SHA", "POLICY"} for t in node.targets
        ):
            name = node.targets[0].id
            if name == "MODEL_SHA":
                namespace[name] = ast.literal_eval(node.value)
            else:
                namespace[name] = {
                    ast.literal_eval(k): namespace[v.id]
                    if isinstance(v, ast.Name)
                    else ast.literal_eval(v)
                    for k, v in zip(node.value.keys, node.value.values)
                }
    original = dict(
        schema_version=1,
        family="frozen_original_reference",
        preview_only=False,
        split="val",
        **{k: data[k] for k in m.DATA_KEYS},
        inventory_sha256=oi,
        photos_sha256="b" * 64,
        geometry_policy_sha256="c" * 64,
        photos=770,
        per_photo=copy.deepcopy(entries),
        reviewed_policy_sha256=None,
        source={
            "runtime": {"evaluation_source_sha256": "3" * 64, "runtime": {"python": "fixture"}},
            "scripts": {"evaluate_original_frozen.py": "4" * 64},
        },
        policy_spec=namespace["POLICY"],
        evaluated_at_utc="2026-01-01T00:00:00Z",
    )
    original.update({key: namespace["MODEL_SHA"][name] for name, key in m.MODEL_KEYS.items()})
    values = {"reference": original, "candidate": evidence, "completion_qa": qa, "manifest": rows}
    values["reference_context"] = {
        "schema_version": 1,
        "kind": "completed_val_original_context",
        "report_sha256": m.sha(m.canonical(original)),
        "rationale": "synthetic independent review",
        "reviewed_at_utc": "2026-01-02T00:00:00Z",
        "bindings": {
            k: original[k]
            for k in (
                "source",
                "policy_spec",
                *m.DATA_KEYS,
                "inventory_sha256",
                "geometry_policy_sha256",
                "photos_sha256",
            )
        },
    }
    values["candidate_context"] = {
        "schema_version": 1,
        "kind": "completed_val_unified_context",
        "report_sha256": m.sha(m.canonical(evidence)),
        "rationale": "synthetic completed review",
        "reviewed_at_utc": "2026-01-02T00:00:00Z",
        "completion_qa_sha256": m.sha(m.canonical(qa)),
        "bindings": {
            k: evidence[k]
            for k in ("source", "configuration", "inference_protocol", "data", "checkpoint")
        },
    }
    paths = {k: tmp_path / (k + ".json") for k in values}
    for k, p in paths.items():
        p.write_bytes(m.canonical(values[k]))
    return values, paths


def run(case):
    _, paths = case
    return m.compare(
        **paths,
        trusted_reference_context_sha256=m.sha(paths["reference_context"].read_bytes()),
        trusted_candidate_context_sha256=m.sha(paths["candidate_context"].read_bytes()),
        repeats=100,
    )


def test_distinct_frozen_sources_full770_and_completed60(case):
    result = run(case)
    assert result["reference_family"] == "frozen_original_reference"
    assert result["completed_training_epochs"] == 60
    assert result["reference_source"] != result["candidate_source"]
    assert result["metadata_only"] and result["test_evaluation_enabled"] is False


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("reference", "family", "optional_fresh_paired_cup_experiment"),
        ("reference", "split", "test"),
        ("reference", "preview_only", True),
        ("reference", "inventory_sha256", "0" * 64),
        ("reference", "checkpoint_sha256", "0" * 64),
        (
            "candidate",
            "source",
            {"evaluation_source_sha256": "a" * 64, "training_source_sha256": "b" * 64},
        ),
        ("completion_qa", "last_epoch", 59),
        ("completion_qa", "eligible_photos", 2),
    ],
)
def test_modified_reports_fail_external_context_before_comparison(case, target, key, value):
    values, paths = case
    values[target][key] = value
    paths[target].write_bytes(m.canonical(values[target]))
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize(
    "mutation",
    [
        "inventory",
        "coverage",
        "photo",
        "annotation",
        "support",
        "av",
        "config",
        "protocol",
        "source",
    ],
)
def test_rehashed_context_still_cannot_bypass_scientific_contract(case, mutation):
    values, paths = case
    e = values["candidate"]
    q = values["completion_qa"]
    if mutation == "inventory":
        e["data"]["split_inventory_sha256"] = "a" * 64
    if mutation == "coverage":
        q["minimum_photo_visits"] = 59
    if mutation == "photo":
        e["metrics"]["per_photo"].pop()
    if mutation == "annotation":
        e["metrics"]["per_photo"][0]["record_ids"] = []
    if mutation == "support":
        e["metrics"]["per_photo"][0]["channels"]["disc"]["valid_pixels"] = 21
    if mutation == "av":
        e["metrics"]["per_photo"][0]["av_conditional_counts"]["correct"] = 19
    if mutation == "config":
        e["configuration"]["values"]["epochs"] = 59
    if mutation == "protocol":
        e["inference_protocol"]["threshold"] = 0.35
    if mutation == "source":
        e["source"]["evaluation_source_sha256"] = "not-a-digest"
    c = values["candidate_context"]
    c["report_sha256"] = m.sha(m.canonical(e))
    c["completion_qa_sha256"] = m.sha(m.canonical(q))
    c["bindings"] = {k: e[k] for k in c["bindings"]}
    for k in ("candidate", "completion_qa", "candidate_context"):
        paths[k].write_bytes(m.canonical(values[k]))
    with pytest.raises(ValueError):
        run(case)


def test_wrong_external_context_pin_rejected(case):
    _, p = case
    with pytest.raises(ValueError, match="External"):
        m.compare(
            **p,
            trusted_reference_context_sha256="0" * 64,
            trusted_candidate_context_sha256="0" * 64,
        )


def test_old_composed_cup_gate_still_rejects_new_family(case):
    import compare_completed_val as old

    _, p = case
    with pytest.raises(ValueError, match="strongest calibrated"):
        old.compare(
            p["reference"],
            p["candidate"],
            p["manifest"],
            completion_qa_path=p["completion_qa"],
            repeats=100,
        )


def test_no_checkpoint_or_model_module_import_needed():
    import subprocess

    code = "import sys;sys.path.insert(0,'scripts');import compare_frozen_completed_val;assert 'torch' not in sys.modules;assert 'deepretina.unified' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_existing_output_rejected_before_comparison(tmp_path, monkeypatch):
    output = tmp_path / "preserved.json"
    output.write_text("original evidence")
    args = ["compare"]
    for name in [
        "reference",
        "candidate",
        "manifest",
        "completion-qa",
        "reference-context",
        "candidate-context",
    ]:
        args.extend(["--" + name, "absent.json"])
    args.extend(
        [
            "--trusted-reference-context-sha256",
            "a" * 64,
            "--trusted-candidate-context-sha256",
            "b" * 64,
            "--output",
            str(output),
        ]
    )
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        m, "compare", lambda **kw: pytest.fail("Must reject existing output before comparison")
    )
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "original evidence"


def test_input_change_during_comparison_is_rejected(case, monkeypatch):
    _, paths = case
    actual = m.compare_reports

    def mutate(*args, **kwargs):
        result = actual(*args, **kwargs)
        paths["reference"].write_bytes(b"changed during comparison")
        return result

    monkeypatch.setattr(m, "compare_reports", mutate)
    with pytest.raises(ValueError, match="Inputs changed"):
        run(case)
