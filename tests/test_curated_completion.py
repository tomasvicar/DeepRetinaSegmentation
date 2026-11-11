"""Synthetic tiny constructed weights + metadata; no real weights/GT/NN forward."""

import copy
import io
import json
from pathlib import Path

import pytest
import torch

from deepretina import curated_completion as m
from deepretina import selection as gate


def blob(value):
    if isinstance(value, bytes):
        return value
    return gate.canonical(value) + b"\n"


def save_weights(value):
    f = io.BytesIO()
    torch.save(value, f)
    return f.getvalue()


def bind(case):
    v = case["values"]
    raw = {
        k: (save_weights(x) if k in ("best", "last") else blob(x))
        for k, x in v.items()
        if k not in ("copy", "qa60", "receipt60", "receipt27", "task60", "task27")
    }
    arm = case["arm"]
    study = v["study"]
    if case.get("original_pretty_json"):
        raw["study"] = (json.dumps(study, indent=2) + "\n").encode()
    config = study["arms"][arm]["config"]
    ph = m.portable(v["profile"]["photometry_cache"])
    parent_sha = gate.digest(raw["parent"])
    helper_sha = gate.digest(raw["helpers"])
    study_sha = m.digest(study)
    v["profile"].update(
        study_contract_sha256=study_sha,
        reviewed_helper_manifest_sha256=helper_sha,
        parent_tpe_manifest_sha256=parent_sha,
    )
    raw["profile"] = (
        (json.dumps(v["profile"], indent=2) + "\n").encode()
        if case.get("original_pretty_json")
        else blob(v["profile"])
    )
    profile_sha = m.digest(v["profile"])
    v["identity60"]["profile_sha256"] = profile_sha
    raw["identity60"] = blob(v["identity60"])
    plan = {
        "candidates": {arm: config},
        "derived_cache": m.RAW_CACHE,
        "training_source_sha256": m.SOURCE["training_sha256"],
        "manifest_sha256": m.MANIFEST,
        "prepared_content_sha256": m.CONTENT,
        "geometry_policy_sha256": m.GEOMETRY,
        "photometry_caches": {
            a["config"]["photometric_profile"]: m.portable(a["photometry_cache"])
            for a in study["arms"].values()
            if a["photometry_cache"] is not None
        },
    }
    for key, epoch in [("task60", 60), ("task27", 27)]:
        v[key] = {
            "trial": arm,
            "source_commit": m.SOURCE["commit"],
            "execution_code_sha256": m.SOURCE["execution_sha256"],
            "study_kind": "curated_patch_photometry_v1",
            "target_epoch": epoch,
            "previous_epoch": 27 if epoch == 60 else 0,
            "resume": epoch == 60,
            "plan": plan,
            "plan_sha256": m.digest(plan),
            "config_sha256": m.digest(config),
            "study_contract_sha256": study_sha,
            "parent_tpe_manifest_sha256": parent_sha,
            "photometry_cache": v["profile"]["photometry_cache"],
            "gpu_uuid": "syntheticGPU",
            "profile_sha256": profile_sha,
            "worker_id": "syntheticworker",
            "input_artifact_shas": {
                "profile": gate.digest(raw["profile"]),
                "study": gate.digest(raw["study"]),
                "parent_tpe": gate.digest(raw["parent"]),
                "helpers": gate.digest(raw["helpers"]),
            },
        }
        raw[key] = blob(v[key])
    proof = {
        "status": "independent_copy_sha_verified_readonly",
        "hardlinks_used": False,
        "sha256": {name: gate.digest(raw[key]) for name, key in m.ARTIFACTS.items()},
    }
    v["copy"] = proof
    raw["copy"] = blob(proof)
    best, last = v["best"], v["last"]
    records = {r["id"] for r in case["rows"] if r["split"] == "train"}

    def coverage(p):
        return {
            "epoch": p["epoch"],
            "contributing_annotation_records": len(p["coverage"]["annotation_visits"]),
            "eligible_annotation_records": 5700,
            "all_annotations_contributed": set(p["coverage"]["annotation_visits"]) == records,
        }

    qa = {
        "kind": "curated_unified_rung_evidence_v1",
        "status": "PASS",
        "actual_epoch": 60,
        "completed_budget": True,
        "scheduler_T_max": 60,
        "scheduler_last_epoch": 60,
        "steps": 122880,
        "delivered_samples": 491520,
        "finite_weights": True,
        "finite_optimizer": True,
        "source_commit": m.SOURCE["commit"],
        "execution_code_sha256": m.SOURCE["execution_sha256"],
        "training_source_sha256": m.SOURCE["training_sha256"],
        "independent_milestone": proof,
        "best_epoch": best["epoch"],
        "metric": best["val_selection_score"],
        "checkpoint_sha256": gate.digest(raw["last"]),
        "best_checkpoint_sha256": gate.digest(raw["best"]),
        "last_validation_metrics": last["validation_metrics"],
        "best_checkpoint_coverage": coverage(best),
        "terminal_checkpoint_coverage": coverage(last),
        "selected_best_full_annotations": coverage(best)["all_annotations_contributed"],
        "photometry_cache": ph,
        "gpu_uuid": "syntheticGPU",
        "worker_id": "syntheticworker",
        "manifest_sha256": m.MANIFEST,
        "canonical_sha256": m.CONTENT,
        "cache_sha256": m.RAW_CACHE["fingerprint"],
        **{k: v["task60"][k] for k in ["plan_sha256", "config_sha256", "profile_sha256"]},
    }
    v["qa60"] = qa
    raw["qa60"] = blob(qa)
    v["receipt60"] = {
        "kind": "curated_pbs_stage_receipt_v1",
        "status": "COMPLETE",
        "task": v["task60"],
        "run_result": v["run_result60"],
        "checkpoint_qa": dict(
            qa,
            host_independent_milestone_sha256_verified=True,
            original_cpu_qa_path="original/qa.json",
            original_cpu_qa_sha256=gate.digest(raw["qa60"]),
            original_copy_verification_sha256=gate.digest(raw["copy"]),
            qa_returncode=0,
        ),
    }
    v["receipt27"] = {
        "kind": "curated_pbs_stage_receipt_v1",
        "status": "COMPLETE",
        "task": v["task27"],
        "run_result": v["run_result60"],
        "checkpoint_qa": {
            "kind": "curated_unified_rung_evidence_v1",
            "status": "PASS",
            "actual_epoch": 27,
            "host_independent_milestone_sha256_verified": True,
        },
    }
    v["marker"] = {
        "plan_sha256": v["task60"]["plan_sha256"],
        "trial": arm,
        "execution_code_sha256": m.SOURCE["execution_sha256"],
        "manifest_sha256": m.MANIFEST,
    }
    raw["marker"] = blob(v["marker"])
    proof["sha256"]["search_trial.json"] = gate.digest(raw["marker"])
    # Rebind copy/QA receipt hashes after marker finalization.
    raw["copy"] = blob(proof)
    v["qa60"]["independent_milestone"] = proof
    raw["qa60"] = blob(v["qa60"])
    v["receipt60"]["checkpoint_qa"].update(
        independent_milestone=proof,
        original_cpu_qa_sha256=gate.digest(raw["qa60"]),
        original_copy_verification_sha256=gate.digest(raw["copy"]),
    )
    raw.update({k: blob(v[k]) for k in ["receipt60", "receipt27"]})
    case["raw"] = raw
    return write(case)


def write(case):
    files = {}
    for k, data in case["raw"].items():
        p = case["root"] / (k + (".pt" if k in ("best", "last") else ".json"))
        p.write_bytes(data)
        files[k] = {"path": str(p), "sha256": gate.digest(data)}
    index = {"schema_version": 1, "kind": "original_curated_completion_inputs", "files": files}
    path = case["root"] / "index.json"
    path.write_bytes(blob(index))
    case["index"] = path
    return {
        "original_index": path,
        "expected_index_sha256": gate.digest(path.read_bytes()),
        "manifest": case["manifest"],
        "checkpoint_role": "training_best",
        "family": "curated_fd44",
    }


@pytest.fixture
def case(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *a, **k: pytest.fail("No NN forward"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA"))
    rows = []
    for split, n in [("train", 2529), ("val", 770), ("test", 856)]:
        for i in range(5700 if split == "train" else n):
            photo = f"{split}-p{i % n}"
            rows.append(
                {
                    "id": f"{split}-a{i}",
                    "photo_id": photo,
                    "source_id": photo,
                    "split_group_id": photo,
                    "dataset": f"s{i % 21}",
                    "split": split,
                    "task": "av",
                    "preprocessing_profile": "legacy25_raw",
                    "pixels_per_degree": 25,
                    "preparation_max_size": None,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(blob(rows))
    monkeypatch.setattr(m, "MANIFEST", gate.digest(manifest.read_bytes()))
    config = {
        "architecture": "compact_unet",
        "base_channels": 2,
        "depth": 1,
        "norm": "batch",
        "epochs": 60,
        "epoch_samples": 8192,
        "batch_size": 4,
        "patch_size": 384,
        "photometric_profile": "raw",
        "selection_metric": "macro_dataset_mean_dice",
    }
    parent = {
        "full_future60_config": {k: v for k, v in config.items() if k != "photometric_profile"}
    }
    helpers = {"synthetic": "fixture only"}
    monkeypatch.setattr(m, "PARENT", gate.digest(blob(parent)))
    monkeypatch.setattr(m, "HELPERS", gate.digest(blob(helpers)))
    monkeypatch.setattr(m, "IMAGE", "synthetic-image")
    study = {
        "kind": "curated_patch_photometry_v1",
        "source": m.SOURCE,
        "baseline_config": config,
        "arms": {
            arm: {
                "config": dict(config, patch_size=patch, photometric_profile=photo),
                "photometry_cache": None,
            }
            for arm, (patch, photo) in m.ARMS.items()
        },
        "photometric_descriptors": {"raw": {"profile": "raw"}},
    }
    monkeypatch.setattr(m, "STUDIES", {m.digest(study)})
    groups = {}
    for row in rows[:5700]:
        groups.setdefault(row["photo_id"], []).append(row)

    def coverage(epoch):
        visits = dict.fromkeys(groups, epoch)
        visits[next(iter(visits))] += epoch * 8192 - sum(visits.values())
        return {
            "eligible_photos": 2529,
            "seen_photos": 2529,
            "photo_visits": visits,
            "eligible_annotation_records": 5700,
            "contributing_annotation_records": 5700,
            "annotation_visits": {r["id"]: 1 for r in rows[:5700]},
        }

    cfg = study["arms"]["raw256"]["config"]
    model = m.UNet(5, 2, 1, "batch")
    state = copy.deepcopy(model.state_dict())
    common = {
        "format_version": 1,
        "family": "unified",
        "config": cfg,
        "channels": list(gate.CHANNELS),
        "state_dict": state,
        "architecture": {
            "name": "compact_unet",
            "base_channels": 2,
            "depth": 1,
            "normalization": "batch",
            "outputs": 5,
        },
        "training_source_sha256": m.SOURCE["training_sha256"],
        "manifest_sha256": m.MANIFEST,
        "prepared_content_sha256": m.CONTENT,
        "geometry_policy": {"sha256": m.GEOMETRY},
        "purpose": "full_data",
        "normalization": "RGB/255-0.5",
        "probability_fusion": "joint_av_mixture_then_conditional",
        "inference_max_size": None,
        "supervised_channels": [True] * 5,
        "lineage": {
            "initialization": "scratch",
            "historical_exposure": False,
            "independent_model_eligible": True,
        },
        "derived_cache": m.RAW_CACHE,
        "preprocessing": {
            "profile": "legacy25_raw",
            "pixels_per_degree": 25.0,
            "normalized": False,
        },
    }
    best = dict(
        common,
        epoch=52,
        steps=52 * 2048,
        val_selection_score=0.9,
        validation_metrics={"macro_dataset_mean_dice": 0.9},
        coverage=coverage(52),
    )
    last = dict(
        common,
        epoch=60,
        steps=122880,
        val_selection_score=0.8,
        validation_metrics={"macro_dataset_mean_dice": 0.8},
        coverage=coverage(60),
    )
    optimizer = {
        "state": {
            i: {
                "step": torch.tensor(122880.0),
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }
            for i, p in enumerate(model.parameters())
        },
        "param_groups": [{"params": list(range(len(list(model.parameters()))))}],
    }
    last["training_state"] = {
        "optimizer": optimizer,
        "scheduler": {"T_max": 60, "last_epoch": 60},
        "rng": {
            k: torch.ones(2, dtype=torch.uint8)
            for k in ["torch", "python", "numpy", "sampler", "augmentation", "cuda"]
        },
        "coverage": last["coverage"],
        "best_validation_score": 0.9,
    }
    csv = "epoch,steps,val_selection_score\n" + "".join(
        f"{i},{i * 2048},{0.9 if i == 52 else 0.8}\n" for i in range(1, 61)
    )
    runtime = {
        "status": "PASS",
        "gpu_uuid": "syntheticGPU",
        "expected_gpu_uuid": "syntheticGPU",
        "pbs_job_id": "synthetic-job",
    }
    runresult = {
        "returncode": 0,
        "observed_returncode": True,
        "timed_out": False,
        "runtime": runtime,
    }
    values = {
        "best": best,
        "last": last,
        "parent": parent,
        "helpers": helpers,
        "study": study,
        "profile": {
            "source_identity": m.SOURCE,
            "arm_id": "raw256",
            "execution_authorized": True,
            "photometry_cache": None,
            "source": "/synthetic-source",
            "image_sha256": "synthetic-image",
        },
        "runtime60": runtime,
        "run_result60": runresult,
        "identity60": {
            "source_commit": m.SOURCE["commit"],
            "gpu_uuid": "syntheticGPU",
            "pbs_job_id": "synthetic-job",
            "profile_sha256": "REBOUND",
            "worker_id": "syntheticworker",
        },
        "terminal": {
            "kind": "curated_pbs_terminal_v1",
            "status": "completed_reviewed_stages",
            "actual_epoch": 60,
            "completed_budget": True,
        },
        "cleanup": {"status": "owned_allowlisted_scratch_contents_removed"},
        "pbs": {
            "returncode": 0,
            "PBS": {"job_state": "F", "Exit_status": 0},
            "command": ["qstat", "-x", "-f", "-F", "json", "synthetic-job"],
        },
        "csv": csv.encode(),
        "coverage": last["coverage"],
        "validation": best["validation_metrics"],
        "run": {
            "config": cfg,
            "training_source_sha256": m.SOURCE["training_sha256"],
            "manifest_sha256": m.MANIFEST,
            "prepared_content_sha256": m.CONTENT,
            "derived_cache": m.RAW_CACHE,
        },
        "summary": {
            "completed_budget": True,
            "steps": 122880,
            "best_validation_score": 0.9,
            "selection_metric": "macro_dataset_mean_dice",
        },
        "marker": {"synthetic": True},
        "source_attestation": {
            "kind": "curated_host_source_attestation_v1",
            "bindings": {
                "container_source": "/workspace",
                "host_source": "/synthetic-source",
                "source_commit": m.SOURCE["commit"],
                "execution_sha256": m.SOURCE["execution_sha256"],
                "training_sha256": m.SOURCE["training_sha256"],
                "image_sha256": "synthetic-image",
                "git_status_porcelain": "",
            },
        },
    }
    case = {"root": tmp_path, "values": values, "rows": rows, "manifest": manifest, "arm": "raw256"}
    case["args"] = bind(case)
    return case


@pytest.mark.parametrize("role", ["training_best", "fixed_terminal60"])
def test_tiny_synthetic_own_role_positive_no_forward(case, role):
    args = dict(case["args"], checkpoint_role=role)
    proof = m.check_curated_completion(**args).proof
    assert proof["checkpoint_epoch"] == (52 if role == "training_best" else 60)
    assert proof["selected_checkpoint_coverage"]["contributing_annotation_records"] == 5700
    assert proof["selection_created"] is False and proof["test_evaluation_enabled"] is False


@pytest.mark.parametrize(
    "bad",
    [
        "zero_BEST",
        "foreign_BEST",
        "partial_LAST",
        "stale_optimizer",
        "wrong_horizon",
        "nonfinite",
        "bad_shape",
    ],
)
def test_own_weight_and_budget_negative_with_rebound_provenance(case, bad):
    best, last = case["values"]["best"], case["values"]["last"]
    if bad == "zero_BEST":
        best["coverage"]["annotation_visits"]["train-a0"] = 0
    if bad == "foreign_BEST":
        best["coverage"]["annotation_visits"]["foreign"] = best["coverage"][
            "annotation_visits"
        ].pop("train-a0")
    if bad == "partial_LAST":
        last["epoch"] = 27
        last["steps"] = 55296
    if bad == "stale_optimizer":
        next(iter(last["training_state"]["optimizer"]["state"].values()))["step"] = torch.tensor(
            55296.0
        )
    if bad == "wrong_horizon":
        last["training_state"]["scheduler"]["T_max"] = 27
    if bad == "nonfinite":
        next(iter(best["state_dict"].values())).flatten()[0] = float("nan")
    if bad == "bad_shape":
        key = next(iter(best["state_dict"]))
        best["state_dict"][key] = torch.zeros(1)
    args = bind(case)
    with pytest.raises((gate.SelectionError, RuntimeError)):
        m.check_curated_completion(**args)


def test_partial_BEST_cannot_borrow_LAST_but_explicit_LAST_still_valid(case):
    p = case["values"]["best"]
    p["coverage"]["annotation_visits"].pop("train-a0")
    p["coverage"]["contributing_annotation_records"] = 5699
    args = bind(case)
    with pytest.raises(gate.SelectionError, match="OWN weights"):
        m.check_curated_completion(**args)
    assert (
        m.check_curated_completion(**dict(args, checkpoint_role="fixed_terminal60")).proof[
            "checkpoint_epoch"
        ]
        == 60
    )


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("task60", "source_commit", "retagged72"),
        ("task60", "plan_sha256", "0" * 64),
        ("task60", "photometry_cache", {"fallback": True}),
        ("receipt60", "status", "FAILED"),
        ("pbs", "PBS", {"job_state": "R", "Exit_status": 0}),
        ("run_result60", "observed_returncode", False),
        ("study", "kind", "v3_protocol"),
    ],
)
def test_original_record_changes_rejected_after_index_repin(case, name, field, value):
    record = copy.deepcopy(gate.parse_json(case["raw"][name]))
    record[field] = value
    case["raw"][name] = blob(record)
    args = write(case)
    with pytest.raises((gate.SelectionError, KeyError)):
        m.check_curated_completion(**args)


def test_external_index_pin_required(case):
    with pytest.raises(gate.SelectionError, match="Externally pinned"):
        m.check_curated_completion(**dict(case["args"], expected_index_sha256="0" * 64))


def prepare_arm(case, arm, monkeypatch):
    v = case["values"]
    study = v["study"]
    case["arm"] = arm
    v["profile"]["arm_id"] = arm
    config = study["arms"][arm]["config"]
    photo = config["photometric_profile"]
    cache = None
    if photo != "raw":
        descriptor = {"profile": photo, "backend_versions": {"synthetic": "fixture-only"}}
        study["photometric_descriptors"][photo] = descriptor
        fp, _ = m.PHOTO_CACHE[photo]
        monkeypatch.setitem(m.PHOTO_CACHE, photo, (fp, m.digest(descriptor)))
        cache = {
            "schema": 1,
            "kind": "full_prepared_photometry_RGB",
            "fingerprint": fp,
            "canonical_fingerprint": m.CONTENT,
            "descriptor_sha256": m.digest(descriptor),
            "operator_source_sha256": "196da8626c87d6b07bed6508252dd37328ee79846c4040c1403f9a090002708e",
            "coverage": "train_val",
            "fallback": False,
        }
        study["arms"][arm]["photometry_cache"] = cache
    v["profile"]["photometry_cache"] = cache
    for name in ["best", "last"]:
        v[name]["config"] = copy.deepcopy(config)
        if cache is not None:
            v[name]["photometry_cache"] = copy.deepcopy(cache)
            v[name]["preprocessing"] = dict(
                v[name]["preprocessing"],
                photometry=copy.deepcopy(study["photometric_descriptors"][photo]),
            )
    v["run"]["config"] = config
    if cache is not None:
        v["run"]["photometry_cache"] = cache
    monkeypatch.setattr(m, "STUDIES", {m.digest(study)})
    return bind(case)


@pytest.mark.parametrize("arm", ["raw384", "raw512", "clahe384", "lc384"])
def test_all_other_declared_arms_keep_full_config_and_cache(case, monkeypatch, arm):
    args = prepare_arm(case, arm, monkeypatch)
    proof = m.check_curated_completion(**args).proof
    assert proof["raw_training_configuration"]["photometric_profile"] == m.ARMS[arm][1]
    assert proof["raw_training_configuration"]["patch_size"] == m.ARMS[arm][0]
    assert (
        proof["TTA_admission_enabled"] is False
        and proof["fresh_sourcecompatible_VAL_required"] is True
    )


@pytest.mark.parametrize("bad", ["cache_fallback", "descriptor_backend", "field_stripping"])
def test_saved_nonraw_identity_cannot_be_silently_replaced(case, monkeypatch, bad):
    prepare_arm(case, "clahe384", monkeypatch)
    best = case["values"]["best"]
    if bad == "cache_fallback":
        best["photometry_cache"]["fallback"] = True
    if bad == "descriptor_backend":
        best["preprocessing"]["photometry"]["backend_versions"]["synthetic"] = "changed"
    if bad == "field_stripping":
        best["config"].pop("photometric_profile")
    with pytest.raises(gate.SelectionError):
        m.check_curated_completion(**bind(case))


def test_final_adapter_requires_distinct_original_index_before_reading_inputs():
    from deepretina.completion_profiles import PROFILES
    from deepretina.freeze_eligibility import check_freeze_eligibility

    assert PROFILES[m.FAMILY][3] == m.SOURCE["training_sha256"]
    missing = Path("must-not-be-read")
    with pytest.raises(gate.SelectionError, match="requires original curated index"):
        check_freeze_eligibility(
            checkpoint=missing,
            best_checkpoint=missing,
            last_checkpoint=missing,
            manifest=missing,
            metrics_csv=missing,
            training_summary=missing,
            completion_qa=missing,
            expected_qa_sha256="0" * 64,
            family=m.FAMILY,
            checkpoint_role="training_best",
        )


def test_original_raw_dependency_bytes_and_canonical_profile_are_separate(case):
    case["original_pretty_json"] = True
    result = m.check_curated_completion(**bind(case))
    task = gate.parse_json(case["raw"]["task60"])
    assert task["input_artifact_shas"]["profile"] != task["profile_sha256"]
    assert task["input_artifact_shas"]["study"] != task["study_contract_sha256"]
    assert result.proof["checkpoint_epoch"] == 52


def test_raw_dependency_cannot_be_replaced_with_canonical_hash(case):
    case["original_pretty_json"] = True
    bind(case)
    task = gate.parse_json(case["raw"]["task60"])
    task["input_artifact_shas"]["profile"] = task["profile_sha256"]
    case["raw"]["task60"] = blob(task)
    with pytest.raises(gate.SelectionError, match="frozen input dependency"):
        m.check_curated_completion(**write(case))


# The existing fixture executes the genuine checker over tiny constructed weights.
# Canonical asset traversal is mocked here: this task performs no real GT decode.
def final_args(case, monkeypatch, role="training_best"):
    from deepretina.freeze_eligibility import check_freeze_eligibility

    index = gate.parse_json(case["index"].read_bytes())["files"]
    monkeypatch.setattr(gate, "_canonical_metadata", lambda manifest: {"synthetic": True})
    monkeypatch.setattr(gate, "data_identity", lambda manifest, selected: ({"synthetic": True}, {}))
    monkeypatch.setattr(gate, "_finish_canonical_read", lambda *args: None)
    args = {
        "checkpoint": index["best" if role == "training_best" else "last"]["path"],
        "best_checkpoint": index["best"]["path"],
        "last_checkpoint": index["last"]["path"],
        "manifest": case["manifest"],
        "metrics_csv": index["csv"]["path"],
        "training_summary": index["summary"]["path"],
        "completion_qa": index["qa60"]["path"],
        "expected_qa_sha256": index["qa60"]["sha256"],
        "family": m.FAMILY,
        "checkpoint_role": role,
        "original_curated_index": case["index"],
        "expected_curated_index_sha256": gate.digest(case["index"].read_bytes()),
    }
    return check_freeze_eligibility, args


@pytest.mark.parametrize("pretty", [False, True])
@pytest.mark.parametrize("role", ["training_best", "fixed_terminal60"])
def test_final_adapter_preserves_original_chain_and_own_role(case, monkeypatch, pretty, role):
    case["original_pretty_json"] = pretty
    case["args"] = bind(case)
    operation, args = final_args(case, monkeypatch, role)
    original = m.check_curated_completion(**dict(case["args"], checkpoint_role=role)).proof
    result = operation(**args)
    assert result.proof["original_curated_completion"] == original
    assert result.proof["raw_training_configuration"] == case["values"]["best"]["config"]
    assert result.proof["checkpoint_epoch"] == (52 if role == "training_best" else 60)
    assert result.proof["source_profile"]["training_source_sha256"] == m.SOURCE["training_sha256"]
    assert result.proof["selection_created"] is False
    assert result.proof["test_evaluation_enabled"] is False


@pytest.mark.parametrize(
    "key",
    [
        "checkpoint",
        "best_checkpoint",
        "last_checkpoint",
        "metrics_csv",
        "training_summary",
        "completion_qa",
    ],
)
def test_final_adapter_rejects_foreign_path_even_identical_bytes(case, monkeypatch, key):
    operation, args = final_args(case, monkeypatch)
    foreign = case["root"] / ("foreign-" + key)
    foreign.write_bytes(Path(args[key]).read_bytes())
    args[key] = foreign
    with pytest.raises(gate.SelectionError, match="path differs"):
        operation(**args)


@pytest.mark.parametrize("key", ["expected_qa_sha256", "expected_curated_index_sha256"])
def test_final_adapter_requires_both_external_pins(case, monkeypatch, key):
    operation, args = final_args(case, monkeypatch)
    args[key] = "0" * 64
    with pytest.raises(gate.SelectionError):
        operation(**args)


def test_final_adapter_rejects_wrong_role_and_native_envelope(case, monkeypatch):
    operation, args = final_args(case, monkeypatch)
    with pytest.raises(gate.SelectionError, match="path differs"):
        operation(**dict(args, checkpoint_role="fixed_terminal60"))
    with pytest.raises(gate.SelectionError, match="cannot use standalone/native"):
        operation(**dict(args, original_pool_evidence="foreign"))


def test_final_adapter_partial_BEST_cannot_borrow_LAST(case, monkeypatch):
    case["values"]["best"]["coverage"]["annotation_visits"].pop("train-a0")
    case["values"]["best"]["coverage"]["contributing_annotation_records"] = 5699
    bind(case)
    operation, args = final_args(case, monkeypatch)
    with pytest.raises(gate.SelectionError):
        operation(**args)


@pytest.mark.parametrize("family", ["standalone_0b", "standalone_72", "pool_6b", "pool_72_v3"])
def test_old_family_rejects_curated_index_before_any_read(family):
    from deepretina.freeze_eligibility import check_freeze_eligibility

    with pytest.raises(gate.SelectionError, match="require the explicit curated_fd44 family"):
        check_freeze_eligibility(
            checkpoint="unread",
            best_checkpoint="unread",
            last_checkpoint="unread",
            manifest="unread",
            metrics_csv="unread",
            training_summary="unread",
            completion_qa="unread",
            expected_qa_sha256="0" * 64,
            family=family,
            checkpoint_role="training_best",
            original_curated_index="foreign-index",
            expected_curated_index_sha256="a" * 64,
        )


def test_final_adapter_rejects_original_chain_drift_after_checker(case, monkeypatch):
    operation, args = final_args(case, monkeypatch)

    def changed(manifest, payload):
        Path(args["metrics_csv"]).write_bytes(b"changed after original completion checker")
        return {"synthetic": True}, {}

    monkeypatch.setattr(gate, "data_identity", changed)
    with pytest.raises(gate.SelectionError, match="chain changed"):
        operation(**args)
