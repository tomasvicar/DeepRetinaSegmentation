"""Independent curated_fd44 completion gate; no inference/selection/TEST permit.

This family preserves original curated records; it is neither nativeV3 nor a
standalone supervisor. Full configuration is retained, never filtered to fit an
older loader. Only the known compact architecture is constructed on CPU, without
forward, after exact study/config admission.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import selection as gate
from .freeze_eligibility import _coverage, _groups, _pinned_json
from .model import UNet

FAMILY = "curated_fd44"
SOURCE = {
    "commit": "fd44cdfa83d1df9ccc42960df6d1d2018db29900",
    "execution_sha256": "fcb186c98b812e9ead0fe19bec9f7e06063d31a7c7c020a785169b6c13c761ee",
    "training_sha256": "2b7076759c8fab1594e5108bcbf348888007a94e06c4d21d4431dbfb31057747",
}
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
GEOMETRY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
RAW_CACHE = {
    "schema": 1,
    "fingerprint": "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e",
    "canonical_fingerprint": CONTENT,
    "allow_canonical_fallback": False,
}
IMAGE = "7a30b914ec3bb17c3a548932e8940f6e4f5c61824b34ca9b30ce1b83f98343b7"
PARENT = "89f304fa69749c25999f06d70638ee4e011c7066223fe03478f1dad258affaef"
HELPERS = "177c53bbcd1d99f49d639db7270d30c0ed10c82421d9a250715282c94d051cc9"
STUDIES = {
    "1936a0c67af430058a74a973415e1058f7edf9af2b590e4a9b7fb2f9777b818b",
    "58d5d851281d579c6dcddee535c689eab83e8c090e684674688dabaa0d5f06de",
    "e7a2b3066e32fb91dbbde3d1504181762ad7eb654304be3a2909b389a4cd5c89",
}
ARMS = {
    "raw256": (256, "raw"),
    "raw384": (384, "raw"),
    "raw512": (512, "raw"),
    "clahe384": (384, "clahe_hsv_v1"),
    "lc384": (384, "legacy_lc_clahe_v1"),
}
PHOTO_CACHE = {
    "clahe_hsv_v1": (
        "b5e781330d2e28b65d70301384fe1e841b64f0e8f631fe04dac3381847664ec2",
        "ae74060cbab113faba337521b4c42973f3008346001f772a35f781cb7b7e01c6",
    ),
    "legacy_lc_clahe_v1": (
        "b2e42864c0e53b8fec30f55114e30f1ebbb624dd7fbefeccd31c7206cc324b01",
        "c86d71a2221ed77cf3be6ed159a3aaa321fbb0aa51a9eb27888d9af476070572",
    ),
}
CACHE_KEYS = (
    "schema",
    "kind",
    "fingerprint",
    "canonical_fingerprint",
    "descriptor_sha256",
    "operator_source_sha256",
    "coverage",
    "fallback",
)
ARTIFACTS = {
    "best.pt": "best",
    "last.pt": "last",
    "metrics.csv": "csv",
    "coverage.json": "coverage",
    "validation.json": "validation",
    "run.json": "run",
    "training_summary.json": "summary",
    "search_trial.json": "marker",
}
FILES = set(ARTIFACTS.values()) | {
    "receipt60",
    "receipt27",
    "task60",
    "task27",
    "qa60",
    "run_result60",
    "runtime60",
    "identity60",
    "study",
    "profile",
    "parent",
    "helpers",
    "terminal",
    "cleanup",
    "pbs",
    "copy",
    "source_attestation",
}


def same(a, b, label):
    gate.require(gate.canonical(a) == gate.canonical(b), label)


def digest(value):
    return gate.digest(gate.canonical(value) + b"\n")


def portable(cache):
    return None if cache is None else {k: cache[k] for k in CACHE_KEYS}


def study_config(records, raw):
    study, parent, profile = records["study"], records["parent"], records["profile"]
    gate.require(
        gate.digest(raw["parent"]) == PARENT and gate.digest(raw["helpers"]) == HELPERS,
        "Original parent/helper dependency bytes differ",
    )
    gate.require(
        digest(study) in STUDIES and study["kind"] == "curated_patch_photometry_v1",
        "Unreviewed curated study",
    )
    same(study["source"], SOURCE, "Curated source profile differs")
    same(
        study["baseline_config"],
        dict(parent["full_future60_config"], photometric_profile="raw"),
        "Full parent training configuration differs",
    )
    gate.require(set(study["arms"]) == set(ARMS), "Exact five-arm study required")
    for name, (patch, photo) in ARMS.items():
        same(
            study["arms"][name]["config"],
            dict(study["baseline_config"], patch_size=patch, photometric_profile=photo),
            "Unknown arm configuration/extra or erased field",
        )
    arm = records["task60"]["trial"]
    gate.require(
        arm in ARMS and profile["arm_id"] == arm and profile["execution_authorized"] is True,
        "Original authorized arm/profile differs",
    )
    same(profile["source_identity"], SOURCE, "Profile source differs")
    same(
        profile["photometry_cache"],
        study["arms"][arm]["photometry_cache"],
        "Profile photometry cache differs",
    )
    gate.require(profile["image_sha256"] == IMAGE, "Original SIF environment identity differs")
    config = study["arms"][arm]["config"]
    photo = config["photometric_profile"]
    cache = portable(profile["photometry_cache"])
    if photo == "raw":
        gate.require(cache is None, "Raw must not silently use a photometry cache")
    else:
        fp, descriptor = PHOTO_CACHE[photo]
        same(
            cache,
            {
                "schema": 1,
                "kind": "full_prepared_photometry_RGB",
                "fingerprint": fp,
                "canonical_fingerprint": CONTENT,
                "descriptor_sha256": descriptor,
                "operator_source_sha256": "196da8626c87d6b07bed6508252dd37328ee79846c4040c1403f9a090002708e",
                "coverage": "train_val",
                "fallback": False,
            },
            "Exact admitted nonraw immutable cache differs",
        )
        gate.require(
            digest(study["photometric_descriptors"][photo]) == descriptor,
            "Saved photometric backend/descriptor differs",
        )
    return arm, config, cache


def original_chain(r, raw):
    arm, config, cache = study_config(r, raw)
    t, t27, receipt, qa = r["task60"], r["task27"], r["receipt60"], r["qa60"]
    for task, epoch in ((t, 60), (t27, 27)):
        gate.require(
            task["source_commit"] == SOURCE["commit"]
            and task["execution_code_sha256"] == SOURCE["execution_sha256"]
            and task["study_kind"] == "curated_patch_photometry_v1",
            "Task source/family differs",
        )
        gate.require(
            task["target_epoch"] == epoch
            and task["previous_epoch"] == (27 if epoch == 60 else 0)
            and task["resume"] is (epoch == 60),
            "Original actual27-to60 task required",
        )
        same(task["plan"]["candidates"], {arm: config}, "Task full configuration differs")
        gate.require(
            task["plan_sha256"] == digest(task["plan"])
            and task["config_sha256"] == digest(config)
            and task["study_contract_sha256"] == digest(r["study"]),
            "Task study/plan/config hashes differ",
        )
        gate.require(
            task["parent_tpe_manifest_sha256"] == PARENT,
            "Wrong original parameter-selection parent",
        )
        same(task["photometry_cache"], r["profile"]["photometry_cache"], "Task cache differs")
        same(task["plan"]["derived_cache"], RAW_CACHE, "Raw derived cache differs")
        gate.require(
            task["plan"]["training_source_sha256"] == SOURCE["training_sha256"]
            and task["plan"]["manifest_sha256"] == MANIFEST
            and task["plan"]["prepared_content_sha256"] == CONTENT
            and task["plan"]["geometry_policy_sha256"] == GEOMETRY,
            "Task canonical/source identity differs",
        )
    for k in (
        "config_sha256",
        "plan_sha256",
        "gpu_uuid",
        "profile_sha256",
        "worker_id",
        "study_contract_sha256",
    ):
        same(t[k], t27[k], "Original same-worker/config/GPU resume differs")
    same(
        t["input_artifact_shas"],
        {k: gate.digest(raw[k]) for k in ("profile", "study", "parent", "helpers")},
        "Original frozen input dependency hashes differ",
    )
    gate.require(t["profile_sha256"] == gate.digest(raw["profile"]), "Profile bytes differ")
    same(receipt["task"], t, "E60 receipt task differs")
    same(receipt["run_result"], r["run_result60"], "Original observed child result differs")
    for rec, epoch in ((receipt, 60), (r["receipt27"], 27)):
        gate.require(
            rec["kind"] == "curated_pbs_stage_receipt_v1" and rec["status"] == "COMPLETE",
            "Original curated COMPLETE receipt required",
        )
        result = rec["run_result"]
        gate.require(
            type(result["returncode"]) is int
            and result["returncode"] == 0
            and result["observed_returncode"] is True
            and result["timed_out"] is False,
            "Actual child exit0 required, not scheduler/logical-only success",
        )
        gate.require(
            rec["checkpoint_qa"]["kind"] == "curated_unified_rung_evidence_v1"
            and rec["checkpoint_qa"]["status"] == "PASS"
            and rec["checkpoint_qa"]["actual_epoch"] == epoch
            and rec["checkpoint_qa"]["host_independent_milestone_sha256_verified"] is True,
            "Genuine curated HOST milestone QA required",
        )
    same(r["receipt27"]["task"], t27, "E27 receipt task differs")
    overlay = {
        "host_independent_milestone_sha256_verified",
        "original_cpu_qa_path",
        "original_cpu_qa_sha256",
        "original_copy_verification_sha256",
        "qa_returncode",
    }
    same(
        {k: v for k, v in receipt["checkpoint_qa"].items() if k not in overlay},
        qa,
        "Raw curated QA was retagged/changed",
    )
    host = receipt["checkpoint_qa"]
    gate.require(
        host["original_cpu_qa_sha256"] == gate.digest(raw["qa60"])
        and host["original_copy_verification_sha256"] == gate.digest(raw["copy"])
        and type(host["qa_returncode"]) is int
        and host["qa_returncode"] == 0,
        "Observed original CPU QA/copy proof hash chain differs",
    )
    same(r["copy"], qa["independent_milestone"], "Original HOST copy bytes differ")
    gate.require(
        r["copy"]["status"] == "independent_copy_sha_verified_readonly"
        and r["copy"]["hardlinks_used"] is False
        and set(r["copy"]["sha256"]) == set(ARTIFACTS),
        "Independent eight-file non-hardlinked milestone required",
    )
    for name, key in ARTIFACTS.items():
        gate.require(
            r["copy"]["sha256"][name] == gate.digest(raw[key]),
            "Copied artifact SHA differs: " + name,
        )
    same(
        r["terminal"],
        {
            "kind": "curated_pbs_terminal_v1",
            "status": "completed_reviewed_stages",
            "actual_epoch": 60,
            "completed_budget": True,
        },
        "Original curated terminal differs; no invented supervisor",
    )
    pbs = r["pbs"]
    gate.require(
        type(pbs["returncode"]) is int
        and pbs["returncode"] == 0
        and pbs["PBS"]["job_state"] == "F"
        and type(pbs["PBS"]["Exit_status"]) is int
        and pbs["PBS"]["Exit_status"] == 0,
        "Independent actual PBS F/Exit0 required",
    )
    gate.require(
        pbs["command"] == ["qstat", "-x", "-f", "-F", "json", r["identity60"]["pbs_job_id"]],
        "PBS evidence is not bound to this allocation",
    )
    gate.require(
        r["identity60"]["gpu_uuid"] == t["gpu_uuid"]
        and r["identity60"]["source_commit"] == SOURCE["commit"]
        and r["runtime60"]["status"] == "PASS"
        and r["runtime60"]["gpu_uuid"] == t["gpu_uuid"],
        "Original runtime/GPU/source differs",
    )
    same(r["runtime60"], r["run_result60"]["runtime"], "Actual training runtime bytes differ")
    gate.require(
        qa["completed_budget"] is True
        and qa["actual_epoch"] == qa["scheduler_T_max"] == qa["scheduler_last_epoch"] == 60
        and qa["steps"] == 122880
        and qa["delivered_samples"] == 491520
        and qa["finite_weights"] is True
        and qa["finite_optimizer"] is True,
        "Original finite full60 budget QA required",
    )
    gate.require(
        qa["source_commit"] == SOURCE["commit"]
        and qa["execution_code_sha256"] == SOURCE["execution_sha256"]
        and qa["training_source_sha256"] == SOURCE["training_sha256"],
        "QA source profile differs",
    )
    for key in ("plan_sha256", "config_sha256", "profile_sha256", "gpu_uuid", "worker_id"):
        same(qa[key], t[key], "Original CPU QA/task identity differs")
    gate.require(
        qa["manifest_sha256"] == MANIFEST
        and qa["canonical_sha256"] == CONTENT
        and qa["cache_sha256"] == RAW_CACHE["fingerprint"],
        "CPU QA canonical/cache identity differs",
    )
    same(qa["photometry_cache"], cache, "Original CPU QA portable cache differs")
    same(r["profile"]["study_contract_sha256"], digest(r["study"]), "Profile/study differs")
    gate.require(
        r["profile"]["reviewed_helper_manifest_sha256"] == HELPERS
        and r["profile"]["parent_tpe_manifest_sha256"] == PARENT,
        "Profile dependency contracts differ",
    )
    same(
        t["plan"].get("photometry_caches"),
        {
            a["config"]["photometric_profile"]: portable(a["photometry_cache"])
            for a in r["study"]["arms"].values()
            if a["photometry_cache"] is not None
        },
        "Full immutable photometry plan registry differs",
    )
    attestation = r["source_attestation"]
    gate.require(
        attestation["kind"] == "curated_host_source_attestation_v1",
        "Original HOST source attestation required",
    )
    same(
        attestation["bindings"],
        {
            "container_source": "/workspace",
            "host_source": r["profile"]["source"],
            "source_commit": SOURCE["commit"],
            "execution_sha256": SOURCE["execution_sha256"],
            "training_sha256": SOURCE["training_sha256"],
            "image_sha256": r["profile"]["image_sha256"],
            "git_status_porcelain": "",
        },
        "Actual original source/image attestation differs",
    )
    gate.require(
        r["identity60"]["profile_sha256"] == t["profile_sha256"]
        and r["identity60"]["worker_id"] == t["worker_id"]
        and r["runtime60"]["expected_gpu_uuid"] == t["gpu_uuid"]
        and r["runtime60"]["pbs_job_id"] == r["identity60"]["pbs_job_id"],
        "Actual allocation/runtime/profile identity differs",
    )
    return config, cache


def weights(raw, config, cache, descriptor):
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    same(
        payload["config"],
        config,
        "Full checkpoint configuration differs; fields cannot be stripped",
    )
    gate.require(
        type(payload["format_version"]) is int
        and payload["format_version"] == 1
        and payload["family"] == "unified"
        and payload["channels"] == list(gate.CHANNELS),
        "Wrong checkpoint family/channels",
    )
    gate.require(
        payload["training_source_sha256"] == SOURCE["training_sha256"]
        and payload["manifest_sha256"] == MANIFEST
        and payload["prepared_content_sha256"] == CONTENT
        and payload["geometry_policy"]["sha256"] == GEOMETRY,
        "Checkpoint training/data identity differs",
    )
    gate.require(
        payload["purpose"] == "full_data"
        and payload["normalization"] == "RGB/255-0.5"
        and payload["probability_fusion"] == "joint_av_mixture_then_conditional"
        and payload["inference_max_size"] is None
        and all(type(x) is bool and x for x in payload["supervised_channels"])
        and len(payload["supervised_channels"]) == 5,
        "Checkpoint inference/supervision contract differs",
    )
    gate.require(
        payload["lineage"]["initialization"] == "scratch"
        and payload["lineage"]["historical_exposure"] is False
        and payload["lineage"]["independent_model_eligible"] is True,
        "Independent fresh initialization required",
    )
    same(payload.get("derived_cache"), RAW_CACHE, "Saved raw cache differs")
    same(payload.get("photometry_cache"), cache, "Saved photometry cache differs")
    same(
        payload["preprocessing"].get("photometry"),
        descriptor,
        "Saved full-frame photometric descriptor differs",
    )
    gate.require(
        payload["preprocessing"]["profile"] == "legacy25_raw"
        and payload["preprocessing"]["pixels_per_degree"] == 25
        and payload["preprocessing"]["normalized"] is False,
        "Preparation normalization differs",
    )
    gate.require(
        type(payload["epoch"]) is int
        and 1 <= payload["epoch"] <= 60
        and payload["steps"] == payload["epoch"] * 2048,
        "Invalid checkpoint own epoch/budget",
    )
    gate.require(
        gate._finite(payload["state_dict"])
        and gate._finite(payload["validation_metrics"])
        and all(
            not v.is_floating_point() or v.dtype == torch.float32
            for v in payload["state_dict"].values()
        ),
        "Require finite FP32 model state/metrics",
    )
    # Keep ALL configuration above; only the architecture constructor's explicit
    # arguments are used here. No incompatible UnifiedConfig load or forward.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = UNet(5, config["base_channels"], config["depth"], config["norm"])
        model.load_state_dict(payload["state_dict"], strict=True)
    expected_preparation = {
        "profile": "legacy25_raw",
        "pixels_per_degree": 25.0,
        "normalized": False,
    }
    if descriptor is not None:
        expected_preparation["photometry"] = descriptor
    same(payload["preprocessing"], expected_preparation, "Full saved preparation fields differ")
    same(
        payload["architecture"],
        {
            "name": "compact_unet",
            "base_channels": config["base_channels"],
            "depth": config["depth"],
            "normalization": config["norm"],
            "outputs": 5,
        },
        "Architecture/config differs",
    )
    return payload, model


def history(best, last, model, r):
    gate.require(last["epoch"] == 60 and last["steps"] == 122880, "Actual LAST60 required")
    state = last["training_state"]
    optimizer = state["optimizer"]
    params = list(model.parameters())
    ids = [p for g in optimizer["param_groups"] for p in g["params"]]
    gate.require(
        len(ids) == len(set(ids)) == len(params)
        and set(ids) == set(optimizer["state"])
        and gate._finite(optimizer),
        "Optimizer parameter coverage/finite state differs",
    )
    for key, param in zip(ids, params, strict=True):
        s = optimizer["state"][key]
        gate.require(
            torch.is_tensor(s["step"]) and s["step"].numel() == 1 and s["step"].item() == 122880,
            "Stale optimizer step",
        )
        gate.require(
            all(
                torch.is_tensor(s[k]) and s[k].shape == param.shape
                for k in ("exp_avg", "exp_avg_sq")
            ),
            "Optimizer moments/parameter shape differs",
        )
    gate.require(
        state["scheduler"]["T_max"] == state["scheduler"]["last_epoch"] == 60,
        "Scheduler horizon differs",
    )
    gate.require(
        set(state["rng"]) == {"torch", "python", "numpy", "sampler", "augmentation", "cuda"}
        and all(
            torch.is_tensor(state["rng"][k])
            and state["rng"][k].dtype == torch.uint8
            and state["rng"][k].numel() > 0
            for k in ("torch", "sampler", "cuda")
        ),
        "Actual resume RNG bytes required",
    )
    same(state["coverage"], last["coverage"], "Resume coverage differs")
    rows = list(csv.DictReader(r["csv"].decode().splitlines()))
    gate.require(
        [int(x["epoch"]) for x in rows] == list(range(1, 61))
        and all(
            int(x["steps"]) == i * 2048 and math.isfinite(float(x["val_selection_score"]))
            for i, x in enumerate(rows, 1)
        ),
        "Full60 CSV history differs",
    )
    scores = [float(x["val_selection_score"]) for x in rows]
    epoch = max(range(60), key=lambda i: scores[i]) + 1
    gate.require(
        best["epoch"] == epoch
        and best["val_selection_score"] == scores[epoch - 1]
        and last["val_selection_score"] == scores[-1]
        and state["best_validation_score"] == scores[epoch - 1],
        "BEST/LAST role selection score differs",
    )
    for p in (best, last):
        gate.require(
            p["validation_metrics"]["macro_dataset_mean_dice"] == p["val_selection_score"],
            "Saved selected epoch metrics differ",
        )
    summary = gate.parse_json(r["summary"])
    gate.require(
        summary["completed_budget"] is True
        and summary["steps"] == 122880
        and summary["best_validation_score"] == best["val_selection_score"]
        and summary["selection_metric"] == "macro_dataset_mean_dice",
        "Completion summary differs",
    )


@dataclass(frozen=True)
class CheckedCuratedCompletion:
    checkpoint_snapshot: bytes = field(repr=False)
    proof_snapshot: bytes = field(repr=False)

    @property
    def proof(self):
        return gate.parse_json(self.proof_snapshot)


def check_curated_completion(
    *, original_index, expected_index_sha256, manifest, checkpoint_role, family
):
    gate.require(
        family == FAMILY and checkpoint_role in ("training_best", "fixed_terminal60"),
        "Explicit curated_fd44 family/role required",
    )
    gate.require(
        torch.get_default_dtype() == torch.float32, "CPU checker requires FP32 default dtype"
    )
    before = gate.evaluation_source_identity()
    index_raw, index = _pinned_json(
        original_index, expected_index_sha256, "original curated input index"
    )
    gate.exact(index, ("schema_version", "kind", "files"), "original curated index")
    gate.require(
        type(index["schema_version"]) is int
        and index["schema_version"] == 1
        and index["kind"] == "original_curated_completion_inputs"
        and set(index["files"]) == FILES,
        "Require explicit original curated schema/inventory, no native envelope or fake terminal",
    )
    raw = {}
    records = {}
    paths = {}
    for name, entry in index["files"].items():
        path = Path(entry["path"])
        gate.require(not path.is_symlink(), "Input symlink forbidden")
        blob = path.read_bytes()
        gate.require(gate.digest(blob) == entry["sha256"], "Original input SHA differs: " + name)
        raw[name] = blob
        paths[name] = path
        if name not in {"best", "last", "csv"}:
            records[name] = gate.parse_json(blob)
    config, cache = original_chain(records, raw)
    manifest_path = Path(manifest)
    manifest_raw = manifest_path.read_bytes()
    gate.require(gate.digest(manifest_raw) == MANIFEST, "Trusted canonical manifest bytes differ")
    groups = _groups(manifest_raw)["train"]
    descriptor = (
        None
        if config["photometric_profile"] == "raw"
        else records["study"]["photometric_descriptors"][config["photometric_profile"]]
    )
    best, _ = weights(raw["best"], config, cache, descriptor)
    last, model = weights(raw["last"], config, cache, descriptor)
    history(best, last, model, raw)
    same(records["coverage"], last["coverage"], "Original terminal coverage sidecar differs")
    same(
        records["validation"],
        best["validation_metrics"],
        "Original selected BEST validation sidecar differs",
    )
    same(records["run"]["config"], config, "Original run full configuration differs")
    gate.require(
        records["run"]["training_source_sha256"] == SOURCE["training_sha256"]
        and records["run"]["manifest_sha256"] == MANIFEST
        and records["run"]["prepared_content_sha256"] == CONTENT,
        "Run source/canonical binding differs",
    )
    same(records["run"].get("derived_cache"), RAW_CACHE, "Run raw cache differs")
    same(records["run"].get("photometry_cache"), cache, "Run photometry cache differs")
    same(
        records["marker"],
        {
            "plan_sha256": records["task60"]["plan_sha256"],
            "trial": records["task60"]["trial"],
            "execution_code_sha256": SOURCE["execution_sha256"],
            "manifest_sha256": MANIFEST,
        },
        "Original HPO marker differs",
    )
    del model
    selected = best if checkpoint_role == "training_best" else last
    selected_key = "best" if checkpoint_role == "training_best" else "last"
    terminal_coverage = _coverage(last, groups, "Curated LAST60")
    selected_coverage = _coverage(selected, groups, "Curated selected OWN weights")
    qa = records["qa60"]
    expected_records = {r["id"] for rows in groups.values() for r in rows}
    for key, payload in (
        ("best_checkpoint_coverage", best),
        ("terminal_checkpoint_coverage", last),
    ):
        visits = payload["coverage"]["annotation_visits"]
        gate.require(
            set(visits) <= expected_records
            and all(type(v) is int and v > 0 for v in visits.values()),
            "Foreign or invalid own annotation visits",
        )
        same(
            qa[key],
            {
                "epoch": payload["epoch"],
                "contributing_annotation_records": len(visits),
                "eligible_annotation_records": 5700,
                "all_annotations_contributed": set(visits) == expected_records,
            },
            "Original role coverage summary differs from its OWN weights",
        )
    gate.require(
        qa["selected_best_full_annotations"]
        is (set(best["coverage"]["annotation_visits"]) == expected_records),
        "Original BEST full annotation flag differs",
    )
    gate.require(
        qa["best_checkpoint_sha256"] == gate.digest(raw["best"])
        and qa["checkpoint_sha256"] == gate.digest(raw["last"])
        and qa["metric"] == best["val_selection_score"]
        and qa["best_epoch"] == best["epoch"],
        "Original QA role/hash/score differs",
    )
    same(qa["last_validation_metrics"], last["validation_metrics"], "Original LAST metrics differ")
    for key, path in paths.items():
        gate.require(path.read_bytes() == raw[key], "Input changed during check: " + key)
    gate.require(
        manifest_path.read_bytes() == manifest_raw and gate.evaluation_source_identity() == before,
        "Manifest/source/runtime changed during check",
    )
    proof = {
        "schema_version": 1,
        "kind": "checked_curated_fd44_completion",
        "family": FAMILY,
        "checkpoint_role": checkpoint_role,
        "checkpoint_sha256": gate.digest(raw[selected_key]),
        "checkpoint_epoch": selected["epoch"],
        "checkpoint_steps": selected["steps"],
        "selected_training_score": selected["val_selection_score"],
        "selected_training_metric": "macro_dataset_mean_dice",
        "raw_training_configuration": config,
        "selected_checkpoint_coverage": selected_coverage,
        "terminal_coverage": terminal_coverage,
        "original_qa_kind": "curated_unified_rung_evidence_v1",
        "original_receipt_sha256": gate.digest(raw["receipt60"]),
        "original_index_sha256": gate.digest(index_raw),
        "original_inputs_sha256": {k: gate.digest(v) for k, v in raw.items()},
        "source_profile": SOURCE,
        "actual_checker_source": before,
        "original_photometric_descriptor": descriptor,
        "portable_photometry_cache": cache,
        "PBS_actual_F_Exit0_checked_separately": True,
        "native_envelope_supplied_or_fabricated": False,
        "selection_created": False,
        "test_evaluation_enabled": False,
        "TTA_admission_enabled": False,
        "fresh_sourcecompatible_VAL_required": True,
        "prior_blindness_proven": False,
        "remaining_integration": "Independent completion only; no current selection/TTA/TEST path accepts this proof",
    }
    return CheckedCuratedCompletion(raw[selected_key], gate.canonical(proof))
