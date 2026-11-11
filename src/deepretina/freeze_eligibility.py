"""CPU freeze eligibility for typed original completion families; never selection/TEST.

Preserve raw training configuration and original terminal/QA bytes. Runtime loader
compatibility must not erase fields or transfer LAST coverage to an earlier BEST.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import selection as gate
from .completion_profiles import CONTENT, GEOMETRY, MANIFEST, PROFILES, ROLES, VERIFIER


def _sha_json(value, *, newline=False):
    return gate.digest(gate.canonical(value) + (b"\n" if newline else b""))


def _pinned_json(path, expected, label):
    gate.require(
        isinstance(expected, str)
        and len(expected) == 64
        and set(expected) <= set("0123456789abcdef"),
        "Require external SHA pin: " + label,
    )
    raw, value = gate.json_snapshot(path)
    gate.require(gate.digest(raw) == expected, "Externally pinned bytes differ: " + label)
    return raw, value


def _groups(manifest_bytes):
    rows = gate.parse_json(manifest_bytes)
    gate.require(isinstance(rows, list), "Canonical manifest must be a record list")
    partitions, ids, groups = {}, set(), {s: {} for s in ("train", "val", "test")}
    for row in rows:
        gate.require(
            row["id"] not in ids and row["split"] in groups, "Record identity/partition differs"
        )
        ids.add(row["id"])
        for key in ("photo_id", "source_id", "split_group_id"):
            identity = row.get(key)
            gate.require(
                isinstance(identity, str) and identity, "Missing canonical identity: " + key
            )
            previous = partitions.setdefault((key, identity), row["split"])
            gate.require(previous == row["split"], "Known identity crosses partitions")
        if (
            row["task"] in gate.SOURCE_CHANNELS
            and row.get("usable_for_training", True)
            and not row.get("exclude_from_training", False)
        ):
            metadata = row.get("metadata", {})
            gate.require(
                row.get("preparation_max_size") is None
                and not row.get("preparation_preview", metadata.get("preparation_preview", False))
                and row.get("preprocessing_profile", metadata.get("preprocessing_profile"))
                == "legacy25_raw"
                and row.get("pixels_per_degree", metadata.get("pixels_per_degree")) == 25,
                "Require uncapped canonical legacy25 preparation",
            )
            groups[row["split"]].setdefault(row["photo_id"], []).append(row)
    gate.require(
        {k: len(v) for k, v in groups.items()} == {"train": 2529, "val": 770, "test": 856},
        "This reviewed corpusv2 profile requires actual2529/770/856 photos",
    )
    return groups


def _coverage(payload, groups, label):
    coverage = payload.get("coverage", {})
    photos, annotations = coverage.get("photo_visits", {}), coverage.get("annotation_visits", {})
    records = {row["id"] for rows in groups.values() for row in rows}
    epoch = payload["epoch"]
    gate.require(
        set(photos) == set(groups)
        and len(groups) == 2529
        and coverage.get("eligible_photos") == coverage.get("seen_photos") == 2529
        and all(type(v) is int and v >= epoch for v in photos.values())
        and sum(photos.values()) == epoch * 8192,
        label + " must cover all actual TRAIN photos and full epoch sample budget",
    )
    gate.require(
        len(records) == 5700
        and set(annotations) == records
        and all(type(v) is int and v > 0 for v in annotations.values())
        and coverage.get("eligible_annotation_records")
        == coverage.get("contributing_annotation_records")
        == 5700,
        label + " must cover all actual5700 TRAIN annotations in its OWN weights",
    )
    return {
        "epoch": epoch,
        "eligible_photos": 2529,
        "seen_photos": len(photos),
        "minimum_photo_visits": min(photos.values()),
        "delivered_samples": sum(photos.values()),
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": len(annotations),
        "minimum_annotation_visits": min(annotations.values()),
        "eligible_photo_ids_sha256": _sha_json(sorted(groups), newline=True),
        "eligible_annotation_ids_sha256": _sha_json(sorted(records), newline=True),
        "photo_visits_sha256": _sha_json(photos),
        "annotation_visits_sha256": _sha_json(annotations),
    }


def _optimizer(last_bytes, last):
    # Native registration order, CPU construction only; never a model forward.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model, _, _ = gate.load_unified(io.BytesIO(last_bytes), device="cpu")
    parameters = list(model.parameters())
    state = last["training_state"]
    optimizer = state["optimizer"]
    ids = [p for group in optimizer["param_groups"] for p in group["params"]]
    gate.require(
        len(ids) == len(set(ids)) == len(parameters) and set(ids) == set(optimizer["state"]),
        "Optimizer parameter/state coverage differs",
    )
    gate.require(gate._finite(optimizer), "Nonfinite optimizer")
    for key, parameter in zip(ids, parameters, strict=True):
        value = optimizer["state"][key]
        step = value.get("step")
        gate.require(
            torch.is_tensor(step) and step.numel() == 1 and 0 < step.item() <= last["steps"],
            "Invalid optimizer step",
        )
        if last["config"]["architecture"] == "compact_unet":
            gate.require(step.item() == last["steps"], "Stale compact optimizer step")
        for name in ("exp_avg", "exp_avg_sq"):
            gate.require(
                torch.is_tensor(value.get(name)) and value[name].shape == parameter.shape,
                "Optimizer moment/parameter shape differs",
            )
    gate.require(
        set(state.get("rng", {})) == {"torch", "python", "numpy", "sampler", "augmentation", "cuda"}
        and gate._finite(state["rng"]),
        "Missing/nonfinite complete RNG state",
    )


def _common_qa(qa, profile, best, last, csv_bytes, summary_bytes):
    kind, commit, execution, training = profile
    gate.require(
        qa.get("kind") == kind
        and type(qa.get("schema_version")) is int
        and qa.get("schema_version") == 1
        and qa.get("terminal") is True
        and qa.get("last_epoch") == 60
        and qa.get("finite_best_last_weights_and_last_optimizer") is True,
        "Require genuine explicit-family finite completed60 QA; no retagging",
    )
    gate.require(
        qa["source_commit"] == commit
        and qa["execution_code_sha256"] == execution
        and qa["checkpoint_training_source_sha256"] == training,
        "Reviewed training-family source bytes differ",
    )
    gate.require(
        qa["checkpoint_sha256"]
        == {"best.pt": best["checkpoint_sha256"], "last.pt": last["checkpoint_sha256"]}
        and qa["best_epoch"] == best["epoch"]
        and qa["steps"] == last["steps"]
        and qa["delivered_samples"] == 491520,
        "Original QA and raw BEST/LAST history differ",
    )
    gate.require(
        qa["manifest_sha256"] == MANIFEST
        and qa["canonical_fingerprint"] == CONTENT
        and qa["geometry_policy_sha256"] == GEOMETRY
        and qa["derived_cache"] == last.get("derived_cache"),
        "Original QA corpus/cache/policy differs",
    )
    gate.require(
        qa["sidecar_sha256"]["metrics.csv"] == gate.digest(csv_bytes)
        and qa["sidecar_sha256"]["training_summary.json"] == gate.digest(summary_bytes),
        "Original QA CSV/summary bytes differ",
    )


def _standalone(qa, profile, best, last, protocol, protocol_raw, terminal, terminal_raw):
    _kind, commit, execution, training = profile
    gate.require(
        qa.get("verifier_sha256") == VERIFIER and qa.get("source_git_verified") is True,
        "Require original pinned standalone verifier/source observations",
    )
    gate.require(
        qa["reviewed_protocol_sha256"] == gate.digest(protocol_raw)
        and qa["original_supervisor_terminal_sha256"] == gate.digest(terminal_raw)
        and qa["original_protocol_text"].encode() == protocol_raw
        and qa["original_supervisor_terminal_text"].encode() == terminal_raw,
        "Original standalone protocol/terminal bytes differ",
    )
    gate.require(
        protocol.get("kind") == "reviewed_standalone_unified_protocol"
        and terminal.get("kind") == "bounded_unified_supervisor_terminal"
        and terminal.get("status") == "completed"
        and type(terminal.get("returncode")) is int
        and terminal.get("returncode") == 0
        and terminal.get("observed_returncode") is True
        and terminal.get("timed_out") is False
        and terminal.get("test_evaluation_performed") is False,
        "Require actual observed original standalone completion, not a pool receipt",
    )
    source = protocol["source"]
    gate.require(
        source["commit"] == commit
        and source["execution_code_sha256"] == execution
        and source["training_source_sha256"] == training
        and terminal["source"] == source,
        "Original standalone source binding differs",
    )
    gate.require(
        protocol["config"] == best["config"] == last["config"]
        and terminal["protocol_sha256"] == gate.digest(protocol_raw)
        and terminal["config_sha256"] == _sha_json(protocol["config"]),
        "Original exact training configuration/protocol differs",
    )
    gate.require(
        protocol["data"] == terminal["data"]
        and protocol["data"]["manifest_sha256"] == MANIFEST
        and protocol["data"]["prepared_content_sha256"] == CONTENT
        and protocol["data"]["geometry_policy_sha256"] == GEOMETRY
        and protocol["data"]["derived_cache"] == last.get("derived_cache"),
        "Original standalone canonical/cache binding differs",
    )
    gate.require(
        protocol["budget"]
        == {
            "epochs": 60,
            "epoch_samples": 8192,
            "train_photos": 2529,
            "train_annotation_records": 5700,
            "dataset_count": 21,
        },
        "Original standalone reviewed full-budget profile differs",
    )
    gate.require(
        terminal["completed_epochs"] == 60
        and terminal["steps"] == last["steps"]
        and terminal["delivered_samples"] == 491520,
        "Original terminal did not observe full60 budget",
    )
    gate.require(
        protocol["inference"]
        == {
            "threshold": 0.5,
            "tile_size": best["config"]["tile_size"],
            "overlap": best["config"]["overlap"],
        },
        "Original reviewed training inference declaration differs",
    )
    raw_values = {k: v for k, v in best["config"].items() if k != "device"}
    gate.require(
        qa["configuration"] == {"values": raw_values, "sha256": _sha_json(raw_values)},
        "Original QA raw training configuration was normalized or changed",
    )
    return {
        "kind": "original_standalone_supervisor_terminal",
        "sha256": gate.digest(terminal_raw),
        "reviewed_protocol_sha256": gate.digest(protocol_raw),
    }


def _pool(qa, best, last):
    # Preserve the original trusted6b pool family; never claim this is standalone.
    raw = qa["original_receipt_text"].encode()
    receipt = gate.parse_json(raw)
    task, decision = qa["original_task"], qa["original_decision"]
    gate.require(
        gate.digest(raw) == qa["original_receipt_sha256"]
        and receipt["status"] == "COMPLETE"
        and receipt["actual_epoch"] == receipt["target_epoch"] == 60,
        "Require genuine original pool COMPLETE receipt",
    )
    gate.require(
        decision["action"] == "finished"
        and decision["trial_state"] == "COMPLETE"
        and decision["receipt_sha256"] == qa["original_receipt_sha256"],
        "Require original pool completion decision",
    )
    for key in ("worker_id", "task_id", "lease", "plan_sha256", "config_sha256"):
        gate.require(
            task[key] == receipt[key] == decision[key],
            "Original pool task/receipt/decision differ: " + key,
        )
    gate.require(
        task["source_commit"] == PROFILES["pool_6b"][1]
        and task["execution_code_sha256"] == PROFILES["pool_6b"][2]
        and task["plan"]["training_source_sha256"] == PROFILES["pool_6b"][3]
        and task["plan"]["candidates"][task["trial"]] == best["config"] == last["config"],
        "Original pool source/config differs",
    )
    plan = task["plan"]
    gate.require(
        task["plan_sha256"] == qa["plan_sha256"] == _sha_json(plan, newline=True)
        and task["config_sha256"] == _sha_json(best["config"], newline=True)
        and plan["manifest_sha256"] == MANIFEST
        and plan["prepared_content_sha256"] == CONTENT
        and plan["geometry_policy_sha256"] == GEOMETRY
        and plan["derived_cache"] == last.get("derived_cache"),
        "Original pool plan/config/data byte identity differs",
    )
    native = receipt["checkpoint_qa"]
    gate.require(
        native["completed_budget"] is True
        and native["checkpoint_sha256"] == last["checkpoint_sha256"]
        and native["best_checkpoint_sha256"] == best["checkpoint_sha256"],
        "Original pool receipt does not bind raw completed checkpoints",
    )
    return {
        "kind": "original_pool_COMPLETE_receipt",
        "sha256": gate.digest(raw),
        "original_task_sha256": qa["original_task_sha256"],
        "original_decision_sha256": qa["original_decision_sha256"],
    }


def _verify_original_coverage(
    qa, family, terminal_coverage, selected_coverage, last, checkpoint_role
):
    gate.require(
        qa["eligible_photos"] == 2529
        and qa["eligible_annotation_records"] == qa["contributing_annotation_records"] == 5700
        and qa["all_eligible_annotations_contributed"] is True
        and qa["minimum_photo_visits"] == terminal_coverage["minimum_photo_visits"]
        and qa["minimum_annotation_visits"] == terminal_coverage["minimum_annotation_visits"]
        and qa["by_dataset"] == last["coverage"]["by_dataset"],
        "Original QA terminal own coverage differs",
    )
    if family != "pool_6b":
        own = qa[
            "best_checkpoint_coverage"
            if checkpoint_role == "training_best"
            else "last_checkpoint_coverage"
        ]
        for key in (
            "epoch",
            "eligible_photos",
            "seen_photos",
            "eligible_annotation_records",
            "contributing_annotation_records",
            "minimum_photo_visits",
            "annotation_visits_sha256",
            "photo_visits_sha256",
        ):
            gate.require(
                own[key] == selected_coverage[key],
                "Original QA selected own coverage differs: " + key,
            )
        gate.require(
            own["all_eligible_annotations_contributed"] is True
            and own["missing_annotation_ids"] == [],
            "Original QA selected weights lack all annotations",
        )
    else:
        gate.require(
            qa["eligible_photo_ids_sha256"] == terminal_coverage["eligible_photo_ids_sha256"]
            and qa["eligible_annotation_ids_sha256"]
            == terminal_coverage["eligible_annotation_ids_sha256"],
            "Original pool exact TRAIN identity sets differ",
        )


@dataclass(frozen=True)
class CheckedFreezeEligibility:
    checkpoint_snapshot: bytes = field(repr=False)
    proof_snapshot: bytes = field(repr=False)

    @property
    def proof(self):
        return gate.parse_json(self.proof_snapshot)


def check_freeze_eligibility(
    *,
    checkpoint,
    best_checkpoint,
    last_checkpoint,
    manifest,
    metrics_csv,
    training_summary,
    completion_qa,
    expected_qa_sha256,
    family,
    checkpoint_role,
    reviewed_protocol=None,
    expected_protocol_sha256=None,
    original_terminal=None,
    expected_terminal_sha256=None,
    original_pool_evidence=None,
    expected_pool_evidence_sha256=None,
):
    """Check exact own-weight coverage/provenance; does not create a selection.

    V3 pool72 requires its genuine native original COMPLETE receipt and externally
    pinned original input index; never repackage it as standalone completion.
    """
    gate.require(family in PROFILES, "Unknown explicit completion family")
    gate.require(
        checkpoint_role in ROLES, "Require explicit training_best or fixed_terminal60 role"
    )
    source_before = gate.evaluation_source_identity()
    paths = {
        "selected": Path(checkpoint),
        "best": Path(best_checkpoint),
        "last": Path(last_checkpoint),
        "manifest": Path(manifest),
        "csv": Path(metrics_csv),
        "summary": Path(training_summary),
        "qa": Path(completion_qa),
    }
    qa_raw, qa = _pinned_json(completion_qa, expected_qa_sha256, "completion QA")
    if family == "pool_72_v3":
        gate.require(
            qa.get("status") == "COMPLETE"
            and qa.get("actual_epoch") == qa.get("target_epoch") == 60,
            "Original V3 pool receipt must actually complete60; partial rung is ineligible",
        )
    best_raw, best = gate.checkpoint_snapshot(best_checkpoint)
    last_raw, last = gate.checkpoint_snapshot(last_checkpoint)
    gate.require(
        last["epoch"] == 60, "Require actual completed60 LAST; partial72/E3 is not eligible"
    )
    selected_raw, selected = gate.checkpoint_snapshot(checkpoint)
    wanted = best if checkpoint_role == "training_best" else last
    gate.require(
        gate.digest(selected_raw) == wanted["checkpoint_sha256"],
        "Selected checkpoint bytes do not match the explicit BEST/LAST role",
    )
    gate.require(
        best["config"] == last["config"] == selected["config"],
        "Raw original training config differs",
    )
    profile = PROFILES[family]
    gate.require(
        all(p["training_source_sha256"] == profile[3] for p in (best, last, selected)),
        "Raw checkpoint training source differs from explicit family",
    )
    for p in (best, last, selected):
        gate.require(
            p["manifest_sha256"] == MANIFEST
            and p["prepared_content_sha256"] == CONTENT
            and p.get("geometry_policy", {}).get("sha256") == GEOMETRY,
            "Raw corpus/geometry profile differs",
        )
    manifest_raw = paths["manifest"].read_bytes()
    gate.require(gate.digest(manifest_raw) == MANIFEST, "Actual manifest bytes differ")
    groups = _groups(manifest_raw)
    terminal_coverage = _coverage(last, groups["train"], "Terminal LAST60")
    selected_coverage = _coverage(selected, groups["train"], "Selected " + checkpoint_role)
    csv_bytes, summary_bytes = paths["csv"].read_bytes(), paths["summary"].read_bytes()
    completion = gate.completion_evidence(best, last, csv_bytes, summary_bytes, groups["train"])
    _optimizer(last_raw, last)
    gate.require(
        gate._finite(best.get("validation_metrics"))
        and gate._finite(last.get("validation_metrics")),
        "Nonfinite original validation metadata",
    )
    if family != "pool_72_v3":
        _common_qa(qa, profile, best, last, csv_bytes, summary_bytes)
    if family == "pool_72_v3":
        from .v3_completion import verify_original_pool

        gate.require(
            original_pool_evidence is not None,
            "V3 requires original externally pinned pool evidence index",
        )
        index_raw, index = _pinned_json(
            original_pool_evidence, expected_pool_evidence_sha256, "original V3 evidence index"
        )
        gate.require(
            reviewed_protocol is None and original_terminal is None,
            "V3 pool must not be relabeled standalone",
        )
        original, pool_snapshots = verify_original_pool(
            qa,
            index,
            best,
            last,
            {
                "best.pt": best_raw,
                "last.pt": last_raw,
                "metrics.csv": csv_bytes,
                "training_summary.json": summary_bytes,
            },
            return_snapshots=True,
        )
        original["sha256"] = gate.digest(qa_raw)
        paths["pool_index"] = Path(original_pool_evidence)
    elif family == "pool_6b":
        gate.require(
            reviewed_protocol is None and original_terminal is None,
            "Pool proof cannot be supplied as a standalone protocol/terminal",
        )
        original = _pool(qa, best, last)
    else:
        gate.require(
            all(
                v is not None
                for v in (
                    reviewed_protocol,
                    expected_protocol_sha256,
                    original_terminal,
                    expected_terminal_sha256,
                )
            ),
            "Standalone family requires original reviewed protocol and supervisor terminal pins",
        )
        protocol_raw, protocol = _pinned_json(
            reviewed_protocol, expected_protocol_sha256, "standalone protocol"
        )
        terminal_raw, terminal = _pinned_json(
            original_terminal, expected_terminal_sha256, "standalone terminal"
        )
        paths.update(protocol=Path(reviewed_protocol), terminal=Path(original_terminal))
        original = _standalone(
            qa, profile, best, last, protocol, protocol_raw, terminal, terminal_raw
        )
        for name, raw in (
            ("best.pt", best_raw),
            ("last.pt", last_raw),
            ("metrics.csv", csv_bytes),
            ("training_summary.json", summary_bytes),
        ):
            gate.require(
                terminal["artifact_sha256"][name] == gate.digest(raw),
                "Original terminal artifact bytes differ: " + name,
            )
    if family != "pool_72_v3":
        _verify_original_coverage(
            qa, family, terminal_coverage, selected_coverage, last, checkpoint_role
        )
    # Actual source/canonical operation lease; encoded TEST bytes may be hashed,
    # but this route never decodes GT arrays or calls any prediction/evaluator.
    canonical_before = gate._canonical_metadata(manifest)
    data, actual_groups = gate.data_identity(manifest, selected)
    gate.require(
        actual_groups
        == {
            s: {pid: sorted(rows, key=lambda r: r["id"]) for pid, rows in photos.items()}
            for s, photos in groups.items()
        },
        "Canonical inventory changed during eligibility",
    )
    gate._finish_canonical_read(manifest, CONTENT, canonical_before)
    gate.require(
        gate.evaluation_source_identity() == source_before,
        "Evaluation code/runtime changed during eligibility",
    )
    if family == "pool_72_v3":
        gate.require(
            all(path.read_bytes() == raw for path, raw in pool_snapshots),
            "Original V3 inputs changed during complete eligibility operation",
        )
    snapshots = {
        "selected": selected_raw,
        "best": best_raw,
        "last": last_raw,
        "manifest": manifest_raw,
        "csv": csv_bytes,
        "summary": summary_bytes,
        "qa": qa_raw,
    }
    if family == "pool_72_v3":
        snapshots["pool_index"] = index_raw
    elif family != "pool_6b":
        snapshots.update(protocol=protocol_raw, terminal=terminal_raw)
    gate.require(
        all(paths[k].read_bytes() == v for k, v in snapshots.items()),
        "Original inputs changed during eligibility",
    )
    # Read this selected epoch's own training score; never relabel macro BEST as LAST.
    score = selected["val_selection_score"]
    gate.require(
        selected["validation_metrics"][selected["config"]["selection_metric"]] == score,
        "Selected epoch's saved validation score differs from its own raw metrics",
    )
    proof = {
        "schema_version": 1,
        "kind": "checked_freeze_eligibility",
        "family": family,
        "checkpoint_role": checkpoint_role,
        "checkpoint_sha256": gate.digest(selected_raw),
        "checkpoint_epoch": selected["epoch"],
        "checkpoint_steps": selected["steps"],
        "selected_training_metric": selected["config"]["selection_metric"],
        "selected_training_score": score,
        "raw_training_configuration": selected["config"],
        "raw_training_configuration_sha256": _sha_json(selected["config"]),
        "training_best": {
            "sha256": best["checkpoint_sha256"],
            "epoch": best["epoch"],
            "score": best["val_selection_score"],
        },
        "training_completion": completion,
        "terminal_coverage": terminal_coverage,
        "selected_checkpoint_coverage": selected_coverage,
        "source_profile": {
            "commit": profile[1],
            "execution_code_sha256": profile[2],
            "training_source_sha256": profile[3],
        },
        "actual_evaluation_source": source_before,
        "canonical_data": data,
        "original_qa_kind": qa["checkpoint_qa"]["kind"] if family == "pool_72_v3" else qa["kind"],
        "original_qa_sha256": gate.digest(qa_raw),
        "original_completion": original,
        "input_sha256": {k: gate.digest(v) for k, v in snapshots.items()},
        "selection_created": False,
        "test_evaluation_enabled": False,
        "prior_blindness_proven": False,
        "remaining_integration": "This eligibility receipt alone is not a selection or execution permit",
    }
    return CheckedFreezeEligibility(selected_raw, gate.canonical(proof))
