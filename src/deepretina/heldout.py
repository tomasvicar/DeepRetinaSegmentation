"""Explicit stage2 execution after the frozen stage1 metadata gate.

The stage1 receipt stays metadata-only. This separate route requires an explicit
execution request, preserves the reviewed pins and never claims prior blindness.
"""

from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from . import selection as gate
from .unified import UnifiedConfig, evaluate_unified_model, inference_config, load_unified
from .validation_evidence import evaluation_runtime


def evaluate_reviewed_heldout(
    *,
    selection,
    expected_selection_sha256,
    checkpoint,
    last_checkpoint,
    manifest,
    validation_evidence,
    metrics_csv,
    training_summary,
    output,
    execute_reviewed_heldout=False,
    device="cpu",
):
    """Verify the complete selection before any test GT decoding/CUDA/forward.

    No protocol overrides are accepted. The artifact's externally pinned protocol
    determines inference; all weights originate in the checked byte snapshot.
    Canonical/source observations span the gate, inference and report emission.
    """
    output, manifest = Path(output), Path(manifest)
    gate.require(not output.exists() and not output.is_symlink(), "Held-out output already exists")
    gate.require(
        execute_reviewed_heldout is True,
        "Explicit reviewed stage2 held-out execution request required",
    )
    gate.require(
        not output.resolve().is_relative_to(manifest.parent.resolve()),
        "Write held-out evidence outside immutable canonical data",
    )
    source_before = gate.evaluation_source_identity()
    canonical_before = gate._canonical_metadata(manifest)
    selection_bytes, artifact = gate.json_snapshot(selection)
    gate.require(
        gate.digest(selection_bytes) == expected_selection_sha256,
        "Selection differs from externally pinned review digest",
    )
    gate.require(isinstance(artifact, dict), "Invalid selection artifact fields")
    frozen = artifact.get("inference_protocol")
    gate.require(isinstance(frozen, dict), "Missing frozen inference protocol")
    checked = gate.check_selection(
        selection=selection,
        expected_selection_sha256=expected_selection_sha256,
        checkpoint=checkpoint,
        last_checkpoint=last_checkpoint,
        manifest=manifest,
        validation_evidence=validation_evidence,
        metrics_csv=metrics_csv,
        training_summary=training_summary,
        threshold=frozen.get("threshold"),
        tile_size=frozen.get("tile_size"),
        overlap=frozen.get("overlap"),
    )
    # False is the documented stage1 boundary, not an implicit permission bit.
    # The separate reviewed stage2 request above enables this execution only.
    gate.require(checked.test_evaluation_enabled is False, "Expected metadata-only stage1 receipt")
    gate.require(
        checked.selection_sha256 == expected_selection_sha256, "Checked selection pin differs"
    )
    gate.require(
        gate.digest(checked.checkpoint_snapshot) == checked.checkpoint_sha256,
        "Checked checkpoint snapshot differs",
    )
    data, protocol = checked.canonical_data, checked.protocol
    gate.require(protocol == frozen, "Checked protocol differs from pinned artifact")

    def lease():
        gate._finish_canonical_read(manifest, data["prepared_content_sha256"], canonical_before)
        gate.require(
            gate.evaluation_source_identity() == source_before,
            "Source/runtime changed during stage2 held-out evaluation",
        )
        gate.require(
            gate._canonical_metadata(manifest) == canonical_before,
            "Canonical data changed during stage2 held-out evaluation",
        )

    # Close changes during the metadata gate before CUDA initialization or test
    # decoding. Hashing held-out bytes for identity is not GT interpretation.
    lease()
    with torch.random.fork_rng(devices=[]):
        model, payload, actual_device = load_unified(
            io.BytesIO(checked.checkpoint_snapshot), device=device
        )
    gate.require(
        payload["checkpoint_sha256"] == checked.checkpoint_sha256,
        "Evaluated checkpoint differs from checked snapshot",
    )
    config = inference_config(payload, tile_size=protocol["tile_size"], overlap=protocol["overlap"])
    gate.require(
        gate.inference_protocol(
            payload,
            threshold=protocol["threshold"],
            tile_size=protocol["tile_size"],
            overlap=protocol["overlap"],
        )
        == protocol,
        "Evaluated inference protocol differs from frozen selection",
    )
    actual_data, groups = gate.data_identity(manifest, payload)
    gate.require(actual_data == data, "Evaluated canonical identity differs from frozen selection")
    runtime = evaluation_runtime(actual_device)
    lease()
    runtime["evaluation_started_at_utc"] = datetime.now(UTC).isoformat()
    with torch.inference_mode():
        metrics = evaluate_unified_model(
            model,
            manifest,
            split="test",
            device=actual_device,
            config=config,
            threshold=protocol["threshold"],
        )
    runtime["evaluation_finished_at_utc"] = datetime.now(UTC).isoformat()
    metrics["evaluation_runtime"] = runtime
    gate.validate_metrics(
        metrics,
        groups["test"],
        protocol,
        UnifiedConfig(**payload["config"]),
        manifest.parent,
        expected_split="test",
    )
    lease()
    report = {
        "schema_version": 1,
        "kind": "unified_reviewed_heldout_evaluation",
        "execution_context": {
            "route": "explicit_reviewed_stage2_script",
            "explicit_execution_requested": True,
            "stage1_test_evaluation_enabled": checked.test_evaluation_enabled,
            "prior_blindness_proven": False,
            "reviewer_identity_authenticated": False,
            "new_model_selection_performed": False,
        },
        "selection_sha256": checked.selection_sha256,
        "selection_decision": artifact["decision"],
        "training_completion": artifact["training_completion"],
        "validation_evidence": artifact["validation_evidence"],
        "checkpoint": {
            "sha256": checked.checkpoint_sha256,
            "epoch": payload["epoch"],
            "steps": payload["steps"],
            "lineage": payload["lineage"],
        },
        "configuration": gate.config_identity(payload),
        "source": {**source_before, "training_source_sha256": payload["training_source_sha256"]},
        "canonical_data": data,
        "inference_protocol": protocol,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    snapshot = gate.canonical(report) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lease through report emission. On an operation-scope failure,
    # discard only our newly created report; preserve a concurrently replaced file.
    with output.open("xb") as stream:
        try:
            stream.write(snapshot)
            stream.flush()
            lease()
            written = os.fstat(stream.fileno())
            published = output.stat()
            gate.require(
                (written.st_dev, written.st_ino) == (published.st_dev, published.st_ino),
                "Held-out output replaced during emission",
            )
        except BaseException:
            info = os.fstat(stream.fileno())
            try:
                current = output.stat()
                if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                    output.unlink()
            except FileNotFoundError:
                pass
            raise
    return report
