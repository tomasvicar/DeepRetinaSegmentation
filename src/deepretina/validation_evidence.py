"""Fresh VAL-only evidence from checked immutable checkpoint bytes.

This producer does not open or score the test partition. Device observations are
reported for reproducibility, not authenticated hardware or historical blindness.
"""

from __future__ import annotations

import io
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from . import selection as gate
from .completion_profiles import PROFILES
from .inference_tta import evidence_kind, protocol_extension
from .unified import (
    UnifiedConfig,
    VerifiedPhotometryCache,
    evaluate_unified_model,
    inference_config,
    load_unified,
)


def evaluation_runtime(device):
    """Observe the actual evaluation device; no hardware attestation is implied."""
    device = torch.device(device)
    result = {
        "device": str(device),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_cpu_interop_threads": torch.get_num_interop_threads(),
        "torch_cuda_build": torch.version.cuda,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "hardware_attestation": False,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        try:
            drivers = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise gate.SelectionError(
                "CUDA evidence requires actual nvidia-smi driver observations"
            ) from error
        gate.require(bool(drivers), "Missing CUDA driver observations")
        result.update(
            device_name=properties.name,
            device_uuid=str(getattr(properties, "uuid", "unavailable")),
            device_total_memory_bytes=properties.total_memory,
            cudnn_version=torch.backends.cudnn.version(),
            nvidia_smi_gpu_driver_rows=drivers,
        )
    return result


def produce_validation_evidence(
    *,
    checkpoint,
    manifest,
    output,
    device="cpu",
    threshold=0.5,
    tile_size=None,
    overlap=None,
    tta="none",
    photometry_cache_root=None,
    expected_photometry_cache_fingerprint=None,
):
    """Evaluate an eligible full-data E1..60 snapshot, then write fresh evidence.

    Planned budget must remain60; only the separate selection gate requires an
    actually completed last60 checkpoint. No historical reports are accepted as
    inputs, and there is intentionally no split argument.
    """
    protocol_extension(tta)
    gate.require(
        (photometry_cache_root is None) == (expected_photometry_cache_fingerprint is None),
        "Photometry cache requires its runtime directory and external fingerprint together",
    )
    output, manifest = Path(output), Path(manifest)
    gate.require(
        not output.exists() and not output.is_symlink(),
        "Validation output already exists",
    )
    gate.require(
        not output.resolve().is_relative_to(manifest.parent.resolve()),
        "Write validation evidence outside immutable canonical data",
    )
    source_before = gate.evaluation_source_identity()
    snapshot, payload = gate.checkpoint_snapshot(checkpoint)
    gate.canonical({"training_selection_score": payload["val_selection_score"]})
    gate.require(
        tta == "none" or payload["training_source_sha256"] == PROFILES["standalone_72"][3],
        "Reviewed TTA producer requires source72 checkpoint; completion eligibility is checked separately",
    )
    configuration = gate.config_identity(payload)
    protocol = gate.inference_protocol(
        payload, threshold=threshold, tile_size=tile_size, overlap=overlap, tta=tta
    )
    config = inference_config(payload, tile_size=tile_size, overlap=overlap)
    photometry_cache = None
    if photometry_cache_root is not None:
        photometry_cache = VerifiedPhotometryCache(
            photometry_cache_root,
            manifest,
            config.photometric_profile,
            expected_fingerprint=expected_photometry_cache_fingerprint,
            expected_canonical_fingerprint=payload["prepared_content_sha256"],
        )
    canonical_before = gate._canonical_metadata(manifest)
    data, groups = gate.data_identity(manifest, payload)
    source = {
        **source_before,
        "training_source_sha256": payload["training_source_sha256"],
    }

    # The path is never reopened: both the checked metadata and the evaluated
    # weights originate in this exact immutable byte string, weights_only=True.
    with torch.random.fork_rng(devices=[]):
        model, loaded, actual_device = load_unified(io.BytesIO(snapshot), device=device)
    gate.require(
        loaded["checkpoint_sha256"] == gate.digest(snapshot),
        "Evaluated snapshot differs",
    )
    runtime = evaluation_runtime(actual_device)
    if photometry_cache is not None:
        runtime["photometry_cache"] = photometry_cache.identity
    gate._finish_canonical_read(manifest, payload["prepared_content_sha256"], canonical_before)
    gate.require(
        gate.evaluation_source_identity() == source_before,
        "Source/runtime changed before VAL evaluation",
    )
    runtime["evaluation_started_at_utc"] = datetime.now(UTC).isoformat()
    with torch.inference_mode():
        metrics = evaluate_unified_model(
            model,
            manifest,
            split="val",
            device=actual_device,
            config=config,
            threshold=threshold,
            **({"tta": tta} if tta != "none" else {}),
            **({"photometry_cache": photometry_cache} if photometry_cache is not None else {}),
        )
    runtime["evaluation_finished_at_utc"] = datetime.now(UTC).isoformat()
    # Recompute provenance from this actual evaluation, never retrofit a thin
    # saved report. The checkpoint's training-selection score remains separate.
    metrics["selection_metric"] = config.selection_metric
    metrics["selection_score"] = metrics[config.selection_metric]
    metrics["evaluation_runtime"] = runtime
    gate.validate_metrics(
        metrics,
        groups["val"],
        protocol,
        UnifiedConfig(**payload["config"]),
        manifest.parent,
    )
    gate._finish_canonical_read(manifest, payload["prepared_content_sha256"], canonical_before)
    gate.require(
        gate.evaluation_source_identity() == source_before,
        "Source/runtime changed during VAL evaluation",
    )
    gate.require(
        gate._canonical_metadata(manifest) == canonical_before,
        "Canonical data changed during VAL evaluation",
    )
    evidence = {
        "schema_version": gate.VERSION,
        "kind": evidence_kind(tta),
        "checkpoint": {
            "sha256": gate.digest(snapshot),
            "epoch": payload["epoch"],
            "steps": payload["steps"],
            "training_selection_score": payload["val_selection_score"],
        },
        "configuration": configuration,
        "source": source,
        "data": data,
        "inference_protocol": protocol,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    serialized = gate.canonical(evidence) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(serialized)
    return evidence
