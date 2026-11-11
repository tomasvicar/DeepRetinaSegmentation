"""Synthetic provenance tests only: no actual vascular4/proposal results exist."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "standalone_val_compare_test", ROOT / "scripts/compare_standalone_val.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case():
    source, execution, training = m.CONTRACTS["source72"]
    cfg = {
        "device": "cuda:0",
        "epochs": 60,
        "epoch_samples": 8192,
        "batch_size": 4,
        "coverage_first": True,
        "max_steps": None,
        "legacy_initialization": None,
        "allow_historical_exposure": False,
        "purpose": "full_data",
        "mixed_precision": "bf16",
        "workers": 8,
    }
    cache = {
        "schema": 1,
        "fingerprint": m.CACHE,
        "canonical_fingerprint": m.CANONICAL,
        "allow_canonical_fallback": False,
    }
    data = {
        "manifest_sha256": m.MANIFEST,
        "prepared_content_sha256": m.CANONICAL,
        "geometry_policy_sha256": m.POLICY,
        "derived_cache": cache,
    }
    pins = {
        "commit": source,
        "execution_code_sha256": execution,
        "training_source_sha256": training,
    }
    protocol = {
        "kind": "reviewed_standalone_unified_protocol",
        "source": pins,
        "config": cfg,
        "data": data,
        "inference": {"threshold": 0.5, "tile_size": 512, "overlap": 64},
    }
    praw = m.canonical(protocol)
    terminal = {
        "kind": "bounded_unified_supervisor_terminal",
        "status": "completed",
        "returncode": 0,
        "observed_returncode": True,
        "timed_out": False,
        "completed_epochs": 60,
        "source": pins,
        "data": data,
        "protocol_sha256": m.sha(praw),
        "config_sha256": m.sha(m.canonical(cfg)),
        "steps": 122880,
        "delivered_samples": 491520,
        "artifact_sha256": {"best.pt": "b" * 64, "last.pt": "a" * 64},
    }
    traw = m.canonical(terminal)
    coverage = {
        "epoch": 60,
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "all_eligible_annotations_contributed": True,
        "missing_annotation_ids": [],
        "minimum_photo_visits": 66,
    }
    values = {k: v for k, v in cfg.items() if k != "device"}
    configuration = {"values": values, "sha256": m.sha(m.canonical(values))}
    qa = {
        "kind": "standalone_unified_completion_qa",
        "schema_version": 1,
        "verifier_sha256": m.VERIFIER,
        "terminal": True,
        "last_epoch": 60,
        "finite_best_last_weights_and_last_optimizer": True,
        "source_commit": source,
        "execution_code_sha256": execution,
        "checkpoint_training_source_sha256": training,
        "reviewed_protocol_sha256": m.sha(praw),
        "original_supervisor_terminal_sha256": m.sha(traw),
        "original_protocol_text": praw.decode(),
        "original_supervisor_terminal_text": traw.decode(),
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "minimum_photo_visits": 66,
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "all_eligible_annotations_contributed": True,
        "delivered_samples": 491520,
        "steps": 122880,
        "checkpoint_sha256": {"best.pt": "b" * 64, "last.pt": "a" * 64},
        "last_checkpoint_coverage": coverage,
        "best_checkpoint_coverage": dict(coverage, epoch=59),
        "selected_best_full_annotations": True,
        "configuration": configuration,
        "manifest_sha256": m.MANIFEST,
        "canonical_fingerprint": m.CANONICAL,
        "geometry_policy_sha256": m.POLICY,
        "derived_cache": cache,
    }
    evidence = {
        "kind": "unified_validation_evidence",
        "checkpoint": {"sha256": "a" * 64, "epoch": 60, "steps": 122880},
        "configuration": configuration,
        "source": {"training_source_sha256": training},
        "data": data,
        "inference_protocol": {
            "threshold": 0.5,
            "tile_size": 512,
            "overlap": 64,
            "precision": "float32_no_autocast",
            "cup_containment": "cup_and_disc",
        },
        "metrics": {"split": "val", "photos": 770, "threshold": 0.5, "per_photo": []},
    }
    return (
        evidence,
        qa,
        protocol,
        terminal,
        {
            "role": "fixed_terminal60",
            "contract": "source72",
            "protocol_raw": praw,
            "terminal_raw": traw,
        },
    )


def test_exact_last_and_own_full_best_roles(case):
    evidence, qa, protocol, terminal, kwargs = case
    assert (
        m.checked_candidate(evidence, qa, protocol, terminal, **kwargs)["checkpoint_sha256"]
        == "a" * 64
    )
    evidence["checkpoint"].update(sha256="b" * 64, epoch=59, steps=59 * 2048)
    kwargs["role"] = "best"
    assert (
        m.checked_candidate(evidence, qa, protocol, terminal, **kwargs)["checkpoint_sha256"]
        == "b" * 64
    )


@pytest.mark.parametrize(
    "defect",
    [
        "pool",
        "source",
        "terminal_source",
        "terminal_exit",
        "protocol_bytes",
        "role",
        "best_incomplete",
        "selected_incomplete",
        "config",
        "cache",
        "threshold",
    ],
)
def test_mixed_or_incomplete_provenance_rejected(case, defect):
    evidence, qa, protocol, terminal, kwargs = copy.deepcopy(case)
    if defect == "pool":
        qa["kind"] = "trusted_pool_completion"
    elif defect == "source":
        qa["source_commit"] = "0" * 40
    elif defect == "terminal_source":
        terminal["source"]["commit"] = "0" * 40
    elif defect == "terminal_exit":
        terminal["returncode"] = 1
    elif defect == "protocol_bytes":
        kwargs["protocol_raw"] += b"\n"
    elif defect == "role":
        kwargs["role"] = "best"
    elif defect == "best_incomplete":
        kwargs["role"] = "best"
        qa["selected_best_full_annotations"] = False
        evidence["checkpoint"].update(sha256="b" * 64, epoch=59, steps=59 * 2048)
    elif defect == "selected_incomplete":
        qa["last_checkpoint_coverage"]["contributing_annotation_records"] = 5699
    elif defect == "config":
        evidence["configuration"]["values"]["epochs"] = 27
    elif defect == "cache":
        qa["derived_cache"]["fingerprint"] = "0" * 64
    elif defect == "threshold":
        evidence["inference_protocol"]["threshold"] = 0.4
    with pytest.raises(ValueError):
        m.checked_candidate(evidence, qa, protocol, terminal, **kwargs)


def test_external_sha_failure_before_reading_referenced_files(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="manifest SHA"):
        m.compare(inputs, "0" * 64, output)
    assert not output.exists()


def test_existing_output_never_overwritten(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"kind": "reviewed_standalone_paired_val_inputs", "schema_version": 1})
    )
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    with pytest.raises(ValueError, match="Fresh output"):
        m.compare(inputs, m.sha(inputs.read_bytes()), output)
    assert marker.read_text() == "original"
