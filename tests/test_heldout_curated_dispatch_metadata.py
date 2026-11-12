"""Stage2 curated TTA metadata dispatch only: no weights, GT, forward or CUDA."""

import json

import pytest

from deepretina import heldout
from deepretina import selection as gate
from deepretina.inference_tta import protocol_extension


@pytest.mark.parametrize("mode", ["none", "rot4", "dihedral8"])
@pytest.mark.parametrize("role", ["training_best", "fixed_terminal60"])
def test_curated_stage2_forwards_exact_original_index_role_and_frozen_protocol(
    tmp_path, monkeypatch, mode, role
):
    protocol = {"threshold": 0.35, "tile_size": 512, "overlap": 64, **protocol_extension(mode)}
    artifact = {
        "schema_version": 2,
        "checkpoint_role": role,
        "freeze_eligibility": {"family": "curated_fd44"},
        "inference_protocol": protocol,
    }
    path = tmp_path / "synthetic-selection.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        gate, "evaluation_source_identity", lambda: {"synthetic_metadata_only": True}
    )
    monkeypatch.setattr(gate, "_canonical_metadata", lambda path: {})

    class ReachedTypedOriginalGate(Exception):
        pass

    def checked(**kwargs):
        assert kwargs["checkpoint_role"] == role
        assert kwargs["completion_family"] == "curated_fd44"
        assert kwargs["original_curated_index"] == "original-index-not-read.json"
        assert kwargs["expected_curated_index_sha256"] == "a" * 64
        assert kwargs["completion_qa"] == "original-qa-not-read.json"
        assert kwargs["expected_qa_sha256"] == "b" * 64
        assert kwargs["tta"] == mode and kwargs["threshold"] == 0.35
        assert kwargs["tile_size"] == 512 and kwargs["overlap"] == 64
        raise ReachedTypedOriginalGate

    monkeypatch.setattr(gate, "check_selection", checked)
    monkeypatch.setattr(heldout, "load_unified", lambda *a, **k: pytest.fail("No model load"))
    monkeypatch.setattr(
        heldout, "evaluate_unified_model", lambda *a, **k: pytest.fail("No GT/forward")
    )
    with pytest.raises(ReachedTypedOriginalGate):
        heldout.evaluate_reviewed_heldout(
            selection=path,
            expected_selection_sha256=gate.digest(path.read_bytes()),
            checkpoint="not-read.pt",
            best_checkpoint="not-read-best.pt",
            last_checkpoint="not-read-last.pt",
            manifest=tmp_path / "canonical" / "absent-manifest.json",
            validation_evidence="not-read-val.json",
            metrics_csv="not-read.csv",
            training_summary="not-read-summary.json",
            output=tmp_path / "must-not-exist.json",
            execute_reviewed_heldout=True,
            checkpoint_role=role,
            completion_family="curated_fd44",
            completion_qa="original-qa-not-read.json",
            expected_qa_sha256="b" * 64,
            original_curated_index="original-index-not-read.json",
            expected_curated_index_sha256="a" * 64,
        )
    assert not (tmp_path / "must-not-exist.json").exists()
