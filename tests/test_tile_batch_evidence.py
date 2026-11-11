"""Research-only tile batching provenance, no CLI/default schema promotion."""

import pytest
from test_unified_selection import case  # noqa: F401

from deepretina.validation_evidence import produce_validation_evidence


def test_tile_batch_only_report_is_explicitly_experimental(case, tmp_path):  # noqa: F811
    result = produce_validation_evidence(
        checkpoint=case["checkpoint"],
        manifest=case["manifest"],
        output=tmp_path / "tile-evidence.json",
        device="cpu",
        tile_batch_size=2,
    )
    assert result["kind"] == "experimental_unified_tile_batch_validation_evidence"
    assert result["inference_protocol"]["tile_batching"]["size"] == 2
    assert result["metrics"]["tile_batching"]["original_standalone_certification"] is False
    assert "tta" not in result["inference_protocol"]


def test_invalid_batch_rejected_before_checkpoint_or_output(tmp_path):
    target = tmp_path / "never-created.json"
    with pytest.raises(ValueError, match="tile_batch_size"):
        produce_validation_evidence(
            checkpoint=tmp_path / "missing.pt",
            manifest=tmp_path / "missing.json",
            output=target,
            tile_batch_size=True,
        )
    assert not target.exists()
