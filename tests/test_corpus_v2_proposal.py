"""A proposal cannot turn frozen holdout photo/patient aliases into training."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/propose_corpus_v2.py"
spec = importlib.util.spec_from_file_location("corpus_v2_proposal", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("alias", ["photo", "source_group"])
def test_proposal_quarantines_v1_holdout_alias_and_preserves_v1(tmp_path, monkeypatch, alias):
    image = np.full((4, 6, 3), 100, np.uint8)
    photo = "sha256:" + module.pixel_sha256(image)
    old = {
        "id": "frozen",
        "dataset": "drions",
        "source_id": "drions:001",
        "photo_id": photo if alias == "photo" else "sha256:" + "e" * 64,
        "split": "test",
    }
    manifest = tmp_path / "manifest.json"
    raw = (json.dumps([old]) + "\n").encode()
    manifest.write_bytes(raw)
    monkeypatch.setattr(module, "V1_MANIFEST", hashlib.sha256(raw).hexdigest())

    class Source:
        names = ()

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def bytes(self, name):
            assert "test" not in name.lower()
            return name.encode()

    monkeypatch.setattr(module, "Source", Source)

    def adapter(source, emit, **kwargs):
        assert kwargs.get("splits", ("train",)) == ("train",)
        emit(
            "001",
            "train",
            image,
            np.ones((2, 4, 6), np.float32),
            source_id="drions:001",
            task="disc_cup",
            metadata={"source_image": "raw.png", "source_annotation": "annotation.png"},
        )

    for name in ["drions", "gamma", "idrid_disc"]:
        monkeypatch.setattr(module, name, adapter)
    for name in ["DRIONS-DB.zip", "GAMMA.zip", "IDRiD.zip"]:
        (tmp_path / name).write_bytes(b"dummy")
    progress = []
    result = module.proposal(tmp_path, manifest, progress.append)
    assert len(progress) == 3
    assert all(
        r["proposed_split"] == "quarantined_v1_holdout_alias" for r in result["candidate_records"]
    )
    assert result["counts"]["quarantined_holdout_alias_variants"] == 3
    assert manifest.read_bytes() == raw
    assert result["v1_masks_opened"] is False
    assert result["test_annotations_decoded"] is False
    assert not any(r["training_ready"] for r in result["candidate_records"])
