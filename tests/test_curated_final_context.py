"""Synthetic metadata wiring only; genuine original-chain tests live separately."""

import json
from pathlib import Path

import pytest
import torch
from test_selection_tta import prepared
from test_unified_selection import case as canonical_case

from deepretina import curated_completion as curated
from deepretina import freeze_eligibility as eligibility
from deepretina import selection as gate
from deepretina import validation_evidence as producer

case = canonical_case


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *a, **k: pytest.fail("No NN"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA"))


def context_inputs(case, monkeypatch, mode):
    inputs = prepared(case, mode, "curated_fd44")
    for path in {inputs["checkpoint"], inputs["last_checkpoint"]}:
        payload = torch.load(path, weights_only=True)
        payload["training_source_sha256"] = curated.SOURCE["training_sha256"]
        torch.save(payload, path)
    snapshot, payload = gate.checkpoint_snapshot(inputs["checkpoint"])
    report = json.loads(inputs["validation_evidence"].read_text())
    report["checkpoint"]["sha256"] = gate.digest(snapshot)
    report["source"]["training_source_sha256"] = payload["training_source_sha256"]
    original = {"scope": "mocked original-checker boundary, never actual curated proof", "epoch": 1}
    report["original_curated_completion"] = original
    inputs["validation_evidence"].write_bytes(gate.canonical(report))
    inputs.update(
        original_curated_index=Path("synthetic-index-only"), expected_curated_index_sha256="a" * 64
    )
    original_check = eligibility.check_freeze_eligibility

    def checked(**kwargs):
        assert kwargs["original_curated_index"] == inputs["original_curated_index"]
        assert kwargs["expected_curated_index_sha256"] == "a" * 64
        result = original_check(**kwargs)  # Existing fixture mocks only the eligibility boundary.
        proof = result.proof
        proof.update(original_curated_completion=original, input_sha256={"curated_index": "a" * 64})
        return eligibility.CheckedFreezeEligibility(
            result.checkpoint_snapshot, gate.canonical(proof)
        )

    monkeypatch.setattr(eligibility, "check_freeze_eligibility", checked)
    return inputs


@pytest.mark.parametrize("mode", ["none", "flips2", "rot4", "dihedral8"])
def test_curated_context_requires_exact_fresh_original_chain(case, monkeypatch, tmp_path, mode):
    inputs = context_inputs(case, monkeypatch, mode)
    path = tmp_path / "synthetic-selection.json"
    result = gate.create_selection(
        output=path, reviewer="fixture", rationale="no actual decision", **inputs
    )
    gate.check_selection(
        selection=path, expected_selection_sha256=result["selection_sha256"], **inputs
    )
    report = json.loads(inputs["validation_evidence"].read_text())
    report["original_curated_completion"]["epoch"] = True  # bool must not alias integer1.
    inputs["validation_evidence"].write_bytes(gate.canonical(report))
    with pytest.raises(gate.SelectionError, match="original curated completion binding"):
        gate._context(**inputs)


def test_frozen_curated_proof_json_types_and_external_index_pin(case, monkeypatch, tmp_path):
    inputs = context_inputs(case, monkeypatch, "rot4")
    path = tmp_path / "synthetic-selection.json"
    result = gate.create_selection(
        output=path, reviewer="fixture", rationale="no actual decision", **inputs
    )
    with pytest.raises(gate.SelectionError, match="external curated index pin"):
        gate.check_selection(
            selection=path,
            expected_selection_sha256=result["selection_sha256"],
            **dict(inputs, expected_curated_index_sha256="b" * 64),
        )
    artifact = json.loads(path.read_text())
    artifact["freeze_eligibility"]["original_curated_completion"]["epoch"] = True
    path.write_bytes(gate.canonical(artifact))
    with pytest.raises(gate.SelectionError, match="Frozen freeze_eligibility"):
        gate.check_selection(
            selection=path, expected_selection_sha256=gate.digest(path.read_bytes()), **inputs
        )


@pytest.mark.parametrize(
    "bad", ["missing", "wrong_role", "foreign_path", "foreign_source", "bad_pin"]
)
def test_producer_rejects_before_any_model_or_gt(case, monkeypatch, tmp_path, bad):
    payload = torch.load(case["checkpoint"], weights_only=True)
    payload["training_source_sha256"] = curated.SOURCE["training_sha256"]
    torch.save(payload, case["checkpoint"])
    index = tmp_path / "index.json"
    spec = {"path": str(case["checkpoint"]), "sha256": gate.digest(case["checkpoint"].read_bytes())}
    index.write_bytes(
        gate.canonical({"files": {"best": spec, "last": {"path": "foreign", "sha256": "0" * 64}}})
    )
    args = {
        "original_curated_index": index,
        "expected_curated_index_sha256": gate.digest(index.read_bytes()),
        "checkpoint_role": "training_best",
    }
    if bad == "missing":
        args = {}
    if bad == "wrong_role":
        args["checkpoint_role"] = "fixed_terminal60"
    if bad == "bad_pin":
        args["expected_curated_index_sha256"] = "0" * 64
    if bad == "foreign_source":
        payload["training_source_sha256"] = "f" * 64
        torch.save(payload, case["checkpoint"])
    if bad == "foreign_path":
        other = tmp_path / "identical.pt"
        other.write_bytes(case["checkpoint"].read_bytes())
        case = dict(case, checkpoint=other)
    monkeypatch.setattr(
        curated,
        "check_curated_completion",
        lambda **kwargs: pytest.fail("Must reject before original checker"),
    )
    monkeypatch.setattr(producer, "load_unified", lambda *a, **k: pytest.fail("No model load"))
    with pytest.raises((gate.SelectionError, FileNotFoundError)):
        producer.produce_validation_evidence(
            checkpoint=case["checkpoint"],
            manifest=case["manifest"],
            output=tmp_path / "not-created.json",
            tta="rot4",
            **args,
        )
    assert not (tmp_path / "not-created.json").exists()


@pytest.mark.parametrize("mode", ["none", "flips2", "rot4", "dihedral8"])
def test_producer_enters_inference_only_after_typed_original_snapshot(
    case, monkeypatch, tmp_path, mode
):
    payload = torch.load(case["checkpoint"], weights_only=True)
    payload["training_source_sha256"] = curated.SOURCE["training_sha256"]
    torch.save(payload, case["checkpoint"])
    snapshot = case["checkpoint"].read_bytes()
    index = tmp_path / "index.json"
    index.write_bytes(
        gate.canonical(
            {"files": {"best": {"path": str(case["checkpoint"]), "sha256": gate.digest(snapshot)}}}
        )
    )
    observed = []

    def checked(**kwargs):
        observed.append(kwargs)
        return curated.CheckedCuratedCompletion(
            snapshot, gate.canonical({"scope": "mock boundary"})
        )

    monkeypatch.setattr(curated, "check_curated_completion", checked)

    class ReachedCheckedConfiguration(Exception):
        pass

    def stop(payload):
        assert len(observed) == 1
        raise ReachedCheckedConfiguration

    monkeypatch.setattr(gate, "config_identity", stop)
    with pytest.raises(ReachedCheckedConfiguration):
        producer.produce_validation_evidence(
            checkpoint=case["checkpoint"],
            manifest=case["manifest"],
            output=tmp_path / "not-created.json",
            tta=mode,
            original_curated_index=index,
            expected_curated_index_sha256=gate.digest(index.read_bytes()),
            checkpoint_role="training_best",
        )
    assert observed[0]["family"] == "curated_fd44"
    assert observed[0]["capture_snapshots"] is True
