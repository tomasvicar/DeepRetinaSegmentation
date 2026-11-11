"""Fresh tiny CPU forwards, never actual corpus/model evaluation or test GT reads."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_unified_selection import case as _canonical_case

from deepretina import selection as gate
from deepretina import validation_evidence as producer

case = _canonical_case  # Reuse the canonical fixture without its module's no-forward guard.


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA initialization"))
    original = np.load

    def no_test_gt(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name.startswith("x0-"):
            pytest.fail("VAL producer must not decode test GT")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", no_test_gt)
    yield
    torch.set_num_threads(previous)


def produce(case, tmp_path, **kwargs):
    return producer.produce_validation_evidence(
        checkpoint=case["checkpoint"],
        manifest=case["manifest"],
        output=tmp_path / "fresh-val.json",
        **kwargs,
    )


def test_actual_forward_report_accepted_by_selection_gate(case, tmp_path):
    before = gate.datetime.now(gate.UTC)
    result = produce(case, tmp_path)
    assert gate.utc(result["evaluated_at_utc"]) >= before
    runtime = result["metrics"]["evaluation_runtime"]
    assert (
        before
        <= gate.utc(runtime["evaluation_started_at_utc"])
        <= gate.utc(runtime["evaluation_finished_at_utc"])
        <= gate.utc(result["evaluated_at_utc"])
    )
    assert result["metrics"]["photos"] == 1
    assert result["metrics"]["per_photo"][0]["photo_id"] == "v0"
    assert result["metrics"]["evaluation_runtime"]["device"] == "cpu"
    assert result["metrics"]["evaluation_runtime"]["hardware_attestation"] is False
    assert result["checkpoint"]["sha256"] == gate.digest(case["checkpoint"].read_bytes())
    assert result["source"] == {
        **gate.evaluation_source_identity(),
        "training_source_sha256": gate.checkpoint_snapshot(case["checkpoint"])[1][
            "training_source_sha256"
        ],
    }
    inputs = {**case, "validation_evidence": tmp_path / "fresh-val.json"}
    receipt = gate.create_selection(
        output=tmp_path / "selection.json",
        reviewer="synthetic reviewer",
        rationale="fixture only",
        **inputs,
    )
    checked = gate.check_selection(
        selection=tmp_path / "selection.json",
        expected_selection_sha256=receipt["selection_sha256"],
        **inputs,
    )
    assert checked.checkpoint_sha256 == result["checkpoint"]["sha256"]


def test_path_replacement_does_not_change_evaluated_weights(case, tmp_path, monkeypatch):
    snapshot = case["checkpoint"].read_bytes()
    expected, _payload, _device = producer.load_unified(io.BytesIO(snapshot), device="cpu")
    original = producer.load_unified
    observed = []

    def replace_path(stream, device):
        assert hasattr(stream, "read")
        case["checkpoint"].write_bytes(b"replaced path after snapshot")
        model, payload, actual = original(stream, device)
        observed.append(payload["checkpoint_sha256"])
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected.state_dict()[key])
        return model, payload, actual

    monkeypatch.setattr(producer, "load_unified", replace_path)
    result = produce(case, tmp_path)
    assert result["checkpoint"]["sha256"] == gate.digest(snapshot) == observed[0]
    assert case["checkpoint"].read_bytes() == b"replaced path after snapshot"


def test_existing_output_refused_before_checkpoint_or_model(case, tmp_path, monkeypatch):
    path = tmp_path / "fresh-val.json"
    path.write_text("preserve report")
    monkeypatch.setattr(
        gate,
        "checkpoint_snapshot",
        lambda *_: pytest.fail("checkpoint must not be opened"),
    )
    monkeypatch.setattr(
        producer, "evaluate_unified_model", lambda *_a, **_k: pytest.fail("no forward")
    )
    with pytest.raises(gate.SelectionError, match="already exists"):
        produce(case, tmp_path)
    assert path.read_text() == "preserve report"


@pytest.mark.parametrize("mutation", ["canonical", "source"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_live_identity_changes_fail_without_report(case, tmp_path, monkeypatch, mutation, when):
    original = producer.evaluate_unified_model
    if mutation == "source":
        identity = gate.evaluation_source_identity()
        calls = [0]

        def changed_source():
            calls[0] += 1
            return (
                identity
                if calls[0] < (2 if when == "before" else 3)
                else {**identity, "evaluation_source_sha256": "f" * 64}
            )

        monkeypatch.setattr(gate, "evaluation_source_identity", changed_source)
    elif when == "before":
        load = producer.load_unified

        def mutate_before(*args, **kwargs):
            result = load(*args, **kwargs)
            (case["manifest"].parent / "v0.png").write_bytes(b"changed before VAL")
            return result

        monkeypatch.setattr(producer, "load_unified", mutate_before)

    def evaluate(*args, **kwargs):
        assert when != "before", "Identity should reject before forward"
        result = original(*args, **kwargs)
        if mutation == "canonical":
            path = case["manifest"].parent / "v0.png"
            snapshot = path.read_bytes()
            path.write_bytes(snapshot + b"changed")
            path.write_bytes(snapshot)
        return result

    monkeypatch.setattr(producer, "evaluate_unified_model", evaluate)
    with pytest.raises((gate.SelectionError, ValueError), match="changed|fingerprint|mismatch"):
        produce(case, tmp_path)
    assert not (tmp_path / "fresh-val.json").exists()


def test_missing_val_photo_counts_rejected(case, tmp_path, monkeypatch):
    original = producer.evaluate_unified_model

    def incomplete(*a, **k):
        result = original(*a, **k)
        result["per_photo"] = []
        return result

    monkeypatch.setattr(producer, "evaluate_unified_model", incomplete)
    with pytest.raises(gate.SelectionError, match="inventory"):
        produce(case, tmp_path)
    assert not (tmp_path / "fresh-val.json").exists()


def test_exclusive_output_race_preserves_other_report(case, tmp_path, monkeypatch):
    original = producer.evaluate_unified_model

    def race(*a, **k):
        result = original(*a, **k)
        (tmp_path / "fresh-val.json").write_text("other report")
        return result

    monkeypatch.setattr(producer, "evaluate_unified_model", race)
    with pytest.raises(FileExistsError):
        produce(case, tmp_path)
    assert (tmp_path / "fresh-val.json").read_text() == "other report"


@pytest.mark.parametrize("epoch", [1, 60])
def test_partial_or_final_epoch_allowed_under_planned60(case, tmp_path, epoch):
    payload = torch.load(case["checkpoint"], weights_only=True, map_location="cpu")
    payload["epoch"] = epoch
    torch.save(payload, case["checkpoint"])
    result = produce(case, tmp_path)
    assert result["checkpoint"]["epoch"] == epoch


@pytest.mark.parametrize("kind", ["pilot", "historical", "wrong-budget", "manifest"])
def test_ineligible_inputs_fail_before_forward(case, tmp_path, monkeypatch, kind):
    payload = torch.load(case["checkpoint"], weights_only=True, map_location="cpu")
    if kind == "pilot":
        payload["config"]["purpose"] = payload["purpose"] = "functionality"
    elif kind == "historical":
        payload["lineage"]["historical_exposure"] = "unknown"
    elif kind == "wrong-budget":
        payload["config"]["epochs"] = 59
    else:
        payload["manifest_sha256"] = "0" * 64
    torch.save(payload, case["checkpoint"])
    monkeypatch.setattr(
        producer, "evaluate_unified_model", lambda *_a, **_k: pytest.fail("no forward")
    )
    with pytest.raises((gate.SelectionError, ValueError)):
        produce(case, tmp_path)
    assert not (tmp_path / "fresh-val.json").exists()


def test_cli_actual_tiny_cpu_roundtrip(case, tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "unified_validation.py"
    env = {
        **os.environ,
        "PYTHONPATH": str(script.parents[1] / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    output = tmp_path / "cli-val.json"
    command = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(case["checkpoint"]),
        "--manifest",
        str(case["manifest"]),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--cpu-threads",
        "1",
        "--tile-size",
        "40",
        "--overlap",
        "4",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["inference_protocol"]["tile_size"] == 40
    assert report["inference_protocol"]["overlap"] == 4
    for forbidden in ["--split", "--validation-evidence"]:
        result = subprocess.run(
            command + [forbidden, "test"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 2


def test_snapshot_is_read_once_and_all_loads_are_safe(case, tmp_path, monkeypatch):
    read_bytes = Path.read_bytes
    calls = []

    def tracked_read(path):
        if path == case["checkpoint"]:
            calls.append(path)
        return read_bytes(path)

    safe_load = torch.load
    flags = []

    def tracked_load(*args, **kwargs):
        flags.append(kwargs.get("weights_only"))
        return safe_load(*args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    monkeypatch.setattr(torch, "load", tracked_load)
    produce(case, tmp_path)
    assert calls == [case["checkpoint"]]
    assert flags == [True, True]


@pytest.mark.parametrize("tta", ["flips2", "rot4", "dihedral8"])
def test_tta_report_is_explicit_experiment_and_rejected_for_standalone_selection(
    case, tmp_path, tta
):
    result = produce(case, tmp_path, tta=tta)
    assert result["kind"] == "experimental_unified_tta_validation_evidence"
    assert result["inference_protocol"]["tta"] == result["metrics"]["tta"]
    assert result["inference_protocol"]["tta"]["mode"] == tta
    assert result["inference_protocol"]["tta"]["original_standalone_certification"] is False
    inputs = {**case, "validation_evidence": tmp_path / "fresh-val.json"}
    with pytest.raises(gate.SelectionError, match="evidence"):
        gate.create_selection(
            output=tmp_path / "selection.json",
            reviewer="synthetic reviewer",
            rationale="fixture only",
            **inputs,
        )
    assert not (tmp_path / "selection.json").exists()
