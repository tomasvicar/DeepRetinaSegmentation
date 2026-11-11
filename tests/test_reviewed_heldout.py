"""Tiny synthetic stage2 forwards; no actual held-out data or GPU experiments."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_unified_selection import case as _canonical_case

from deepretina import heldout
from deepretina import selection as gate
from deepretina.integrity import prepared_content_fingerprint
from deepretina.validation_evidence import produce_validation_evidence

case = _canonical_case


@pytest.fixture(autouse=True)
def cpu_and_reader_guard(monkeypatch):
    previous = torch.get_num_threads()
    backend_before = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "dtype": torch.get_default_dtype(),
    }
    # The CLI starts a fresh interpreter. Keep this synthetic artifact fixture
    # at the same defaults even if earlier training tests changed global flags.
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA initialization"))
    state = {"gate_passed": False, "test_reads": 0}
    original = np.load

    def checked_read(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name.startswith(("x0-", "x1-")):
            assert state["gate_passed"], "Test GT decoded before selection gate passed"
            state["test_reads"] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_read)
    yield state
    torch.use_deterministic_algorithms(
        backend_before["deterministic"], warn_only=backend_before["warn_only"]
    )
    torch.set_float32_matmul_precision(backend_before["precision"])
    torch.backends.cuda.matmul.allow_tf32 = backend_before["cuda_tf32"]
    torch.backends.cudnn.allow_tf32 = backend_before["cudnn_tf32"]
    torch.backends.cudnn.benchmark = backend_before["benchmark"]
    torch.backends.cudnn.deterministic = backend_before["cudnn_deterministic"]
    torch.set_default_dtype(backend_before["dtype"])
    torch.set_num_threads(previous)


@pytest.fixture
def selected(case, tmp_path, monkeypatch, cpu_and_reader_guard):
    root = case["manifest"].parent
    rows = json.loads(case["manifest"].read_text())
    photos = json.loads((root / "photos.json").read_text())
    for row in list(rows):
        if row["photo_id"] == "x0":
            other = {
                **row,
                "id": row["id"].replace("x0", "x1"),
                "photo_id": "x1",
                "source_id": "subject-x1",
                "image": row["image"].replace("x0", "x1"),
                "mask": row["mask"].replace("x0", "x1"),
            }
            shutil.copyfile(root / row["image"], root / other["image"])
            shutil.copyfile(root / row["mask"], root / other["mask"])
            rows.append(other)
    shutil.copyfile(root / "x0-fov.png", root / "x1-fov.png")
    photos.append({"image": "x1.png", "fov": "x1-fov.png"})
    case["manifest"].write_bytes(gate.canonical(rows))
    (root / "photos.json").write_bytes(gate.canonical(photos))
    manifest_sha = gate.digest(case["manifest"].read_bytes())
    (root / "_SUCCESS").write_text(manifest_sha)
    content = prepared_content_fingerprint(case["manifest"])
    audit = json.loads((root / "integrity.json").read_text())
    audit.update(manifest_sha256=manifest_sha, prepared_content_sha256=content)
    (root / "integrity.json").write_bytes(gate.canonical(audit))
    for key in ["checkpoint", "last_checkpoint"]:
        payload = torch.load(case[key], map_location="cpu", weights_only=True)
        payload.update(manifest_sha256=manifest_sha, prepared_content_sha256=content)
        torch.save(payload, case[key])
    evidence = tmp_path / "actual-fresh-val.json"
    produce_validation_evidence(
        checkpoint=case["checkpoint"], manifest=case["manifest"], output=evidence, device="cpu"
    )
    inputs = {**case, "validation_evidence": evidence}
    selection = tmp_path / "selection.json"
    receipt = gate.create_selection(
        output=selection,
        reviewer="synthetic reviewer",
        rationale="synthetic fixture only",
        **inputs,
    )
    original = gate.check_selection

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        cpu_and_reader_guard["gate_passed"] = True
        return result

    monkeypatch.setattr(gate, "check_selection", checked)
    return {
        **inputs,
        "selection": selection,
        "expected_selection_sha256": receipt["selection_sha256"],
    }


def run(selected, tmp_path, **kwargs):
    return heldout.evaluate_reviewed_heldout(
        **selected, output=tmp_path / "test-report.json", execute_reviewed_heldout=True, **kwargs
    )


def test_actual_two_photo_cpu_report_and_metadata_boundary(
    selected, tmp_path, cpu_and_reader_guard
):
    report = run(selected, tmp_path)
    assert report["kind"] == "unified_reviewed_heldout_evaluation"
    assert report["execution_context"]["explicit_execution_requested"] is True
    assert report["execution_context"]["stage1_test_evaluation_enabled"] is False
    assert report["execution_context"]["prior_blindness_proven"] is False
    assert (
        json.loads(selected["selection"].read_text())["boundaries"]["test_evaluation_enabled"]
        is False
    )
    assert report["selection_sha256"] == selected["expected_selection_sha256"]
    assert report["checkpoint"]["sha256"] == gate.digest(selected["checkpoint"].read_bytes())
    assert report["canonical_data"]["eligible_photo_counts"]["test"] == 2
    assert report["metrics"]["split"] == "test" and report["metrics"]["photos"] == 2
    assert {photo["photo_id"] for photo in report["metrics"]["per_photo"]} == {"x0", "x1"}
    assert cpu_and_reader_guard["test_reads"] > 0
    assert "selection_score" not in report["metrics"]
    assert set(report["metrics"]["channels"]) == set(gate.CHANNELS)
    for channel, counts in report["metrics"]["channels"].items():
        assert counts["valid_pixels"] == 16
        assert counts["tp"] == sum(
            p["channels"][channel]["tp"] for p in report["metrics"]["per_photo"]
        )


def test_snapshot_only_after_gate_path_replacement(
    selected, tmp_path, monkeypatch, cpu_and_reader_guard
):
    snapshot = selected["checkpoint"].read_bytes()
    original_check = gate.check_selection
    expected, _, _ = heldout.load_unified(io.BytesIO(snapshot), device="cpu")

    def replacement(*args, **kwargs):
        result = original_check(*args, **kwargs)
        selected["checkpoint"].write_bytes(b"replaced after gate")
        return result

    monkeypatch.setattr(gate, "check_selection", replacement)
    load = heldout.load_unified

    def safe_load(stream, device):
        assert cpu_and_reader_guard["gate_passed"] and hasattr(stream, "read")
        model, payload, actual = load(stream, device)
        assert all(torch.equal(v, expected.state_dict()[k]) for k, v in model.state_dict().items())
        return model, payload, actual

    monkeypatch.setattr(heldout, "load_unified", safe_load)
    report = run(selected, tmp_path)
    assert report["checkpoint"]["sha256"] == gate.digest(snapshot)
    assert selected["checkpoint"].read_bytes() == b"replaced after gate"


@pytest.mark.parametrize(
    "kind",
    [
        "bad-artifact",
        "wrong-external-sha",
        "weights",
        "data",
        "source",
        "protocol",
        "thin-evidence",
    ],
)
def test_invalid_gate_no_test_reader_forward_or_cuda(
    selected, tmp_path, monkeypatch, cpu_and_reader_guard, kind
):
    if kind == "bad-artifact":
        selected["selection"].write_text("{}")
    elif kind == "wrong-external-sha":
        selected["expected_selection_sha256"] = "0" * 64
    elif kind == "weights":
        payload = torch.load(selected["checkpoint"], weights_only=True, map_location="cpu")
        next(iter(payload["state_dict"].values())).add_(1)
        torch.save(payload, selected["checkpoint"])
    elif kind == "data":
        (selected["manifest"].parent / "x0-av.npz").write_bytes(b"changed canonical test data")
    elif kind == "source":
        original = gate.evaluation_source_identity
        monkeypatch.setattr(
            gate,
            "evaluation_source_identity",
            lambda: {**original(), "evaluation_source_sha256": "f" * 64},
        )
    elif kind == "protocol":
        artifact = json.loads(selected["selection"].read_text())
        artifact["inference_protocol"]["threshold"] = 0.3
        selected["selection"].write_bytes(gate.canonical(artifact))
        selected["expected_selection_sha256"] = gate.digest(selected["selection"].read_bytes())
    else:
        selected["validation_evidence"].write_text('{"mean_dice":0.9}')
    monkeypatch.setattr(
        heldout,
        "load_unified",
        lambda *_a, **_k: pytest.fail("Execution loader must be behind gate"),
    )
    monkeypatch.setattr(
        heldout,
        "evaluate_unified_model",
        lambda *_a, **_k: pytest.fail("No test forward before valid gate"),
    )
    with pytest.raises((gate.SelectionError, ValueError, KeyError)):
        run(selected, tmp_path, device="cuda:0")
    assert not cpu_and_reader_guard["gate_passed"]
    assert cpu_and_reader_guard["test_reads"] == 0
    assert not (tmp_path / "test-report.json").exists()


def test_explicit_execution_request_is_separate_and_required(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(
        gate, "check_selection", lambda **_k: pytest.fail("No gate before explicit request")
    )
    with pytest.raises(gate.SelectionError, match="Explicit reviewed stage2"):
        heldout.evaluate_reviewed_heldout(**selected, output=tmp_path / "test-report.json")


def test_existing_output_refused_before_gate(selected, tmp_path, monkeypatch):
    output = tmp_path / "test-report.json"
    output.write_text("preserve")
    monkeypatch.setattr(
        gate, "check_selection", lambda **_k: pytest.fail("No gate for existing output")
    )
    with pytest.raises(gate.SelectionError, match="already exists"):
        run(selected, tmp_path)
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("when", ["gate", "forward", "emission"])
def test_canonical_lease_spans_gate_through_emission(
    selected, tmp_path, monkeypatch, cpu_and_reader_guard, when
):
    def mutate():
        path = selected["manifest"].parent / "x1.png"
        snapshot = path.read_bytes()
        path.write_bytes(snapshot + b"changed")
        path.write_bytes(snapshot)

    if when == "gate":
        original = gate.check_selection

        def changed(*a, **k):
            result = original(*a, **k)
            mutate()
            return result

        monkeypatch.setattr(gate, "check_selection", changed)
        monkeypatch.setattr(
            heldout,
            "load_unified",
            lambda *_a, **_k: pytest.fail("Lease must close before CUDA/load"),
        )
    elif when == "forward":
        original = heldout.evaluate_unified_model

        def changed(*a, **k):
            result = original(*a, **k)
            mutate()
            return result

        monkeypatch.setattr(heldout, "evaluate_unified_model", changed)
    else:
        original = Path.open

        def changed(path, *a, **k):
            stream = original(path, *a, **k)
            if path == tmp_path / "test-report.json" and a and a[0] == "xb":
                mutate()
            return stream

        monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(gate.SelectionError, match="changed"):
        run(selected, tmp_path)
    assert not (tmp_path / "test-report.json").exists()
    if when == "gate":
        assert cpu_and_reader_guard["test_reads"] == 0


def test_source_changes_during_forward_fail(selected, tmp_path, monkeypatch):
    identity = gate.evaluation_source_identity()
    original = heldout.evaluate_unified_model

    def changed(*a, **k):
        result = original(*a, **k)
        monkeypatch.setattr(
            gate,
            "evaluation_source_identity",
            lambda: {**identity, "evaluation_source_sha256": "f" * 64},
        )
        return result

    monkeypatch.setattr(heldout, "evaluate_unified_model", changed)
    with pytest.raises(gate.SelectionError, match="Source/runtime changed"):
        run(selected, tmp_path)
    assert not (tmp_path / "test-report.json").exists()


def test_missing_test_photo_or_wrong_gt_counts_rejected(selected, tmp_path, monkeypatch):
    original = heldout.evaluate_unified_model

    def incomplete(*a, **k):
        result = original(*a, **k)
        result["per_photo"] = result["per_photo"][:1]
        return result

    monkeypatch.setattr(heldout, "evaluate_unified_model", incomplete)
    with pytest.raises(gate.SelectionError, match="inventory"):
        run(selected, tmp_path)
    assert not (tmp_path / "test-report.json").exists()


def test_exclusive_output_race_preserves_other_report(selected, tmp_path, monkeypatch):
    original = heldout.evaluate_unified_model

    def race(*a, **k):
        result = original(*a, **k)
        (tmp_path / "test-report.json").write_text("other report")
        return result

    monkeypatch.setattr(heldout, "evaluate_unified_model", race)
    with pytest.raises(FileExistsError):
        run(selected, tmp_path)
    assert (tmp_path / "test-report.json").read_text() == "other report"


def test_validated_test_is_not_fake_val_relabel(selected, tmp_path, monkeypatch):
    original = heldout.evaluate_unified_model

    def relabeled(*a, **k):
        result = original(*a, **k)
        assert k["split"] == "test"
        result["split"] = "val"
        return result

    monkeypatch.setattr(heldout, "evaluate_unified_model", relabeled)
    with pytest.raises(gate.SelectionError, match="Require full-data TEST"):
        run(selected, tmp_path)


def test_cli_run_dir_and_no_unrecorded_protocol_overrides(selected, tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "unified_test.py"
    env = {
        **os.environ,
        "PYTHONPATH": str(script.parents[1] / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        str(script),
        "--run-dir",
        str(tmp_path),
        "--device",
        "cpu",
        "--cpu-threads",
        "1",
        "--output",
        str(tmp_path / "cli-test.json"),
    ]
    for key in [
        "checkpoint",
        "manifest",
        "selection",
        "expected_selection_sha256",
        "validation_evidence",
        "completion_family",
        "checkpoint_role",
        "completion_qa",
        "expected_qa_sha256",
        "best_checkpoint",
    ]:
        command.extend(["--" + key.replace("_", "-"), str(selected[key])])
    denied = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=30, check=False
    )
    assert denied.returncode == 2 and "execute-reviewed-heldout" in denied.stderr
    # The production subprocess must reject this toy corpus. Parser/delegation
    # success is exercised in-process with the explicit fixture eligibility mock;
    # no fixture bypass is added to production code.
    spec = importlib.util.spec_from_file_location("synthetic_unified_test_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(command[2:] + ["--execute-reviewed-heldout"])
    assert json.loads((tmp_path / "cli-test.json").read_text())["metrics"]["photos"] == 2
    result = subprocess.run(
        command + ["--execute-reviewed-heldout", "--threshold", ".3"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_receipt_metadata_false_is_not_silently_flipped(
    selected, tmp_path, monkeypatch, cpu_and_reader_guard
):
    from dataclasses import replace

    original = gate.check_selection

    def bad_receipt(*a, **k):
        return replace(original(*a, **k), test_evaluation_enabled=True)

    monkeypatch.setattr(gate, "check_selection", bad_receipt)
    monkeypatch.setattr(
        heldout,
        "load_unified",
        lambda *_a, **_k: pytest.fail("No execution with incompatible receipt"),
    )
    with pytest.raises(gate.SelectionError, match="metadata-only"):
        run(selected, tmp_path)
    assert cpu_and_reader_guard["test_reads"] == 0


@pytest.mark.parametrize("field", ["record_ids", "photo_id", "positive_gt", "valid_pixels"])
def test_actual_test_ids_and_gt_support_checked(selected, tmp_path, monkeypatch, field):
    original = heldout.evaluate_unified_model

    def changed(*a, **k):
        result = original(*a, **k)
        photo = result["per_photo"][0]
        if field == "record_ids":
            photo[field] = ["t0-av", "t0-disc_cup"]
        elif field == "photo_id":
            photo[field] = "v0"
        elif field == "positive_gt":
            photo["channels"]["cup"]["tp"] += 1
        else:
            photo["channels"]["cup"]["valid_pixels"] += 1
        return result

    monkeypatch.setattr(heldout, "evaluate_unified_model", changed)
    with pytest.raises(gate.SelectionError):
        run(selected, tmp_path)
    assert not (tmp_path / "test-report.json").exists()


def test_nondefault_frozen_protocol_is_reused_without_override_flags(selected, tmp_path):
    evidence = tmp_path / "alternate-val.json"
    overrides = {"threshold": 0.3, "tile_size": 40, "overlap": 4}
    produce_validation_evidence(
        checkpoint=selected["checkpoint"],
        manifest=selected["manifest"],
        output=evidence,
        device="cpu",
        **overrides,
    )
    selection = tmp_path / "alternate-selection.json"
    inputs = {
        key: value
        for key, value in selected.items()
        if key not in {"selection", "expected_selection_sha256"}
    }
    inputs["validation_evidence"] = evidence
    receipt = gate.create_selection(
        output=selection,
        reviewer="synthetic reviewer",
        rationale="fixture alternate protocol",
        **inputs,
        **overrides,
    )
    report = run(
        {
            **inputs,
            "selection": selection,
            "expected_selection_sha256": receipt["selection_sha256"],
        },
        tmp_path,
    )
    assert report["inference_protocol"]["threshold"] == report["metrics"]["threshold"] == 0.3
    assert report["inference_protocol"]["tile_size"] == 40
    assert report["inference_protocol"]["overlap"] == 4


def test_execution_load_is_safe_and_only_checked_snapshot(selected, tmp_path, monkeypatch):
    original = torch.load
    flags = []

    def safe(*a, **k):
        flags.append(k.get("weights_only"))
        return original(*a, **k)

    monkeypatch.setattr(torch, "load", safe)
    run(selected, tmp_path)
    assert len(flags) >= 5 and all(flag is True for flag in flags)


def test_concurrently_replaced_output_is_preserved_and_not_reported_success(
    selected, tmp_path, monkeypatch
):
    original = Path.open
    output = tmp_path / "test-report.json"

    def replaced(path, *a, **k):
        stream = original(path, *a, **k)
        if path == output and a and a[0] == "xb":
            path.unlink()
            path.write_text("replacement report")
        return stream

    monkeypatch.setattr(Path, "open", replaced)
    with pytest.raises(gate.SelectionError, match="output replaced"):
        run(selected, tmp_path)
    assert output.read_text() == "replacement report"


def test_whole_view_frozen_protocol_uses_no_tile_override(selected, tmp_path):
    for key in ["checkpoint", "last_checkpoint"]:
        payload = torch.load(selected[key], weights_only=True, map_location="cpu")
        payload["config"].update(train_view="whole", view_size=32)
        torch.save(payload, selected[key])
    evidence = tmp_path / "whole-val.json"
    produce_validation_evidence(
        checkpoint=selected["checkpoint"],
        manifest=selected["manifest"],
        output=evidence,
        device="cpu",
    )
    selection = tmp_path / "whole-selection.json"
    inputs = {
        key: value
        for key, value in selected.items()
        if key not in {"selection", "expected_selection_sha256"}
    }
    inputs["validation_evidence"] = evidence
    receipt = gate.create_selection(
        output=selection, reviewer="synthetic reviewer", rationale="whole fixture", **inputs
    )
    report = run(
        {
            **inputs,
            "selection": selection,
            "expected_selection_sha256": receipt["selection_sha256"],
        },
        tmp_path,
    )
    assert report["inference_protocol"]["mode"] == "whole"
    assert report["inference_protocol"]["tile_size"] is None
    assert report["inference_protocol"]["overlap"] is None
    assert report["inference_protocol"]["view_size"] == 32


def test_foreign_symlink_to_open_report_inode_is_preserved(selected, tmp_path, monkeypatch):
    original = Path.open
    output = tmp_path / "test-report.json"
    target = []

    def replaced(path, *a, **k):
        stream = original(path, *a, **k)
        if path == output and a and a[0] == "xb":
            target.append(f"/proc/self/fd/{stream.fileno()}")
            path.unlink()
            path.symlink_to(target[0])
        return stream

    monkeypatch.setattr(Path, "open", replaced)
    with pytest.raises(gate.SelectionError, match="output replaced"):
        run(selected, tmp_path)
    assert output.is_symlink()
    assert os.readlink(output) == target[0]


def test_actual_backend_mismatch_refused_before_test_read(
    selected, tmp_path, monkeypatch, cpu_and_reader_guard
):
    torch.use_deterministic_algorithms(True, warn_only=True)
    monkeypatch.setattr(
        heldout, "load_unified", lambda *_a, **_k: pytest.fail("No execution under changed backend")
    )
    with pytest.raises(gate.SelectionError, match="source/data/inference protocol"):
        run(selected, tmp_path)
    assert cpu_and_reader_guard["test_reads"] == 0
