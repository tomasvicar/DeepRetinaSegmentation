"""Reviewed TTA metadata/math only: no network forward, actual GT or TEST scoring."""

import copy
import json

import numpy as np
import pytest
import torch
from test_unified_selection import case as canonical_case

from deepretina import heldout, unified
from deepretina import selection as gate
from deepretina import validation_evidence as producer
from deepretina.completion_profiles import PROFILES
from deepretina.inference_tta import evidence_kind, mode_from_protocol, protocol_extension

case = canonical_case


@pytest.fixture(autouse=True)
def no_forward(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *a, **k: pytest.fail("No NN"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("No CUDA"))
    yield
    torch.set_num_threads(before)


def rewrite(path, value):
    path.write_bytes(gate.canonical(value))


def prepared(case, mode="rot4", family="standalone_72"):
    """Synthetic metadata fixture; original typed completion stays independently tested."""
    case = dict(case, completion_family=family, tta=mode)
    # Checkpoint identities are genuine safe toy payloads, not historical retags.
    for path in {case["checkpoint"], case["last_checkpoint"]}:
        payload = torch.load(path, weights_only=True)
        payload["training_source_sha256"] = PROFILES["standalone_72"][3]
        torch.save(payload, path)
    raw, payload = gate.checkpoint_snapshot(case["checkpoint"])
    report = json.loads(case["validation_evidence"].read_text())
    report["kind"] = evidence_kind(mode)
    report["checkpoint"]["sha256"] = gate.digest(raw)
    report["source"]["training_source_sha256"] = payload["training_source_sha256"]
    report["inference_protocol"] = gate.inference_protocol(payload, tta=mode)
    report["metrics"].update(protocol_extension(mode))
    rewrite(case["validation_evidence"], report)
    return case


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
@pytest.mark.parametrize("family", ["standalone_72", "pool_72_v3"])
def test_explicit_typed_tta_freeze_roundtrip(case, tmp_path, mode, family):
    inputs = prepared(case, mode, family)
    path = tmp_path / "freeze.json"
    receipt = gate.create_selection(
        output=path, reviewer="fixture", rationale="synthetic only", **inputs
    )
    checked = gate.check_selection(
        selection=path, expected_selection_sha256=receipt["selection_sha256"], **inputs
    )
    assert checked.protocol["tta"] == protocol_extension(mode)["tta"]
    assert checked.checkpoint_role == "training_best"
    assert checked.test_evaluation_enabled is False
    # Absence/another mode cannot silently select a different test method.
    for wrong in ["none", "dihedral8" if mode == "rot4" else "rot4"]:
        with pytest.raises(gate.SelectionError):
            gate.check_selection(
                selection=path,
                expected_selection_sha256=receipt["selection_sha256"],
                **dict(inputs, tta=wrong),
            )


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "activation",
        "fusion",
        "source",
        "stale",
        "kind",
        "metric_tta",
        "threshold",
        "tile",
        "checkpoint",
    ],
)
def test_changed_fresh_evidence_rejected(case, tmp_path, field):
    inputs = prepared(case)
    report = json.loads(inputs["validation_evidence"].read_text())
    if field == "mode":
        report["inference_protocol"]["tta"]["mode"] = "dihedral8"
    if field == "activation":
        report["inference_protocol"]["activation"][1] = "sigmoid"
    if field == "fusion":
        report["inference_protocol"]["tta"]["combination"] = "mean_conditional_AV"
    if field == "source":
        report["source"]["training_source_sha256"] = "0" * 64
    if field == "stale":
        report["source"]["evaluation_source_sha256"] = "0" * 64
    if field == "kind":
        report["kind"] = "unified_validation_evidence"
    if field == "metric_tta":
        report["metrics"]["tta"]["precision"] = "bfloat16"
    if field == "threshold":
        report["inference_protocol"]["threshold"] = 0.35
    if field == "tile":
        report["inference_protocol"]["tile_size"] = 768
    if field == "checkpoint":
        report["checkpoint"]["sha256"] = "0" * 64
    rewrite(inputs["validation_evidence"], report)
    with pytest.raises(gate.SelectionError):
        gate.create_selection(
            output=tmp_path / "bad.json", reviewer="fixture", rationale="fixture", **inputs
        )


@pytest.mark.parametrize("family", ["pool_6b", "standalone_0b", "curated_fd44"])
def test_tta_other_family_not_implicitly_admitted(case, tmp_path, family):
    inputs = prepared(case, family=family)
    with pytest.raises(gate.SelectionError, match="TTA"):
        gate.create_selection(
            output=tmp_path / "bad.json", reviewer="fixture", rationale="fixture", **inputs
        )


@pytest.mark.parametrize("change", ["QA", "coverage"])
def test_tta_still_requires_typed_original_eligibility(case, tmp_path, monkeypatch, change):
    from deepretina import freeze_eligibility

    inputs = prepared(case)
    # Assert the unchanged external QA/fullcoverage checker runs, not a TTA bypass.
    original = freeze_eligibility.check_freeze_eligibility

    def deny(**kwargs):
        assert kwargs["expected_qa_sha256"] == inputs["expected_qa_sha256"]
        original(**kwargs)
        raise gate.SelectionError("Original typed " + change + " rejected")

    monkeypatch.setattr(freeze_eligibility, "check_freeze_eligibility", deny)
    with pytest.raises(gate.SelectionError, match=change):
        gate.create_selection(
            output=tmp_path / "bad.json", reviewer="fixture", rationale="fixture", **inputs
        )


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_fresh_producer_dispatch_without_any_forward(case, tmp_path, monkeypatch, mode):
    inputs = prepared(case, mode)
    fixture = json.loads(inputs["validation_evidence"].read_text())
    calls = []

    def metadata_only(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["split"] == "val" and kwargs["tta"] == mode
        return copy.deepcopy(fixture["metrics"])

    monkeypatch.setattr(producer, "evaluate_unified_model", metadata_only)
    report = producer.produce_validation_evidence(
        checkpoint=inputs["checkpoint"],
        manifest=inputs["manifest"],
        output=tmp_path / "new.json",
        tta=mode,
    )
    assert len(calls) == 1 and report["kind"] == evidence_kind(mode)
    assert report["inference_protocol"]["tta"] == protocol_extension(mode)["tta"]


def test_none_default_protocol_and_report_schema_exact(case):
    payload = gate.checkpoint_snapshot(case["checkpoint"])[1]
    old = json.loads(case["validation_evidence"].read_text())
    assert gate.inference_protocol(payload) == old["inference_protocol"]
    assert gate.inference_protocol(payload, tta="none") == old["inference_protocol"]
    assert evidence_kind("none") == old["kind"] and protocol_extension("none") == {}


@pytest.mark.parametrize(
    "mutation", ["mode", "order", "fusion", "precision", "experimental", "none_field"]
)
def test_bad_frozen_descriptor_rejected_before_stage2_checker(
    case, tmp_path, monkeypatch, mutation
):
    protocol = gate.inference_protocol(gate.checkpoint_snapshot(case["checkpoint"])[1], tta="rot4")
    if mutation == "mode":
        protocol["tta"]["mode"] = "unknown"
    if mutation == "order":
        protocol["tta"]["operators"].reverse()
    if mutation == "fusion":
        protocol["tta"]["combination"] = "average_classes"
    if mutation == "precision":
        protocol["tta"]["precision"] = "bf16"
    if mutation == "experimental":
        protocol["tta"]["experimental"] = True
    if mutation == "none_field":
        protocol["tta"] = {"mode": "none"}
    artifact = {
        "schema_version": 2,
        "checkpoint_role": "training_best",
        "freeze_eligibility": {"family": "standalone_72"},
        "inference_protocol": protocol,
    }
    path = tmp_path / "bad-frozen.json"
    rewrite(path, artifact)
    monkeypatch.setattr(
        gate, "check_selection", lambda **k: pytest.fail("Bad protocol reached selection/load")
    )
    with pytest.raises((ValueError, TypeError)):
        heldout.evaluate_reviewed_heldout(
            selection=path,
            expected_selection_sha256=gate.digest(path.read_bytes()),
            output=tmp_path / "never.json",
            execute_reviewed_heldout=True,
            **{
                k: v
                for k, v in dict(case, completion_family="standalone_72").items()
                if k not in {"tta"}
            },
        )


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_rectangular_inverse_and_joint_fusion_tiny_math(monkeypatch, mode):
    image = torch.linspace(0, 1, 3 * 5 * 7).reshape(3, 5, 7)
    observed = []

    def toy(_model, transformed, *_args):
        observed.append(transformed.shape)
        value = transformed[0].numpy().astype(np.float32)
        # Equivariant synthetic probabilities, including variable vessel strength.
        return np.stack([value, 0.2 + 0.6 * value, 0.8 - 0.6 * value, value * 0.7, value * 0.4])

    monkeypatch.setattr(unified, "_predict_probabilities", toy)
    expected = unified._conditional_probabilities(unified._joint_probabilities(toy(None, image)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = unified.predict_probabilities(None, image, "cpu", tta=mode)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert (
        result.dtype == np.float32
        and len(observed) == 1 + ({"flips2": 2, "rot4": 4, "dihedral8": 8}[mode])
    )
    if mode != "flips2":
        assert (3, 7, 5) in observed


@pytest.mark.parametrize("mode", ["flips2", "rot4", "dihedral8"])
def test_stage2_passes_frozen_mode_before_any_scoring(case, tmp_path, monkeypatch, mode):
    inputs = prepared(case, mode)
    path = tmp_path / "freeze.json"
    receipt = gate.create_selection(output=path, reviewer="fixture", rationale="fixture", **inputs)

    class BeforeScoring(Exception):
        pass

    def intercepted(*args, **kwargs):
        assert kwargs["tta"] == mode and kwargs["split"] == "test"
        raise BeforeScoring("intercepted before any split scoring or forward")

    monkeypatch.setattr(heldout, "evaluate_unified_model", intercepted)
    with pytest.raises(BeforeScoring):
        heldout.evaluate_reviewed_heldout(
            selection=path,
            expected_selection_sha256=receipt["selection_sha256"],
            output=tmp_path / "never.json",
            execute_reviewed_heldout=True,
            **{k: v for k, v in inputs.items() if k != "tta"},
        )
    assert not (tmp_path / "never.json").exists()


def test_zero_vessel_and_joint_class_mixture(monkeypatch):
    draws = iter([(0.1, 0.9), (0.9, 0.1)])

    def toy(*args):
        vessel, artery = next(draws)
        return np.array([vessel, artery, 1 - artery, 0.7, 0.3], np.float32)[:, None, None]

    monkeypatch.setattr(unified, "_predict_probabilities", toy)
    result = unified.predict_probabilities(None, torch.zeros(3, 1, 1), "cpu", tta="flips2")
    assert result[0, 0, 0] == pytest.approx(0.5)
    assert result[1, 0, 0] == pytest.approx(0.18)  # Joint mixture, not mean conditional .5.


def test_operator_types_are_not_bool_int_aliases():
    p = protocol_extension("rot4")
    p["tta"]["operators"][1]["rot90_ccw"] = True
    with pytest.raises(ValueError):
        mode_from_protocol(p)


def test_none_preserves_exact_return_object_and_rng(monkeypatch):
    token = np.zeros((5, 2, 3), np.float32)
    calls = []

    def same(*args):
        calls.append(args)
        return token

    monkeypatch.setattr(unified, "_predict_probabilities", same)
    before = torch.get_rng_state().clone()
    assert unified.predict_probabilities(None, torch.zeros(3, 2, 3), "cpu") is token
    assert len(calls) == 1 and torch.equal(torch.get_rng_state(), before)


def test_bf16_training_metadata_does_not_enable_tta_autocast(monkeypatch):
    calls = []

    def fp32(*args):
        calls.append(torch.is_autocast_enabled("cpu"))
        return np.array([0.7, 0.3, 0.7, 0.8, 0.2], np.float32)[:, None, None]

    monkeypatch.setattr(unified, "_predict_probabilities", fp32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = unified.predict_probabilities(
            None, torch.zeros(3, 1, 1), "cpu", config={"mixed_precision": "bf16"}, tta="rot4"
        )
    assert calls == [False] * 4 and result.dtype == np.float32


def test_non72_producer_rejected_before_forward(case, tmp_path, monkeypatch):
    monkeypatch.setattr(
        producer,
        "evaluate_unified_model",
        lambda *a, **k: pytest.fail("Wrong training source reached producer"),
    )
    with pytest.raises(gate.SelectionError, match="source72"):
        producer.produce_validation_evidence(
            checkpoint=case["checkpoint"],
            manifest=case["manifest"],
            output=tmp_path / "bad.json",
            tta="rot4",
        )


def test_declared_operators_match_actual_prediction_order():
    for mode in ("flips2", "rot4", "dihedral8"):
        operators = protocol_extension(mode)["tta"]["operators"]
        assert [(o["rot90_ccw"], o["horizontal_flip_after_rotation"]) for o in operators] == list(
            unified.TTA_OPERATORS[mode]
        )


def test_header_only_bool_rotation_rejected_with_unchanged_metrics(case, tmp_path):
    inputs = prepared(case)
    report = json.loads(inputs["validation_evidence"].read_text())
    original_metrics = copy.deepcopy(report["metrics"])
    report["inference_protocol"]["tta"]["operators"][1]["rot90_ccw"] = True
    assert report["metrics"] == original_metrics
    rewrite(inputs["validation_evidence"], report)
    with pytest.raises(gate.SelectionError, match="TTA header"):
        gate.create_selection(
            output=tmp_path / "bad-header.json", reviewer="fixture", rationale="fixture", **inputs
        )
