"""Actual completed standalone report metadata; no GT/model access."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vascular_compare", ROOT / "scripts/compare_vascular2_val.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
REPORTS = ROOT / "reports/vascular2-paired-val"


def inputs(role="best"):
    return (
        json.loads((REPORTS / (role + "-validation.json")).read_text()),
        json.loads((REPORTS / "standalone-completion-summary.json").read_text()),
    )


@pytest.mark.parametrize("role", ["best", "fixed_terminal60"])
def test_actual_roles_preserve_own_coverage(role):
    report, qa = inputs(role)
    value = module.checked_candidate(report, qa, role)
    assert value["photos"] == 770
    assert value["checkpoint_sha256"] == report["checkpoint"]["sha256"]


@pytest.mark.parametrize(
    "change", ["kind", "coverage", "checkpoint", "source", "sampling", "protocol"]
)
def test_no_retaging_or_relaxation(change):
    report, qa = inputs()
    if change == "kind":
        qa["original_qa_kind"] = "trusted_pool_completion_qa"
    elif change == "coverage":
        qa["best_checkpoint_coverage"]["contributing_annotation_records"] -= 1
    elif change == "checkpoint":
        report["checkpoint"]["sha256"] = qa["checkpoint_sha256"]["last.pt"]
    elif change == "source":
        report["source"]["training_source_sha256"] = "0" * 64
    elif change == "sampling":
        # Even internally consistent altered config is a different experiment.
        report["configuration"]["values"]["vascular_sampling_weight"] = 4
        qa["configuration"] = copy.deepcopy(report["configuration"])
    else:
        report["inference_protocol"]["threshold"] = 0.35
    with pytest.raises(ValueError):
        module.checked_candidate(report, qa, "best")


def test_worst6_is_pooled_six_objective_not_source_order():
    value = {
        "channels": {
            h: {"pooled_dice": {"candidate": 0.8, "reference": 0.5}} for h in module.CHANNELS
        },
        "av_accuracy": {"conditional_accuracy": {"candidate": 0.6, "reference": 0.9}},
    }
    result = module.objective(value)
    assert len(result["ratios"]) == 6
    assert result["bottleneck"] == "conditional_av"
    assert result["minimum_ratio"] == pytest.approx(2 / 3)


def test_actual_source_aggregates_reject_tampered_header():
    report, _ = inputs()
    module.validate_source_aggregates(report["metrics"])
    report["metrics"]["by_dataset"]["aria"]["vessel"]["fp"] += 1
    with pytest.raises(ValueError, match="Source aggregate"):
        module.validate_source_aggregates(report["metrics"])
