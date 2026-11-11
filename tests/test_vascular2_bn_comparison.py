"""Real completed research metadata; no model/GT/forward access."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "bn_counts", ROOT / "scripts/compare_vascular2_bn_val.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def actual():
    root = ROOT / "reports/vascular2-bn-recalibration"
    return (
        json.loads((root / "actual-val770/validation.json").read_text()),
        json.loads((root / "actual-train8192/calibration.json").read_text()),
    )


def test_real_research_family_preserves_770_counts():
    report, calibration = actual()
    view = m.derived_view(report, calibration)
    assert view["photos"] == 770
    assert view["checkpoint_sha256"] == report["checkpoint_sha256"]


@pytest.mark.parametrize("change", ["score", "parent_role", "non_bn", "family"])
def test_no_retaging_or_inherited_measurement(change):
    report, calibration = actual()
    if change == "score":
        report["training_selection_score"] = 0.9
    if change == "parent_role":
        report["derivation"]["parent_gradient_epoch"] = 60
    if change == "non_bn":
        calibration["non_bn_state_bitidentical"] = False
    if change == "family":
        report["kind"] = "unified_validation_evidence"
    with pytest.raises(ValueError):
        m.derived_view(report, calibration)
