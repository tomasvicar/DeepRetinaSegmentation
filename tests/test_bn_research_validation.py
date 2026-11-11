"""Derived metadata/byte checks only; no actual calibration or VAL run."""

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_bn_recalibration as m


@pytest.fixture
def derived():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2)).eval()
    parent = {
        "config": {"fixture": True},
        "epoch": 60,
        "steps": 122880,
        "state_dict": copy.deepcopy(net.state_dict()),
        "lineage": {},
    }
    r = m.recipe.recalibrate(net, [torch.ones(4, 3, 4, 4)])
    r.update(
        parent_sha256=m.recipe.PARENT_SHA,
        recipe=m.recipe.RECIPE,
        helper_sha256=m.CALIBRATOR_SHA,
        delivered_samples=8192,
        photo_visits={str(i): 1 for i in range(2529)},
    )
    r["photo_visits"]["0"] += 8192 - 2529
    output = m.recipe.derived_payload(
        parent, net, parent_sha=m.recipe.PARENT_SHA, report=copy.deepcopy(r)
    )
    r["derived_sha256"] = "d" * 64
    return net, parent, output, r


def check(derived):
    net, parent, output, r = derived
    return m.validate_derivation(
        net, parent, output, r, parent_sha=m.recipe.PARENT_SHA, derived_sha="d" * 64
    )


def test_derived_parent_stats_and_coverage_checked_without_old_score(derived):
    assert check(derived)["independently_checked_non_bn_bitidentical"]
    assert check(derived)["selection_or_test_authorized"] is False


@pytest.mark.parametrize(
    "change", ["parameter", "stale_score", "derived_sha", "missing_photo", "wrong_recipe"]
)
def test_research_producer_rejects_false_derivation(derived, change):
    _, _, output, r = derived
    if change == "parameter":
        output["state_dict"]["0.weight"].add_(1)
    if change == "stale_score":
        output["val_selection_score"] = 0.9
    if change == "derived_sha":
        r["derived_sha256"] = "e" * 64
    if change == "missing_photo":
        r["photo_visits"].pop("1")
    if change == "wrong_recipe":
        r["recipe"] = {**r["recipe"], "samples": 4096}
    with pytest.raises(ValueError):
        check(derived)
