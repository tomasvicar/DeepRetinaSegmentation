"""Frozen CLI delegation and nested metadata only; no Torch/model/GT/NN work."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "vascular_postcheck", ROOT / "reports/vascular2-val-readiness/postcheck_example.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_frozen_cli_delegates_without_pool_qa_dependency(monkeypatch):
    raw = subprocess.check_output(
        ["git", "show", "0b8762e:scripts/unified_validation.py"], cwd=ROOT
    )
    assert (
        hashlib.sha256(raw).hexdigest()
        == "824bab3412676a8a918d3efc8c863b7b85803e653ef7cf7c957c375fd1f17fbf"
    )
    calls = []
    torch = ModuleType("torch")
    torch.set_num_threads = lambda n: calls.append(("threads", n))
    selection = ModuleType("deepretina.selection")
    selection.SelectionError = ValueError
    validation = ModuleType("deepretina.validation_evidence")
    validation.produce_validation_evidence = lambda **kw: calls.append(kw)
    for name, value in [
        ("torch", torch),
        ("deepretina.selection", selection),
        ("deepretina.validation_evidence", validation),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    namespace = {"__name__": "frozen_readiness_fixture"}
    exec(compile(raw, "frozen0b/unified_validation.py", "exec"), namespace)  # noqa: S102 - exact byte-pinned user-owned CLI, all ML imports stubbed
    namespace["main"](
        [
            "--checkpoint",
            "/immutable/last.pt",
            "--manifest",
            "/manifest.json",
            "--output",
            "/new/validation.json",
            "--device",
            "cpu",
            "--cpu-threads",
            "4",
            "--threshold",
            ".5",
        ]
    )
    assert calls[0] == ("threads", 4)
    assert calls[1]["checkpoint"] == "/immutable/last.pt" and calls[1]["device"] == "cpu"
    assert "qa" not in calls[1] and "split" not in calls[1]


@pytest.fixture
def values():
    qa = {
        "kind": "standalone_unified_completion_qa",
        "terminal": True,
        "last_epoch": 60,
        "source_commit": m.SOURCE,
        "execution_code_sha256": m.EXECUTION,
        "checkpoint_training_source_sha256": m.TRAINING,
        "reviewed_protocol_sha256": m.PROTOCOL,
        "checkpoint_sha256": {"best.pt": "b" * 64, "last.pt": "l" * 64},
        "best_checkpoint_coverage": {"epoch": 52, "contributing_annotation_records": 5699},
        "last_checkpoint_coverage": {"epoch": 60, "contributing_annotation_records": 5700},
        "selected_best_full_annotations": False,
    }
    report = {
        "kind": "unified_validation_evidence",
        "metrics": {
            "split": "val",
            "photos": 770,
            "per_photo": [{"photo_id": str(i)} for i in range(770)],
        },
        "checkpoint": {"sha256": "l" * 64, "epoch": 60},
    }
    return report, qa


def test_nested_metrics_and_fixed_terminal_role(values):
    report, qa = values
    out = m.postcheck(report, qa, "fixed_terminal60")
    assert out["selected_checkpoint_coverage"]["contributing_annotation_records"] == 5700
    assert out["selected_best_full_annotations"] is False
    assert out["selection_or_test_authorized"] is False


def test_best_role_discloses_its_own_coverage(values):
    report, qa = values
    report["checkpoint"] = {"sha256": "b" * 64, "epoch": 52}
    assert (
        m.postcheck(report, qa, "best")["selected_checkpoint_coverage"][
            "contributing_annotation_records"
        ]
        == 5699
    )


@pytest.mark.parametrize("change", ["pool", "partial", "flat", "wrong_role_hash"])
def test_metadata_mismatch_rejects(values, change):
    report, qa = values
    if change == "pool":
        qa["kind"] = "trusted_pool_completion_qa"
    if change == "partial":
        qa["last_epoch"] = 59
    if change == "flat":
        report["photos"] = report["metrics"]["photos"]
        report.pop("metrics")
    if change == "wrong_role_hash":
        report["checkpoint"]["sha256"] = "b" * 64
    with pytest.raises((ValueError, KeyError)):
        m.postcheck(report, qa, "fixed_terminal60")
