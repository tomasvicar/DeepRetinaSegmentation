"""Synthetic metadata only; no real checkpoint/model/GPU/manifest operations."""

import datetime
import importlib.util
import signal
import subprocess
from pathlib import Path

import pytest

p = Path(__file__).parents[1] / "scripts/watch_control_e060_thresholds.py"
s = importlib.util.spec_from_file_location("watch", p)
h = importlib.util.module_from_spec(s)
s.loader.exec_module(h)


def qa():
    return {
        "terminal": True,
        "original_pid": h.PID,
        "last_epoch": 60,
        "steps": 122880,
        "delivered_samples": 491520,
        "eligible_photos": 2221,
        "optimizer_max_step": 122880,
        "source_commit": h.SOURCE,
        "execution_code_sha256": h.EXECUTION,
        "checkpoint_training_source_sha256": h.TRAINING,
        "plan_sha256": h.PLAN,
        "manifest_sha256": h.MANIFEST,
        "canonical_fingerprint": h.CONTENT,
        "finite_best_last_weights_and_last_optimizer": True,
        "minimum_photo_visits": 60,
        "best_epoch": 55,
        "derived_cache": {"fingerprint": h.CACHE},
        "checkpoint_sha256": {"best.pt": "a" * 64, "last.pt": "b" * 64},
    }


def evidence(t=0.5):
    rows = [
        {
            "photo_id": str(i),
            "record_ids": [str(i)],
            "channel_datasets": {"cup": "toy"},
            "channels": {"cup": {"tp": 1, "fp": 2, "fn": 3, "valid_pixels": 10}},
            "av_conditional_counts": {
                "correct": 2,
                "eligible_pixels": 3,
                "scored_pixels": 3,
                "end_to_end_correct": 1,
            },
        }
        for i in range(770)
    ]
    return {
        "kind": "unified_validation_evidence",
        "checkpoint": {"sha256": "a" * 64, "epoch": 55},
        "configuration": {"sha256": "c" * 64},
        "source": {
            "evaluation_source_sha256": h.EVAL_PACKAGE,
            "training_source_sha256": h.TRAINING,
        },
        "data": {
            "manifest_sha256": h.MANIFEST,
            "prepared_content_sha256": h.CONTENT,
            "eligible_photo_counts": {"val": 770},
        },
        "inference_protocol": {
            "threshold": t,
            "geometry": "canonical_prepared",
            "preprocessing_profile": "legacy25_raw",
            "pixels_per_degree": 25,
            "tile_size": 512,
            "overlap": 64,
        },
        "metrics": {
            "split": "val",
            "photos": 770,
            "records": 993,
            "per_photo": rows,
            "evaluation_runtime": {
                "environment": {"CUDA_VISIBLE_DEVICES": h.GPU},
                "torch_cpu_threads": 2,
                "nvidia_smi_gpu_driver_rows": [f"0, {h.GPU}, GPUname, {h.DRIVER}"],
                "device_uuid": h.GPU,
            },
        },
    }


def test_wait_only_until_true_terminal_or_utc_cutoff():
    clock = [h.DEADLINE - datetime.timedelta(seconds=15)]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += datetime.timedelta(seconds=seconds)

    assert not h.wait_terminal(lambda _: True, sleep, lambda: clock[0])
    assert slept == [15]
    assert h.wait_terminal(
        lambda _: False,
        lambda _: pytest.fail("should notsleep"),
        lambda: h.DEADLINE - datetime.timedelta(seconds=1),
    )


@pytest.mark.parametrize("defect", ["source", "last", "samples", "optimizer", "coverage", "cache"])
def test_reject_terminal_qa_drift(defect):
    q = qa()
    if defect == "source":
        q["source_commit"] = "87"
    elif defect == "last":
        q["last_epoch"] = 59
    elif defect == "samples":
        q["delivered_samples"] = 491519
    elif defect == "optimizer":
        q["optimizer_max_step"] = 122879
    elif defect == "coverage":
        q["minimum_photo_visits"] = 59
    elif defect == "cache":
        q["derived_cache"]["fingerprint"] = "wrong"
    with pytest.raises(ValueError):
        h.check_qa(q)


def test_actual_gpu_uuid_driver_and_no_other_compute_are_required():
    valid = f"{h.GPU}, GPUname, {h.DRIVER}\n"
    assert h.gpu_free(valid, "")["uuid"] == h.GPU
    with pytest.raises(ValueError, match="GPU_BUSY"):
        h.gpu_free(valid, f"123, {h.GPU}, unrelated\n")
    with pytest.raises(ValueError):
        h.gpu_free(valid.replace(h.DRIVER, "wrong"), "")
    with pytest.raises(ValueError):
        h.gpu_free(valid.replace(h.GPU, "GPU-other"), "")


def test_six_reports_can_change_only_threshold_and_prediction_counts():
    first = evidence()
    second = evidence(0.35)
    second["metrics"]["per_photo"][0]["av_conditional_counts"]["end_to_end_correct"] = 2
    second["metrics"]["per_photo"][0]["channels"]["cup"].update(tp=2, fn=2, fp=3)
    h.check_qa(qa())
    h.check_evidence(first, 0.5, qa())
    h.check_evidence(second, 0.35, qa(), first)
    assert h.THRESHOLDS == [0.5, 0.35, 0.4, 0.45, 0.55, 0.6]


@pytest.mark.parametrize(
    "defect", ["weights", "source", "geometry", "GT", "conditional", "runtime", "thin"]
)
def test_reject_any_evidence_pin_support_or_runtime_drift(defect):
    first = evidence()
    second = evidence(0.4)
    if defect == "weights":
        second["checkpoint"]["sha256"] = "wrong"
    elif defect == "source":
        second["source"]["evaluation_source_sha256"] = "wrong"
    elif defect == "geometry":
        second["inference_protocol"]["tile_size"] = 256
    elif defect == "GT":
        second["metrics"]["per_photo"][0]["channels"]["cup"]["valid_pixels"] = 11
    elif defect == "conditional":
        second["metrics"]["per_photo"][0]["av_conditional_counts"]["correct"] = 1
    elif defect == "runtime":
        second["metrics"]["evaluation_runtime"]["torch_cpu_threads"] = 4
    elif defect == "thin":
        second["metrics"]["per_photo"].pop()
    with pytest.raises(ValueError):
        h.check_evidence(second, 0.4, qa(), first)


def test_timeout_signals_only_own_isolated_child_never_training(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "now", lambda: h.DEADLINE - datetime.timedelta(seconds=60))
    signals = []

    class Child:
        pid = 987654

        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("ownchild", timeout)
            return 0

    child = Child()

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(h.subprocess, "Popen", popen)
    monkeypatch.setattr(h.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="Own child"):
        h.run_child(["test"], {}, tmp_path / "log")
    assert signals == [(987654, signal.SIGTERM)] and all(pid != h.PID for pid, _ in signals)
