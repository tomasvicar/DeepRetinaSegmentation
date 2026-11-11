"""Exercise only tiny stdlib child processes; never import Torch or real data."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "av_smoke", Path(__file__).parents[1] / "scripts/run_native_av_real_smoke_cpu.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def paths(tmp_path):
    return [tmp_path / name for name in ("launch.json", "exit.json", "log")]


def test_child_nonzero_is_not_success(tmp_path):
    assert (
        m.bounded(
            [sys.executable, "-c", "raise SystemExit(7)"],
            os.environ.copy(),
            *paths(tmp_path),
            memory_floor=0,
        )
        == 75
    )
    import json

    assert json.loads((tmp_path / "exit.json").read_text())["returncode"] == 7


def test_timeout_stops_owned_child(tmp_path):
    assert (
        m.bounded(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            os.environ.copy(),
            *paths(tmp_path),
            seconds=0.03,
            memory_floor=0,
        )
        == 75
    )
    import json

    result = json.loads((tmp_path / "exit.json").read_text())
    assert result["stop_reason"] == "deadline"
    assert result["returncode"] != 0


def test_post_spawn_metadata_failure_stops_child_and_preserves_foreign(tmp_path, monkeypatch):
    original = Path.write_text
    child_pid = []
    popen = m.subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        child_pid.append(child.pid)
        return child

    def write(path, *args, **kwargs):
        if path.name == "launch.json":
            raise OSError("synthetic ENOSPC")
        return original(path, *args, **kwargs)

    foreign = popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        monkeypatch.setattr(m.subprocess, "Popen", spawn)
        monkeypatch.setattr(Path, "write_text", write)
        assert (
            m.bounded(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                os.environ.copy(),
                *paths(tmp_path),
                memory_floor=0,
            )
            == 75
        )
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid[0], 0)
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=3)


def test_pending_cli_never_enters_worker(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(m.__file__)),
            "--phase",
            "train",
            "--data",
            str(tmp_path),
            "--runs",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "explicit root authorization" in result.stderr
    assert not list(tmp_path.iterdir())


def test_cleanup_has_at_most_two_bounded_waits(monkeypatch):
    class Child:
        pid = 987654321

        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            raise subprocess.TimeoutExpired("synthetic", timeout)

    child = Child()
    signals = []
    monkeypatch.setattr(m.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    m.stop_group(child)
    assert child.waits == [3, 3]
    assert len(signals) == 2
