"""Bounded CPU-only observer; terminates only its newly created process group."""

import datetime
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def put(name, data):
    with (ROOT / name).open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    assert not (ROOT / "launch.json").exists(), "no retry"
    helper = ROOT / "owned-helper.py"
    assert (
        hashlib.sha256(helper.read_bytes()).hexdigest()
        == "382a39797a2695bfc5a4d55e7df1eb6e01e23db9d4fced30ca6021e67607a8a7"
    )
    spec = importlib.util.spec_from_file_location("owned", helper)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(ROOT / "source/src"),
    )
    command = [
        "nice",
        "-n",
        "17",
        "ionice",
        "-c",
        "3",
        "taskset",
        "-c",
        "4",
        "/var/home/tom/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python",
        str(ROOT / "driver.py"),
    ]
    started = time.monotonic()
    child = None
    error = None
    peak = 0
    timed_out = False
    rss_limit = False
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(s, f):
        raise KeyboardInterrupt(s)

    for s in handlers:
        signal.signal(s, interrupted)
    try:
        with (ROOT / "child.log").open("x") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            put(
                "launch.json",
                {
                    "observer_pid": os.getpid(),
                    "child_pid": child.pid,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "command": command,
                    "timeout_seconds": 590,
                    "whole_cap_seconds": 600,
                    "rss_cap_kib": 8 * 1024 * 1024,
                },
            )
            while child.poll() is None:
                rss = 0
                for pid in owned.live_group_members(child.pid):
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        rss += next(
                            (
                                int(x.split()[1])
                                for x in status.splitlines()
                                if x.startswith("VmRSS:")
                            ),
                            0,
                        )
                    except FileNotFoundError:
                        pass
                peak = max(peak, rss)
                timed_out = time.monotonic() - started > 590
                rss_limit = rss > 8 * 1024 * 1024
                if timed_out or rss_limit:
                    raise TimeoutError("time limit" if timed_out else "RSS limit")
                time.sleep(0.2)
    except BaseException as exc:
        error = repr(exc)
    finally:
        if child is not None:
            if owned.live_group_members(child.pid):
                owned.stop_owned_group(child)
            child.wait(timeout=3)
        for s, h in handlers.items():
            signal.signal(s, h)
        rc = None if child is None else child.returncode
        put(
            "observed-exit.json",
            {
                "returncode": rc,
                "observed_returncode": child is not None,
                "timed_out": timed_out,
                "rss_limit": rss_limit,
                "peak_group_rss_kib": peak,
                "error": error,
                "elapsed_seconds": time.monotonic() - started,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    return 0 if rc == 0 and error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
