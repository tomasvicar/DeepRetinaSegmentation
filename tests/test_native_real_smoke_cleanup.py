"""Actual SIGTERM cleanup with owned sleep processes; no trainer/NN/data access."""

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_monitor_sigterm_reaps_owned_sleep_and_preserves_unrelated(tmp_path):
    worker = r"""
import importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
source=Path(sys.argv[1]);out=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location("smoke_runner",source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
real_popen=m.subprocess.Popen
children=[]
def only_sleep(command,**kwargs):
    if command[0]=="git":return real_popen(command,**kwargs)
    assert command[0]=="nice" and "deepretina.legacy_training_cli" in str(command)
    child=real_popen(["/bin/sleep","60"],**kwargs)
    children.append(child)
    (out/"owned-sleep.json").write_text(json.dumps({"pid":child.pid,"start_ticks":Path(f"/proc/{child.pid}/stat").read_text().split()[21]}))
    return child
m.subprocess.Popen=only_sleep
original_read=Path.read_text
def synthetic_meminfo(path,*args,**kwargs):
    return "MemAvailable: 8388608 kB\n" if str(path)=="/proc/meminfo" else original_read(path,*args,**kwargs)
Path.read_text=synthetic_meminfo
original_sleep=time.sleep
def ready_then_wait(_):
    print("READY_INSIDE_OWNED_TRY",flush=True)
    original_sleep(30)
m.time=SimpleNamespace(monotonic=time.monotonic,sleep=ready_then_wait)
try:
    m.run("disc",out/"never-opened-manifest.json",out,min(os.sched_getaffinity(0)))
finally:
    (out/"cleanup-status.json").write_text(json.dumps({"sleep_returncodes":[c.poll() for c in children],"NN_executed":False}))
"""
    unrelated = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
    monitor = subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker,
            str(ROOT / "scripts/run_native_real_smoke_cpu.py"),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        ready, _, _ = select.select([monitor.stdout], [], [], 5)
        assert ready, "Synthetic observer failed to enter guarded monitor loop"
        assert monitor.stdout.readline().strip() == "READY_INSIDE_OWNED_TRY"
        owned = json.loads((tmp_path / "owned-sleep.json").read_text())
        assert Path(f"/proc/{owned['pid']}").exists()
        monitor.send_signal(signal.SIGTERM)
        assert monitor.wait(timeout=8) == 128 + signal.SIGTERM
        status = json.loads((tmp_path / "cleanup-status.json").read_text())
        assert status == {"sleep_returncodes": [-signal.SIGTERM], "NN_executed": False}
        assert not Path(f"/proc/{owned['pid']}").exists(), "Owned child not reaped"
        assert unrelated.poll() is None, "Unrelated process was signalled"
        assert not (tmp_path / "disc").exists()
        assert not (tmp_path / "disc-observed-exit.json").exists(), (
            "Interrupted monitor forged normal exit"
        )
    finally:
        if monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(monitor.pid, signal.SIGKILL)
                monitor.wait(timeout=3)
        marker = tmp_path / "owned-sleep.json"
        if marker.exists():
            identity = json.loads(marker.read_text())
            proc = Path(f"/proc/{identity['pid']}")
            if proc.exists() and (proc / "stat").read_text().split()[21] == identity["start_ticks"]:
                os.killpg(identity["pid"], signal.SIGKILL)
        unrelated.terminate()
        unrelated.wait(timeout=3)
