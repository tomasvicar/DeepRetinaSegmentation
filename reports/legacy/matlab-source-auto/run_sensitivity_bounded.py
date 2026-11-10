"""Bounded isolated CPU2 AV sensitivity; original full forward unchanged."""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-dir", type=Path, required=True)
parser.add_argument("--model-root", type=Path, required=True)
args = parser.parse_args()
root = args.oracle_dir
start = time.monotonic()
peak = 0
reason = None
with (root / "sensitivity-v3-av.log").open("x") as log:
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("diaret_av_sensitivity.py")),
            "av",
            "--oracle-dir",
            str(root),
            "--model-root",
            str(args.model_root),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    while child.poll() is None:
        try:
            status = Path(f"/proc/{child.pid}/status").read_text().splitlines()
            rss = int(next(x for x in status if x.startswith("VmRSS:")).split()[1]) * 1024
            peak = max(peak, rss)
        except (FileNotFoundError, StopIteration):
            rss = 0
        if rss > 6 * 1024**3 or time.monotonic() - start > 300:
            reason = "RSS" if rss > 6 * 1024**3 else "time"
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            break
        time.sleep(0.1)
info = {
    "elapsed_seconds": time.monotonic() - start,
    "sampled_peak_rss_bytes": peak,
    "kernel_children_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    "rss_sample_seconds": 0.1,
    "rss_stop_bytes": 6 * 1024**3,
    "time_stop_seconds": 300,
    "returncode": child.returncode,
    "stop_reason": reason,
    "nice": os.getpriority(os.PRIO_PROCESS, 0),
    "forward": "unchanged converted legacy AV helper; no feature lifetime rewrite/tiling/downsampling",
}
(root / "sensitivity-resource-v3.json").write_text(json.dumps(info, indent=2) + "\n")
print(info)
if child.returncode:
    raise SystemExit(1)
