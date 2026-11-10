"""Wait without compute work; run the reviewed CPU-only disc validation once.

The normalized reference must finish successfully first. No retries, GPU, or
changes to existing processes. An optional HOLD file keeps the run deferred.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def ready(reference, pid, hold):
    if hold.exists():
        return False, "HOLD file present"
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        command = b""
    if b"evaluate_original.py" in command and b"--normalized" in command:
        return False, "Existing normalized reference still running"
    try:
        report = json.loads(reference.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "Successful complete normalized reference not available"
    if (
        report.get("split") != "val"
        or report.get("photos") != 770
        or not report.get("single_net_normalized")
        or report.get("preview_only")
        or report.get("manifest_sha256")
        != "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"
    ):
        return False, "Normalized reference result not complete expected val770"
    if os.getloadavg()[0] > max(1, (os.cpu_count() or 1) - 8):
        return False, "Waiting for CPU headroom"
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < 10 * 1024**2:
        return False, "Waiting for >=10GiB available memory"
    return True, "Normalized reference complete, CPU/RAM headroom checked"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-pid", type=int, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkout
    output = root / "reports/disc-reference/dedicated-disc-val.json"
    launch = output.with_suffix(".launch.json")
    if launch.exists() or output.exists():
        raise FileExistsError("Preserve earlier launch/results; never retry implicitly")
    command = [
        str(args.python),
        "-u",
        str(root / "scripts/evaluate_legacy_disc.py"),
        "--manifest",
        str(args.manifest),
        "--checkpoint",
        f"disc0={root}/src/deepretina/pretrained/legacy-single-net-disc-0.pt",
        "--checkpoint",
        f"disc1={root}/src/deepretina/pretrained/legacy-single-net-disc-1.pt",
        "--profiles",
        "raw",
        "normalized",
        "--threads",
        "4",
        "--output",
        str(output),
    ]
    record = {
        "status": "waiting_for_reference_and_resources",
        "waiter_pid": os.getpid(),
        "reference_pid": args.reference_pid,
        "reference": str(args.reference),
        "command": command,
        "threads": 4,
        "device": "cpu",
        "niceness": os.nice(0),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "script_sha256": hashlib.sha256(
            (root / "scripts/evaluate_legacy_disc.py").read_bytes()
        ).hexdigest(),
        "base_source_commit": "0d6d9087a92f0f07cf75fd1083c590e8af6bb18a",
    }
    while True:
        proceed, reason = ready(args.reference, args.reference_pid, output.parent / "HOLD")
        record["reason"] = reason
        record["last_checked_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        write(launch, record)
        if proceed:
            break
        time.sleep(60)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONPATH=str(root / "src"),
    )
    with output.with_suffix(".log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=root)
        record.update(
            status="running",
            pid=child.pid,
            launched_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        write(launch, record)
        code = child.wait()
    record.update(
        status="finished" if code == 0 else "failed_no_retry",
        returncode=code,
        finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    write(launch, record)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
