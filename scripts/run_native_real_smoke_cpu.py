"""Bound one fresh native single-net CPU smoke; never change the production recipe."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PRESETS = {
    "disc": "native-legacy-disc.toml",
    "disc256": "native-legacy-disc-256.toml",
    "universal": "native-legacy-universal.toml",
}


def run(preset, manifest, output_root, core):
    source = Path(__file__).resolve().parents[1]
    output_root = Path(output_root)
    if not output_root.is_dir() or core not in os.sched_getaffinity(0):
        raise ValueError("Require existing run parent and an available CPU core")
    output = output_root / preset
    launch = output_root / (preset + "-launch.json")
    terminal = output_root / (preset + "-observed-exit.json")
    logpath = output_root / (preset + "-cli.log")
    if any(p.exists() for p in (output, launch, terminal, logpath)):
        raise FileExistsError("Preserve existing smoke artifacts; no automatic retry")
    command = [
        "nice",
        "-n",
        "17",
        "ionice",
        "-c",
        "3",
        "taskset",
        "-c",
        str(core),
        sys.executable,
        "-c",
        (
            "import torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);"
            "from deepretina.legacy_training_cli import main;main()"
        ),
        "--manifest",
        str(manifest),
        "--config",
        str(source / "configs" / PRESETS[preset]),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(source / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    start, peak, reason = time.monotonic(), 0, None
    with logpath.open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        launch.write_text(
            json.dumps(
                {
                    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "pid": child.pid,
                    "command": command,
                    "source_commit": subprocess.check_output(
                        ["git", "-C", str(source), "rev-parse", "HEAD"],
                        text=True,
                    ).strip(),
                    "maximum_seconds": 300,
                    "max_RSS_KiB": 2097152,
                    "minimum_MemAvailable_KiB": 2097152,
                    "device": "cpu",
                },
                indent=2,
            )
            + "\n"
        )
        while child.poll() is None:
            try:
                status = dict(
                    line.split(":", 1)
                    for line in Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    if ":" in line
                )
                peak = max(peak, int(status.get("VmRSS", "0 kB").split()[0]))
            except FileNotFoundError:
                pass
            memory = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            if time.monotonic() - start > 300:
                reason = "deadline"
            elif peak > 2097152:
                reason = "own_RSS_limit"
            elif int(memory["MemAvailable"].split()[0]) < 2097152:
                reason = "system_MemAvailable_limit"
            if reason:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = child.wait()
    receipt = {
        "kind": "actual_external_native_real_CPU_smoke_exit",
        "preset": preset,
        "observed_returncode": True,
        "returncode": code,
        "elapsed_seconds": time.monotonic() - start,
        "peak_RSS_KiB": peak,
        "stop_reason": reason,
        "final_quality_or_full_data_training": False,
    }
    terminal.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-core", type=int, default=4)
    args = parser.parse_args()
    raise SystemExit(run(args.preset, args.manifest, args.output_root, args.cpu_core))
