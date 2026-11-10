"""One authorized terminal60 QA then six fresh VAL reports; no TEST/selection."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
import tomllib
from pathlib import Path

PID = 1257093
DEADLINE = datetime.datetime(2026, 11, 10, 20, 0, tzinfo=datetime.UTC)
QA_SHA = "e88c2c17f9b9503fe1b99ea3f34e011e3b7289df0d62d827bd31fcef9a1f7f15"
THRESHOLD_PLAN = "bf238040e1023c7d6a4dd04d63dc50e47630fd5d58c1a3e5b0b5555dfe7797c8"
EVAL_COMMIT = "8c56594f36a92e569e901fcce0ed99ab2179629f"
EVAL_PACKAGE = "3779e1b5e9577d6870cf5e9a3ee48fb2271941dbbca12891f7a7f002075c4c9c"
PRODUCER = "824bab3412676a8a918d3efc8c863b7b85803e653ef7cf7c957c375fd1f17fbf"
GPU = "GPU-d0675605-e6c2-1c85-9799-08b279aab0ff"
DRIVER = "615.78.08"
THRESHOLDS = [0.5, 0.35, 0.4, 0.45, 0.55, 0.6]
CONTENT = "61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e"
MANIFEST = "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"
SOURCE = "3157bb957fcf6e6f632a5c49d6df2c8a78ea3747"
EXECUTION = "8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b"
TRAINING = "dd69012a1741000ae5841634094094397c690a2371c9a91aa411344a436f1136"
PLAN = "6be48b57fc85e8ffba7772058562aa5a8bf855f9cb445c5180484111ffba5778"
CACHE = "4e39e40def03d0f2859dd147107a3736a62b4fc0249dd440a1280a9803aa8591"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.datetime.now(datetime.UTC)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_terminal(check, sleep, clock):
    while clock() < DEADLINE:
        if not check(PID):
            return True
        sleep(min(15, max(0, (DEADLINE - clock()).total_seconds())))
    return False


def package_hash(root):
    result = hashlib.sha256()
    package = root / "src/deepretina"
    for p in sorted(package.rglob("*.py"), key=lambda p: p.relative_to(package).as_posix()):
        result.update(
            ("src/deepretina/" + p.relative_to(package).as_posix()).encode()
            + b"\0"
            + p.read_bytes()
        )
    for name in ["pyproject.toml", "uv.lock"]:
        result.update(name.encode() + b"\0" + (root / name).read_bytes())
    return result.hexdigest()


def check_qa(r):
    for key, value in {
        "terminal": True,
        "original_pid": PID,
        "last_epoch": 60,
        "steps": 122880,
        "delivered_samples": 491520,
        "eligible_photos": 2221,
        "optimizer_max_step": 122880,
        "source_commit": SOURCE,
        "execution_code_sha256": EXECUTION,
        "checkpoint_training_source_sha256": TRAINING,
        "plan_sha256": PLAN,
        "manifest_sha256": MANIFEST,
        "canonical_fingerprint": CONTENT,
        "finite_best_last_weights_and_last_optimizer": True,
    }.items():
        require(r.get(key) == value, "TerminalQA mismatch: " + key)
    require(
        r["minimum_photo_visits"] >= 60 and 1 <= r["best_epoch"] <= 60,
        "Invalid coverage/best epoch",
    )
    require(r["derived_cache"]["fingerprint"] == CACHE, "QA cache differs")
    for name in ["best.pt", "last.pt"]:
        require(len(r["checkpoint_sha256"][name]) == 64, "Missing checked checkpoint SHA")


def gpu_free(gpu_csv, compute_csv):
    rows = list(csv.reader(gpu_csv.splitlines()))
    matches = [r for r in rows if r and r[0].strip() == GPU]
    require(len(matches) == 1 and matches[0][-1].strip() == DRIVER, "Physical UUID/driver differs")
    computes = list(csv.reader(compute_csv.splitlines()))
    require(
        not any(len(r) >= 2 and r[1].strip() == GPU for r in computes),
        "GPU_BUSY: unrelated/residual compute observed; no launch/no kill",
    )
    return {"uuid": GPU, "driver": DRIVER, "gpu_rows": rows, "compute_rows": computes}


def photo_support(metrics):
    return [
        (
            r["photo_id"],
            r["record_ids"],
            r["channel_datasets"],
            {c: [v["valid_pixels"], v["tp"] + v["fn"]] for c, v in r["channels"].items()},
            r["av_conditional_counts"],
        )
        for r in metrics["per_photo"]
    ]


def check_evidence(e, threshold, qa, baseline=None):
    require(
        e["kind"] == "unified_validation_evidence"
        and e["checkpoint"]["sha256"] == qa["checkpoint_sha256"]["best.pt"]
        and e["checkpoint"]["epoch"] == qa["best_epoch"],
        "Evidence checkpoint differs",
    )
    require(
        e["source"]["evaluation_source_sha256"] == EVAL_PACKAGE
        and e["source"]["training_source_sha256"] == TRAINING,
        "Evidence source differs",
    )
    require(
        e["data"]["manifest_sha256"] == MANIFEST
        and e["data"]["prepared_content_sha256"] == CONTENT
        and e["data"]["eligible_photo_counts"]["val"] == 770,
        "Evidence canonical data differs",
    )
    p = e["inference_protocol"]
    require(
        p["threshold"] == threshold
        and p["geometry"] == "canonical_prepared"
        and p["preprocessing_profile"] == "legacy25_raw"
        and p["pixels_per_degree"] == 25
        and p["tile_size"] == 512
        and p["overlap"] == 64,
        "Geometry/threshold differs",
    )
    m = e["metrics"]
    require(len(m["per_photo"]) == 770, "Missing per-photo VAL inventory")
    require(
        m["split"] == "val" and m["photos"] == 770 and m["records"] == 993, "VAL inventory differs"
    )
    runtime = m["evaluation_runtime"]
    require(
        runtime["environment"]["CUDA_VISIBLE_DEVICES"] == GPU and runtime["torch_cpu_threads"] == 2,
        "Actual producer device/threads differ",
    )
    require(
        any(GPU in row and DRIVER in row for row in runtime["nvidia_smi_gpu_driver_rows"]),
        "Missing actual observed physical CUDA UUID/driver",
    )
    property_uuid = runtime.get("device_uuid", "unavailable")
    if property_uuid != "unavailable":
        require(
            property_uuid.lower().removeprefix("gpu-") == GPU.lower().removeprefix("gpu-"),
            "Torch observed CUDA UUID differs",
        )
    if baseline is not None:
        for key in ["checkpoint", "configuration", "source", "data"]:
            require(e[key] == baseline[key], "Across-threshold pin mismatch: " + key)
        require(
            {k: v for k, v in p.items() if k != "threshold"}
            == {k: v for k, v in baseline["inference_protocol"].items() if k != "threshold"},
            "Across-threshold geometry changed",
        )
        old = baseline["metrics"]
        require(
            photo_support(m) == photo_support(old),
            "GT/source/observer or conditionalAV counts changed across thresholds",
        )
        for key in [
            "device",
            "device_name",
            "device_uuid",
            "nvidia_smi_gpu_driver_rows",
            "torch_cpu_threads",
            "torch_cpu_interop_threads",
            "environment",
            "torch_cuda_build",
            "cudnn_version",
        ]:
            require(
                runtime.get(key) == old["evaluation_runtime"].get(key),
                "Across-threshold evaluationruntime changed: " + key,
            )


def run_child(command, env, log):
    remaining = (DEADLINE - now()).total_seconds()
    require(remaining > 0, "20:00UTC deadline reached; no new command")
    with log.open("x") as stream:
        child = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise RuntimeError(
                "Own child reached20:00UTC deadline; partial outputs preserved"
            ) from None
    require(code == 0, "Childfailed; no retry: " + str(command))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-helper", type=Path, required=True)
    p.add_argument("--threshold-plan", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "Fresh watcher directory required")
    a.output.mkdir(parents=True)

    def event(phase, **fields):
        record = {"utc": now().isoformat(), "phase": phase, **fields}
        with (a.output / "events.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    home = Path.home()
    source = home / "code/DeepRetinaSegmentation_unified_run_3157bb9"
    evaluation = home / "code/DeepRetinaSegmentation_evaluation_8c56594"
    trials = home / "code/DeepRetinaSegmentation_unified_trials_3157bb9"
    canonical_root = home / "code/DeepRetinaSegmentation_for_Leipzig/data/unified25"
    manifest = canonical_root / "manifest.json"
    milestone = trials / "milestones/lr_0003-control-cached-seed42/epoch-060"
    qa_path = trials / "lr0003-control-cached-e060-verified.json"
    python = home / "code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python"
    cache = home / ("code/DeepRetinaSegmentation_derived_storage/full/" + CONTENT + "/v1-train-val")
    try:
        require(
            sha(a.qa_helper) == QA_SHA and sha(a.threshold_plan) == THRESHOLD_PLAN,
            "Reviewed helper/thresholdplan bytes changed",
        )
        event(
            "watching_terminal",
            pid=PID,
            deadline_utc=DEADLINE.isoformat(),
            qa_helper_sha256=QA_SHA,
            threshold_plan_sha256=THRESHOLD_PLAN,
        )
        if not wait_terminal(alive, time.sleep, now):
            event("deadline_training_still_live_no_action")
            return
        event("terminal_observed", pid=PID)
        require(
            sha(a.qa_helper) == QA_SHA and sha(a.threshold_plan) == THRESHOLD_PLAN,
            "Reviewed execution artifacts changed while waiting",
        )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(source / "src"),
        )
        qa_command = [
            "nice",
            "-n",
            "10",
            str(python),
            str(a.qa_helper),
            "--source",
            str(source),
            "--run",
            str(trials / "runs/lr_0003-control-cached-seed42"),
            "--plan",
            str(trials / "plan-av-residual-cached-approved.json"),
            "--manifest",
            str(manifest),
            "--cache",
            str(cache),
            "--milestone",
            str(milestone),
            "--report",
            str(qa_path),
        ]
        save(
            a.output / "qa-command.json",
            {
                "command": qa_command,
                "environment": {
                    k: env[k]
                    for k in [
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "PYTHONPATH",
                    ]
                },
            },
        )
        run_child(qa_command, env, a.output / "qa.log")
        qa = json.loads(qa_path.read_text())
        check_qa(qa)
        require(
            sha(milestone / "best.pt") == qa["checkpoint_sha256"]["best.pt"]
            and sha(milestone / "last.pt") == qa["checkpoint_sha256"]["last.pt"],
            "Independent snapshot differs",
        )
        event(
            "qa_pass",
            best_epoch=qa["best_epoch"],
            best_sha256=qa["checkpoint_sha256"]["best.pt"],
            last_sha256=qa["checkpoint_sha256"]["last.pt"],
            qa_report=str(qa_path),
            qa_sha256=sha(qa_path),
        )
        save(
            a.output / "qa-pass-notification.json",
            {
                "qa_report": str(qa_path),
                "qa_report_sha256": sha(qa_path),
                "best_epoch": qa["best_epoch"],
                "best_checkpoint_sha256": qa["checkpoint_sha256"]["best.pt"],
                "last60_checkpoint_sha256": qa["checkpoint_sha256"]["last.pt"],
            },
        )
        for file in milestone.iterdir():
            require(file.is_file() and not file.is_symlink(), "Unexpected milestone entry")
            file.chmod(0o444)
        milestone.chmod(0o555)
        require(
            subprocess.check_output(
                ["git", "-C", str(evaluation), "rev-parse", "HEAD"], text=True
            ).strip()
            == EVAL_COMMIT,
            "Frozen8c commit differs",
        )
        require(
            not subprocess.check_output(
                ["git", "-C", str(evaluation), "status", "--porcelain", "--untracked-files=no"]
            ).strip(),
            "Frozen8c tracked source changed",
        )
        require(
            package_hash(evaluation) == EVAL_PACKAGE
            and sha(evaluation / "scripts/unified_validation.py") == PRODUCER,
            "Frozen8c source/producer bytes differ",
        )
        probe_env = dict(
            env,
            PYTHONPATH=str(evaluation / "src"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        probe_command = [
            str(python),
            "-c",
            "import json;from deepretina.selection import evaluation_source_identity;print(json.dumps(evaluation_source_identity()))",
        ]
        probe = json.loads(subprocess.check_output(probe_command, env=probe_env, text=True))
        require(
            probe["evaluation_source_sha256"] == EVAL_PACKAGE, "Actual imported evalpackage differs"
        )
        locked = {
            p["name"].lower(): p["version"]
            for p in tomllib.loads((evaluation / "uv.lock").read_text())["package"]
        }
        for name, version in probe["runtime"]["installed_distributions"].items():
            if version is not None:
                require(
                    locked.get(name.lower()) == version,
                    "Actual installed dependency differs from frozenlock: " + name,
                )
        observation = gpu_free(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ),
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ),
        )
        addendum = {
            "kind": "preexecution_VAL_threshold_addendum_NOT_selection",
            "threshold_plan_sha256": THRESHOLD_PLAN,
            "qa_helper_sha256": QA_SHA,
            "qa_report_sha256": sha(qa_path),
            "checkpoint": {
                "path": str(milestone / "best.pt"),
                "sha256": sha(milestone / "best.pt"),
                "selected_epoch": qa["best_epoch"],
            },
            "completed_training": {
                "source": SOURCE,
                "last_epoch": 60,
                "last_sha256": sha(milestone / "last.pt"),
                "steps": 122880,
                "scheduler": {
                    "T_max": 60,
                    "last_epoch": 60,
                    "proof": "Reviewed e88 terminalhelper checked this exact last checkpoint",
                },
            },
            "milestone_permissions": "files0444 directory0555",
            "evaluation_commit": EVAL_COMMIT,
            "evaluation_source": probe,
            "producer_sha256": PRODUCER,
            "canonical_fingerprint": CONTENT,
            "manifest_sha256": MANIFEST,
            "canonical_photo_geometry_sha256": sha(canonical_root / "photos.json"),
            "device_observation": observation,
            "threshold_order": THRESHOLDS,
            "test_evaluation": False,
            "selection_or_autopromotion": False,
        }
        save(a.output / "threshold-addendum.json", addendum)
        event("threshold_addendum_frozen", sha256=sha(a.output / "threshold-addendum.json"))
        baseline = None
        summaries = []
        val_env = dict(probe_env, CUDA_VISIBLE_DEVICES=GPU)
        for threshold in THRESHOLDS:
            observation = gpu_free(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,name,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ),
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid,process_name",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ),
            )
            require(
                package_hash(evaluation) == EVAL_PACKAGE
                and sha(evaluation / "scripts/unified_validation.py") == PRODUCER
                and sha(milestone / "best.pt") == qa["checkpoint_sha256"]["best.pt"],
                "Source/weights changed beforeVAL",
            )
            name = f"threshold-{round(threshold * 100):03d}"
            report = a.output / (name + ".json")
            command = [
                "nice",
                "-n",
                "10",
                str(python),
                str(evaluation / "scripts/unified_validation.py"),
                "--checkpoint",
                str(milestone / "best.pt"),
                "--manifest",
                str(manifest),
                "--output",
                str(report),
                "--device",
                "cuda:0",
                "--cpu-threads",
                "2",
                "--threshold",
                str(threshold),
            ]
            save(
                a.output / (name + "-command.json"),
                {
                    "command": command,
                    "environment": {
                        k: val_env[k]
                        for k in [
                            "CUDA_VISIBLE_DEVICES",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "PYTHONPATH",
                            "CUBLAS_WORKSPACE_CONFIG",
                        ]
                    },
                    "prelaunch_GPU": observation,
                },
            )
            event("validation_start", threshold=threshold)
            run_child(command, val_env, a.output / (name + ".log"))
            e = json.loads(report.read_text())
            require(
                {k: v for k, v in e["source"].items() if k != "training_source_sha256"} == probe,
                "Actual producer runtime/dependencies changed from checked preflight",
            )
            check_evidence(e, threshold, qa, baseline)
            if baseline is None:
                baseline = e
            m = e["metrics"]
            summaries.append(
                {
                    "threshold": threshold,
                    "report_sha256": sha(report),
                    "macro": m["macro_dataset_mean_dice"],
                    "pooled_dice": {c: v["dice"] for c, v in m["channels"].items()},
                    "source_macro": m["macro_dataset_channel_dice"],
                    "by_source": m["by_dataset"],
                    "conditional_AV": m["av_conditional_accuracy"],
                    "end_to_end_AV": m["av_accuracy"],
                    "runtime": m["evaluation_runtime"],
                }
            )
            event(
                "validation_pass",
                threshold=threshold,
                macro=m["macro_dataset_mean_dice"],
                report_sha256=sha(report),
            )
        save(
            a.output / "six-validation-summary.json",
            {
                "complete": True,
                "threshold_plan_sha256": THRESHOLD_PLAN,
                "checkpoint_sha256": qa["checkpoint_sha256"]["best.pt"],
                "all_source_data_GT_runtime_pins_identical": True,
                "conditional_counts_identical": True,
                "results": summaries,
                "interpretation": "Descriptive VALthresholdtradeoffs; no TEST/selectionartifact/autopromotion",
                "completed_utc": now().isoformat(),
            },
        )
        event(
            "all_six_validation_complete",
            summary_sha256=sha(a.output / "six-validation-summary.json"),
        )
    except Exception as error:
        event("stopped_no_retry", error_type=type(error).__name__, message=str(error))
        save(
            a.output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "no_retry": True,
                "no_other_process_signalled": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
