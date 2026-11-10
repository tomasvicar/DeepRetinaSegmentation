"""Explicit root-approved GUI-aware continuation; reuse verified60 snapshot only."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

CONTROL_SHA = "4a66b2eddbe974210089703ddf385411a6254bb4d41223f2ceab8094442fdfa7"
QA_SHA = "f4ebab64f77fc574a112bcd487950cfd8d7df4c3f607363010e3bfa722cfcf20"
BEST = "bfe957460dfed168f2344b1b84ef99b7f502e9dbc8ce1fd4d68632c3234708b1"
LAST = "60f6d66845a0f453d78b50679815a1123a874a2725fd3897b0eb62fa09d77796"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def proc_identity(pid):
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": int(pid),
        "uid": root.stat().st_uid,
        "exe": os.readlink(root / "exe"),
        "starttime_ticks": int(fields[19]),
    }


def xml_processes(blob, gpu):
    tree = ET.fromstring(blob)
    matches = [g for g in tree.findall("gpu") if g.findtext("uuid") == gpu]
    require(len(matches) == 1, "PhysicalGPU XML differs")
    return [
        {"pid": int(p.findtext("pid")), "type": p.findtext("type")}
        for p in matches[0].findall("processes/process_info")
    ]


def check_desktop_processes(rows, allowed, observed):
    approved = {r["pid"]: r for r in allowed}
    accepted = []
    for row in rows:
        if "C" not in row["type"]:
            continue
        pid = row["pid"]
        require(
            pid in approved and row["type"] == "C+G",
            "Unknown/new compute context; no launch/no kill",
        )
        require(
            observed[pid] == approved[pid],
            "Desktop identity UID/exe/starttime changed; no launch/no kill",
        )
        accepted.append(observed[pid])
    return accepted


REUSED_REPORTS = {
    0.5: "f2be07e77a8877339d63db519004098612a7c7249e048b612014a41dc16fa7a5",
    0.35: "2c223ffa1761f2fa2c91608faf95b83d5cea92f262f506f7d9353ba7654b47da",
}


def reuse_completed_report(folder, threshold):
    path = folder / f"threshold-{round(threshold * 100):03d}.json"
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == REUSED_REPORTS[threshold],
        "Completed producer report changed",
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["output", "control-helper", "qa-helper", "threshold-plan", "approved-desktop"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--approved-desktop-sha256", required=True)
    p.add_argument("--reuse-first-two", type=Path)
    a = p.parse_args()
    require(not a.output.exists(), "Fresh continuation directory required")
    a.output.mkdir(parents=True)
    require(sha(a.control_helper) == CONTROL_SHA, "Reviewed controller helper differs")
    h = module(a.control_helper, "reviewed_terminal_control")

    def event(phase, **fields):
        r = {"utc": h.now().isoformat(), "phase": phase, **fields}
        with (a.output / "events.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)

    try:
        require(
            sha(a.qa_helper) == h.QA_SHA and sha(a.threshold_plan) == h.THRESHOLD_PLAN,
            "Verifier/thresholdplan pins differ",
        )
        require(
            sha(a.approved_desktop) == a.approved_desktop_sha256,
            "Approved desktop identity pin differs",
        )
        allow = json.loads(a.approved_desktop.read_text())
        require(
            allow["gpu_uuid"] == h.GPU and allow["uid"] == os.getuid(), "Wrong GUI allowance scope"
        )
        home = Path.home()
        source = home / "code/DeepRetinaSegmentation_unified_run_3157bb9"
        evaluation = home / "code/DeepRetinaSegmentation_evaluation_8c56594"
        trials = home / "code/DeepRetinaSegmentation_unified_trials_3157bb9"
        canonical_root = home / "code/DeepRetinaSegmentation_for_Leipzig/data/unified25"
        manifest = canonical_root / "manifest.json"
        qa_path = trials / "lr0003-control-cached-e060-verified.json"
        python = home / "code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python"
        require(not h.alive(h.PID), "Training PID stillpresent")
        require(sha(qa_path) == QA_SHA, "Existing terminalQA proof changed")
        qa = json.loads(qa_path.read_text())
        h.check_qa(qa)
        require(
            qa["checkpoint_sha256"] == {"last.pt": LAST, "best.pt": BEST},
            "Exact approved snapshot differs",
        )
        milestone = Path(qa["milestone"])
        require(milestone.stat().st_mode & 0o777 == 0o555, "Milestone directory not0555")
        for f in milestone.iterdir():
            require(
                f.is_file() and not f.is_symlink() and f.stat().st_mode & 0o777 == 0o444,
                "Milestonefiles notreadonly0444",
            )
        require(
            sha(milestone / "best.pt") == BEST and sha(milestone / "last.pt") == LAST,
            "Readonlysnapshot bytes changed",
        )
        verifier = module(a.qa_helper, "reviewed_source_verifier")
        verifier.verify_source(source)  # Source-only check; NO checkpointQA rerun.
        require(
            subprocess.check_output(
                ["git", "-C", str(evaluation), "rev-parse", "HEAD"], text=True
            ).strip()
            == h.EVAL_COMMIT,
            "Frozen8ccommit differs",
        )
        require(
            not subprocess.check_output(
                ["git", "-C", str(evaluation), "status", "--porcelain", "--untracked-files=no"]
            ).strip(),
            "Frozen8ctrackedchanges",
        )
        require(
            h.package_hash(evaluation) == h.EVAL_PACKAGE
            and sha(evaluation / "scripts/unified_validation.py") == h.PRODUCER,
            "Frozen8csource/producer differs",
        )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            PYTHONPATH=str(evaluation / "src"),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    "import json;from deepretina.selection import evaluation_source_identity;print(json.dumps(evaluation_source_identity()))",
                ],
                env=env,
                text=True,
            )
        )
        require(probe["evaluation_source_sha256"] == h.EVAL_PACKAGE, "Actualimportedsource differs")
        locked = {
            p["name"].lower(): p["version"]
            for p in tomllib.loads((evaluation / "uv.lock").read_text())["package"]
        }
        for name, version in probe["runtime"]["installed_distributions"].items():
            if version is not None:
                require(
                    locked.get(name.lower()) == version,
                    "Actualdependency differs fromlock: " + name,
                )

        def readiness():
            require(h.now() < h.DEADLINE, "20:00UTC deadline; no newlaunch")
            gpu_csv = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            physical = h.gpu_free(gpu_csv, "")
            rows = xml_processes(subprocess.check_output(["nvidia-smi", "-q", "-x"]), h.GPU)
            observed = {r["pid"]: proc_identity(r["pid"]) for r in rows if "C" in r["type"]}
            accepted = check_desktop_processes(rows, allow["entries"], observed)
            return {
                **physical,
                "verified_established_desktop_CplusG": accepted,
                "no_unknown_compute": True,
                "background_scope": "Established UI contexts stayalive; not background-free andnot a latencybenchmark",
            }

        initial = readiness()
        addendum = {
            "kind": "preexecution_threshold_VAL_addendum_NOT_selection",
            "original_stopped_watcher": str(
                home / "code/DeepRetinaSegmentation_terminal60_watch_20261110/result"
            ),
            "original_qa_sha256": QA_SHA,
            "qa_not_rerun": True,
            "threshold_plan_sha256": h.THRESHOLD_PLAN,
            "checkpoint": {
                "path": str(milestone / "best.pt"),
                "sha256": BEST,
                "selected_epoch": qa["best_epoch"],
            },
            "terminal60": {
                "last_sha256": LAST,
                "steps": 122880,
                "samples": 491520,
                "scheduler": {
                    "T_max": 60,
                    "last_epoch": 60,
                    "proof": "Exact reviewed e88 terminalQA + immutablelast60bytes",
                },
            },
            "source3157": h.SOURCE,
            "evaluation_commit": h.EVAL_COMMIT,
            "evaluation_source": probe,
            "producer_sha256": h.PRODUCER,
            "canonical_fingerprint": h.CONTENT,
            "manifest_sha256": h.MANIFEST,
            "canonical_photo_geometry_sha256": sha(canonical_root / "photos.json"),
            "approved_desktop_sha256": a.approved_desktop_sha256,
            "actual_readiness": initial,
            "threshold_order": h.THRESHOLDS,
            "no_TEST": True,
            "no_selection_or_autopromotion": True,
            "completed_reports_reused_from": str(a.reuse_first_two) if a.reuse_first_two else None,
            "completed_report_pins": REUSED_REPORTS if a.reuse_first_two else {},
        }
        save(a.output / "threshold-addendum.json", addendum)
        event(
            "snapshot_and_addendum_verified",
            best_sha256=BEST,
            selected_epoch=qa["best_epoch"],
            gui_contexts=len(initial["verified_established_desktop_CplusG"]),
        )
        env["CUDA_VISIBLE_DEVICES"] = h.GPU
        baseline = None
        summaries = []
        for threshold in h.THRESHOLDS:
            observed = readiness()
            require(
                sha(milestone / "best.pt") == BEST
                and sha(milestone / "last.pt") == LAST
                and sha(qa_path) == QA_SHA,
                "Snapshot/QA changed beforecall",
            )
            require(
                h.package_hash(evaluation) == h.EVAL_PACKAGE
                and sha(evaluation / "scripts/unified_validation.py") == h.PRODUCER,
                "Evaluationsource changed",
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
            reused = a.reuse_first_two is not None and threshold in REUSED_REPORTS
            if reused:
                payload = reuse_completed_report(a.reuse_first_two, threshold)
                with report.open("xb") as f:
                    f.write(payload)
                event(
                    "validation_reused_no_new_forward",
                    threshold=threshold,
                    report_sha256=sha(report),
                    original_path=str(a.reuse_first_two / report.name),
                )
            else:
                save(
                    a.output / (name + "-command.json"),
                    {
                        "command": command,
                        "environment": {
                            k: env[k]
                            for k in [
                                "CUDA_VISIBLE_DEVICES",
                                "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "PYTHONPATH",
                                "CUBLAS_WORKSPACE_CONFIG",
                            ]
                        },
                        "readiness": observed,
                    },
                )
                event("validation_start", threshold=threshold)
                h.run_child(command, env, a.output / (name + ".log"))
            e = json.loads(report.read_text())
            require(
                {k: v for k, v in e["source"].items() if k != "training_source_sha256"} == probe,
                "Actualsource/depschanged",
            )
            h.check_evidence(e, threshold, qa, baseline)
            if baseline is None:
                baseline = e
            m = e["metrics"]
            summaries.append(
                {
                    "threshold": threshold,
                    "producer_phase": "reused_completed_producer" if reused else "new_producer",
                    "report_sha256": sha(report),
                    "macro": m["macro_dataset_mean_dice"],
                    "pooled_dice": {c: v["dice"] for c, v in m["channels"].items()},
                    "source_macro": m["macro_dataset_channel_dice"],
                    "by_source": m["by_dataset"],
                    "conditional_AV": m["av_conditional_accuracy"],
                    "end_to_end_AV": m["av_accuracy"],
                    "actual_runtime": m["evaluation_runtime"],
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
                "results": summaries,
                "checkpoint_sha256": BEST,
                "threshold_plan_sha256": h.THRESHOLD_PLAN,
                "all6source_data_GT_runtime_and_ungated_conditional_counts_identical": True,
                "desktop_background_scope": initial,
                "no_TEST_selection_or_autopromotion": True,
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
                "error_type": type(error).__name__,
                "message": str(error),
                "no_retry": True,
                "no_other_process_signals": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
