"""Reviewed fresh-native vs clipped cup continuation; historical drift stays explicit."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import compose_original_disc_reference as composer
import numpy as np

PLAN_SHA = "3f82e625b62f089bc7a3a36a9cfb3072e8367821147fba48e1f9831826d5f4ee"
SOURCE = "b4134160983b7358be1ffd06a97ff497076ddb5e"
PACKAGE = "4e24ddf235bd108bebe35437320ad168d332df3274fbd2a8c98567ef07181bfa"
COMPOSER = "28654e8c43973167031279f60a247cc6e298f1541e47828947d2657a0f12acdc"
ORIGINAL_SCRIPT = "f8425bc81b34225137e72ac53353a2099363759ec075e58507900823e277ebf3"
COUNT_KEYS = ("tp", "fp", "fn", "valid_pixels")
MODELS = ("universal", "disc0", "disc1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_counts(value):
    result = {key: value[key] for key in COUNT_KEYS}
    require(all(type(v) is int for v in result.values()), "Counts must be exact integers")
    return composer.metrics(result)


def support_equal(actual, expected):
    fixed_counts(actual)
    fixed_counts(expected)
    require(
        actual["valid_pixels"] == expected["valid_pixels"]
        and actual["tp"] + actual["fn"] == expected["tp"] + expected["fn"],
        "GT validity/positive support differs",
    )


def identical_counts(actual, expected):
    return all(actual[key] == expected[key] for key in COUNT_KEYS)


def historical_delta(actual, old):
    support_equal(actual, old)
    return {
        "exact_counts": identical_counts(actual, old),
        "fresh_minus_historical": {key: actual[key] - old[key] for key in COUNT_KEYS},
    }


def paired_counts(cup_logits, disc0_logits, disc1_logits, target, valid):
    arrays = [np.asarray(value) for value in (cup_logits, disc0_logits, disc1_logits)]
    require(
        all(a.shape == target.shape and np.isfinite(a).all() for a in arrays),
        "Bad native logit geometry or nonfinite value",
    )
    native, disc0, disc1 = [a > 0 for a in arrays]

    def count(pred):
        return composer.metrics(
            {
                "tp": int((pred & target & valid).sum()),
                "fp": int((pred & ~target & valid).sum()),
                "fn": int((~pred & target & valid).sum()),
                "valid_pixels": int(valid.sum()),
            }
        )

    return {"native": count(native), "disc0": count(native & disc0), "disc1": count(native & disc1)}


def checked_row(row, lookups, *, require_historical_exact=False):
    pid = row["photo_id"]
    old = lookups["disc0"][pid]
    require(row["source"] == old["channel_datasets"]["cup"], "Annotation source differs")
    require(row["record_ids"] == old["record_ids"], "Annotation inventory/order differs")
    native = row["cup"]["native"]
    support_equal(native, old["channels"]["cup"])
    for name in ("disc0", "disc1"):
        support_equal(row["cup"][name], native)
        require(
            row["cup"][name]["tp"] <= native["tp"]
            and row["cup"][name]["fp"] <= native["fp"]
            and row["cup"][name]["fn"] >= native["fn"],
            "Clipping is not a subset of SAME fresh native cup",
        )
        support_equal(row["native_disc"][name], lookups[name][pid]["channels"]["disc"])
    drifts = {"cup": historical_delta(native, old["channels"]["cup"])}
    drifts.update(
        {
            name: historical_delta(row["native_disc"][name], lookups[name][pid]["channels"]["disc"])
            for name in ("disc0", "disc1")
        }
    )
    if require_historical_exact:
        require(
            all(v["exact_counts"] for v in drifts.values()),
            "Reused prefix failed its original exact-count proof",
        )
    result = copy.deepcopy(row)
    result.pop("native_counts_exact", None)
    result["historical_native_count_comparison"] = drifts
    result["native_cup_and_clips_from_same_forward"] = True
    return result


def verified_prefix(payload, *, expected_sha, eligible, lookups, count=157):
    require(sha(payload) == expected_sha, "Prefix bytes changed")
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    require(
        len(rows) == count and [r["photo_id"] for r in rows] == eligible[:count],
        "Skipped/duplicate/reordered prefix photo",
    )
    result = []
    for row in rows:
        require(row.get("native_counts_exact") is True, "Missing original prefix proof")
        checked = checked_row(row, lookups, require_historical_exact=True)
        checked["phase"] = "reused_original_b413_prefix"
        checked["original_prefix_native_counts_exact"] = True
        result.append(checked)
    return result


def check_sample(
    row, record_ids, trace, image_shape, target_cup, valid_cup, target_disc, valid_disc
):
    require(record_ids == row["record_ids"], "Canonical record grouping differs")
    require(json.dumps(trace) == json.dumps(row["selected_trace"]), "First-observer trace differs")
    require(list(image_shape) == row["shape"], "Prepared image geometry differs")
    for counts, target, valid in [
        (row["cup"]["native"], target_cup, valid_cup),
        *((row["native_disc"][name], target_disc, valid_disc) for name in ("disc0", "disc1")),
    ]:
        require(
            counts["valid_pixels"] == int(valid.sum())
            and counts["tp"] + counts["fn"] == int((target & valid).sum()),
            "Decoded GT validity/positive pin differs",
        )


def compose_control(base, rows, variant, context):
    require(
        base["split"] == "val" and not base.get("preview_only", False),
        "Full VAL reference required",
    )
    supported = {r["photo_id"] for r in base["per_photo"] if r["channels"]["cup"]["valid_pixels"]}
    replacements = {r["photo_id"]: r["cup"][variant] for r in rows}
    require(
        len(replacements) == len(rows) and set(replacements) == supported,
        "Incomplete or duplicated cup-supported inventory",
    )
    result = copy.deepcopy(base)
    for row in result["per_photo"]:
        if row["photo_id"] in replacements:
            support_equal(replacements[row["photo_id"]], row["channels"]["cup"])
            row["channels"]["cup"] = fixed_counts(replacements[row["photo_id"]])
    for before, after in zip(base["per_photo"], result["per_photo"], strict=True):
        for channel in ("vessel", "artery", "vein", "disc"):
            require(
                before["channels"][channel] == after["channels"][channel],
                "Non-cup component changed",
            )
        for key in ("photo_id", "record_ids", "dataset", "channel_datasets"):
            require(before[key] == after[key], "Composite annotation/photo changed")
    pooled, grouped, macro, score = composer.aggregate(result["per_photo"], base["by_dataset"])
    result.update(
        channels=pooled,
        by_dataset=grouped,
        macro_dataset_channel_dice=macro,
        macro_dataset_mean_dice=score,
    )
    result["family"] = "optional_fresh_paired_cup_experiment"
    result["historical_component_policy"] = base.get("policy")
    result["policy"] = (
        "Untouched historical non-cup components; separately labelled fresh-native or same-forward clipped cup; historical native numeric drift logged separately"
    )
    result["fresh_paired_cup_component"] = {
        "cup_variant": variant,
        "reference_context": context,
        "noncup_components": "Exact historical vascular/disc counts and AVaccuracy; fresh standalone disc only used for clipping and drift diagnostics",
        "policy": "Fresh native cup strictlogit>0 vs cup intersect SAMEforward fresh dedicated disc strictlogit>0; no GTused for clipping",
        "historical_cup_numeric_drift": "Logged separately; baseline for postprocessing is fresh native cup, NOT historical cup",
        "reused_prefix_photos": sum(r["phase"] == "reused_original_b413_prefix" for r in rows),
        "new_photos": sum(r["phase"] == "new_continuation_forward" for r in rows),
        "no_test_evaluation": True,
        "no_independent_quality_claim": True,
    }
    return result


def summarize_rows(rows):
    groups = {"pooled": rows}
    for source in sorted({r["source"] for r in rows}):
        groups[source] = [r for r in rows if r["source"] == source]
    summary = {}
    for source, items in groups.items():
        totals = {
            variant: composer.metrics(
                {key: sum(r["cup"][variant][key] for r in items) for key in COUNT_KEYS}
            )
            for variant in ("native", "disc0", "disc1")
        }
        summary[source] = {
            "photos": len(items),
            "fresh_native": totals["native"],
            "clipped": {
                name: {
                    "counts_and_metrics": totals[name],
                    "TP_removed_vs_fresh": totals["native"]["tp"] - totals[name]["tp"],
                    "FP_removed_vs_fresh": totals["native"]["fp"] - totals[name]["fp"],
                    "FN_added_vs_fresh": totals[name]["fn"] - totals["native"]["fn"],
                }
                for name in ("disc0", "disc1")
            },
            "historical_drift": {
                name: {
                    "photos_with_drift": sum(
                        not r["historical_native_count_comparison"][name]["exact_counts"]
                        for r in items
                    ),
                    "fresh_minus_historical": {
                        key: sum(
                            r["historical_native_count_comparison"][name]["fresh_minus_historical"][
                                key
                            ]
                            for r in items
                        )
                        for key in COUNT_KEYS
                    },
                }
                for name in ("cup", "disc0", "disc1")
            },
        }
    return summary


def ensure_new_output(output, protected):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Output exists; no overwrite")
    require(
        not any(output.resolve().is_relative_to(Path(p).resolve()) for p in protected),
        "Output may not modify protected data/source/previous-run trees",
    )


def package_fingerprint(source):
    result = hashlib.sha256()
    root = source / "src/deepretina"
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        result.update(
            ("src/deepretina/" + path.relative_to(root).as_posix()).encode()
            + b"\0"
            + path.read_bytes()
        )
    for name in ("pyproject.toml", "uv.lock"):
        result.update(name.encode() + b"\0" + (source / name).read_bytes())
    return result.hexdigest()


def verify_frozen_source(source):
    require(
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == SOURCE,
        "Frozen b413 source commit differs",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]
        ).strip(),
        "Frozen source tracked changes",
    )
    require(package_fingerprint(source) == PACKAGE, "Frozen data/forward package bytes differ")


def budget(started):
    require(time.monotonic() - started < 4800, "80-minute continuation wall budget reached")
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2097152, "2GiB RSS budget reached")


def evaluate(args):
    # All hard pins precede any model loading or forward. CPU runtime must already be bounded.
    require(
        sha(Path(__file__).read_bytes()) == args.expected_helper_sha256,
        "Reviewed helper byte pin differs",
    )
    plan_bytes = args.plan.read_bytes()
    require(sha(plan_bytes) == PLAN_SHA, "Original reviewed proposal differs")
    plan = json.loads(plan_bytes)
    verify_frozen_source(args.source)
    require(
        sha(Path(composer.__file__).read_bytes()) == COMPOSER,
        "Count-composition helper bytes differ",
    )
    require(sha(args.manifest.read_bytes()) == plan["manifest_sha256"], "Manifest pin differs")
    prefix_bytes = args.prefix.read_bytes()
    terminal_bytes = args.prefix_terminal.read_bytes()
    require(
        sha(terminal_bytes) == plan["prefix_reuse"]["terminal_proof_sha256"],
        "Aborted-prefix terminal proof changed",
    )
    old_terminal = json.loads(terminal_bytes)
    require(
        old_terminal["source_commit"] == SOURCE
        and old_terminal["script_sha256"] == ORIGINAL_SCRIPT
        and old_terminal["returncode"] == 1
        and old_terminal["full_report_exists"] is False,
        "Wrong original aborted-run proof",
    )
    require(
        Path(sys.executable).resolve() == Path(old_terminal["command"][3]).resolve(),
        "Original frozen Python executable differs",
    )
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        require(
            os.environ.get(key) == value, "Required pre-import CPU/CUDA environment differs: " + key
        )
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Run with nice10 or lower priority")
    ensure_new_output(
        args.output,
        [args.manifest.parent, args.source, args.prefix_terminal.parent, Path.home() / "ownCloud"],
    )
    args.output.mkdir(parents=True)
    args._created_output = True
    started = time.monotonic()
    import cv2
    import torch

    from deepretina import unified
    from deepretina.integrity import verify_prepared_content
    from deepretina.legacy.checkpoints import load_single_net
    from deepretina.legacy.single_net import tiled_logits

    require(
        Path(unified.__file__).resolve() == (args.source / "src/deepretina/unified.py").resolve(),
        "Wrong imported data/model package",
    )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    require(str(torch.__version__) == "2.6.0+cu124", "Frozen Torch runtime differs")
    require(
        verify_prepared_content(args.manifest, plan["canonical_fingerprint"])
        == plan["canonical_fingerprint"],
        "Canonical content differs",
    )
    references = {}
    for name in ("disc0", "disc1"):
        path = args.reference_dir / f"original-calibrated-av-{name}-raw-val.json"
        reference_bytes = path.read_bytes()
        require(
            sha(reference_bytes) == plan["original_reference_sha256"][name],
            "Historical reference changed",
        )
        references[name] = json.loads(reference_bytes)
    lookups = {
        name: {r["photo_id"]: r for r in ref["per_photo"]} for name, ref in references.items()
    }
    require(
        set(lookups["disc0"]) == set(lookups["disc1"]) and len(lookups["disc0"]) == 770,
        "Historical photo inventory differs",
    )
    for pid, ref in lookups["disc0"].items():
        require(
            identical_counts(ref["channels"]["cup"], lookups["disc1"][pid]["channels"]["cup"]),
            "Historical cup components disagree",
        )
    dataset = unified.UnifiedDataset(args.manifest, "val", unified.UnifiedConfig())
    require(
        set(dataset.photo_ids) == set(lookups["disc0"]), "Canonical VAL photo inventory differs"
    )
    eligible = [
        pid for pid in dataset.photo_ids if lookups["disc0"][pid]["channels"]["cup"]["valid_pixels"]
    ]
    require(len(eligible) == 612, "Supported cup inventory differs")
    rows = verified_prefix(
        prefix_bytes,
        expected_sha=plan["prefix_reuse"]["prefix_sha256"],
        eligible=eligible,
        lookups=lookups,
    )
    index_by_id = {pid: i for i, pid in enumerate(dataset.photo_ids)}

    def sample(pid):
        index = index_by_id[pid]
        image, target, valid, trace = dataset.sample(
            index, random_annotator=False, return_trace=True
        )
        chosen_ids = {item[0] for item in trace if item[1] == "disc_cup"}
        chosen = [r for r in dataset.groups[index] if r["id"] in chosen_ids]
        require(
            len(chosen) == 1
            and chosen[0]["dataset"] == lookups["disc0"][pid]["channel_datasets"]["cup"],
            "Decoded cup annotation source differs",
        )
        return (
            image,
            target[4].numpy() >= 0.5,
            valid[4].numpy().copy(),
            target[3].numpy() >= 0.5,
            valid[3].numpy().copy(),
            trace,
            [r["id"] for r in dataset.groups[index]],
        )

    # Re-decode GT/input metadata for prefix proof, with ZERO prefix model forwards.
    for row in rows:
        image, tc, vc, td, vd, trace, ids = sample(row["photo_id"])
        check_sample(row, ids, trace, image.shape[-2:], tc, vc, td, vd)
        row["model_input_sha256_observed_at_continuation"] = sha((image - 0.5).numpy().tobytes())
        row["original_input_pin"] = (
            "Exact canonical image bytes and frozen b413 data package; original per-photo input hash was not saved"
        )
    models, payloads = {}, {}
    for name in MODELS:
        path = (
            args.source
            / "src/deepretina/pretrained"
            / (
                "legacy-single-net-"
                + ("universal" if name == "universal" else "disc-" + name[-1])
                + ".pt"
            )
        )
        model_bytes = path.read_bytes()
        require(
            sha(model_bytes) == plan["models"][name]["checkpoint_sha256"],
            "Original model byte pin differs",
        )
        model, payload = load_single_net(io.BytesIO(model_bytes), "cpu")
        require(
            payload["channels"] == (["vessel", "disc", "cup"] if name == "universal" else ["disc"]),
            "Model channel semantics differ",
        )
        require(
            payload["tiling"] == plan["models"][name]["tiling"]
            and payload["architecture"] == plan["models"][name]["architecture"],
            "Model architecture/tiling differs",
        )
        require(
            sum(p.numel() for p in model.parameters()) == plan["models"][name]["parameters"],
            "Model parameter count differs",
        )
        require(
            all(torch.isfinite(v).all() for v in model.state_dict().values()),
            "Nonfinite original weights",
        )
        models[name], payloads[name] = model, payload
    launch = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "helper_sha256": args.expected_helper_sha256,
        "proposal_sha256": PLAN_SHA,
        "source_commit": SOURCE,
        "source_package_sha256": PACKAGE,
        "prefix_sha256": sha(prefix_bytes),
        "prefix_proof_sha256": sha(terminal_bytes),
        "canonical_fingerprint": plan["canonical_fingerprint"],
        "manifest_sha256": plan["manifest_sha256"],
        "models": plan["models"],
        "threads": 2,
        "interop_threads": 1,
        "niceness": os.getpriority(os.PRIO_PROCESS, 0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "runtime_component_split": {
            "reused_prefix_photos": 157,
            "new_photos_planned": 455,
            "prior_aborted_full_seconds": old_terminal["elapsed_seconds"],
            "prior_aborted_run_status": "157 successful rows then failed historical exact gate; includes inference attempted on failed158",
            "prior_prefix_inference_seconds_by_model": None,
            "prior_prefix_model_timing_status": "Not separately flushed by original failed run; only total aborted wall time is available",
            "continuation_elapsed_is_NOT_total_study_time": True,
        },
    }
    dump(args.output / "launch.json", launch)
    seconds = {name: 0.0 for name in MODELS}
    with (args.output / "per-photo.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        for pid in eligible[157:]:
            budget(started)
            image, tc, vc, td, vd, trace, ids = sample(pid)
            logits = {}
            with torch.inference_mode():
                for name in MODELS:
                    before = time.monotonic()
                    p = payloads[name]
                    value = tiled_logits(
                        models[name],
                        image - 0.5,
                        "cpu",
                        p["tiling"]["crop_size"],
                        p["tiling"]["border"],
                    )
                    logits[name] = value[2 if name == "universal" else 0].numpy().copy()
                    seconds[name] += time.monotonic() - before
                    budget(started)
            cup = paired_counts(logits["universal"], logits["disc0"], logits["disc1"], tc, vc)
            # Dedicated native-disc counts use identical predicted arrays and true disc GT support.
            discs = {
                name: paired_counts(logits[name], logits[name], logits[name], td, vd)["native"]
                for name in ("disc0", "disc1")
            }
            row = {
                "photo_id": pid,
                "source": lookups["disc0"][pid]["channel_datasets"]["cup"],
                "record_ids": ids,
                "selected_trace": trace,
                "shape": list(image.shape[-2:]),
                "cup": cup,
                "native_disc": discs,
                "phase": "new_continuation_forward",
                "model_input_sha256": sha((image - 0.5).numpy().tobytes()),
            }
            row = checked_row(row, lookups)
            check_sample(row, ids, trace, image.shape[-2:], tc, vc, td, vd)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if (len(rows) - 157) % 10 == 0 or len(rows) == 612:
                print(
                    json.dumps(
                        {
                            "processed_total": len(rows),
                            "reused": 157,
                            "new": len(rows) - 157,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    require(
        sha(args.prefix.read_bytes()) == sha(prefix_bytes)
        and sha(args.prefix_terminal.read_bytes()) == sha(terminal_bytes),
        "Historical prefix/proof changed during continuation",
    )
    verify_frozen_source(args.source)
    require(
        verify_prepared_content(args.manifest, plan["canonical_fingerprint"])
        == plan["canonical_fingerprint"],
        "Canonical content changed during continuation",
    )
    budget(started)
    composites = {}
    for name in ("disc0", "disc1"):
        for variant in ("native", name):
            label = (
                f"fresh-native-cup-in-{name}-reference-val"
                if variant == "native"
                else f"fresh-cup-clipped-by-{name}-in-{name}-reference-val"
            )
            composed = compose_control(references[name], rows, variant, name)
            composed["fresh_paired_component_runtime"] = {
                "reused_prefix_source": SOURCE,
                "prefix_sha256": sha(prefix_bytes),
                "prefix_photos": 157,
                "new_forward_photos": 455,
                "prior_aborted_full_seconds": old_terminal["elapsed_seconds"],
                "continuation_seconds_to_composition": time.monotonic() - started,
                "new_model_inference_seconds": dict(seconds),
                "helper_sha256": args.expected_helper_sha256,
                "source_package_sha256": PACKAGE,
                "same_forward_native_vs_clip": True,
            }
            dump(args.output / (label + ".json"), composed)
            composites[label] = {
                "sha256": sha((args.output / (label + ".json")).read_bytes()),
                "five_head_source_macro": composed["macro_dataset_mean_dice"],
                "source_macro": composed["macro_dataset_channel_dice"],
                "pooled": composed["channels"],
            }
    elapsed = time.monotonic() - started
    report = {
        "kind": "complete_fresh_paired_cup_validation_study",
        "split": "val",
        "complete": True,
        "photos": 770,
        "cup_supported_photos": 612,
        "reused_prefix_photos": 157,
        "new_forward_photos": 455,
        "skipped_unsupported_cup_photos": 158,
        "native_vs_historical_counts_must_not_be_mislabelled_exact": True,
        "input_and_GT_pins_exact": True,
        "launch": launch,
        "cup_and_drift_by_source": summarize_rows(rows),
        "composites": composites,
        "continuation_seconds": elapsed,
        "prior_aborted_full_seconds": old_terminal["elapsed_seconds"],
        "total_aborted_full_plus_continuation_seconds": old_terminal["elapsed_seconds"] + elapsed,
        "total_timing_scope": "Includes aborted full run and this continuation, excludes earlier five-photo canary/forensic probes; no claim of fromscratch612-only timing",
        "new_model_inference_seconds": seconds,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_photo_sha256": sha((args.output / "per-photo.jsonl").read_bytes()),
        "historical_refs_unchanged": True,
        "original_defaults_changed": False,
        "no_test": True,
        "no_gpu": True,
        "no_independent_quality_claim": True,
    }
    dump(args.output / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("composites", "cup_and_drift_by_source")}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "source",
        "plan",
        "manifest",
        "reference-dir",
        "prefix",
        "prefix-terminal",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    args = parser.parse_args()
    args._created_output = False
    try:
        evaluate(args)
    except Exception as error:
        # Only an already-created fresh output directory can receive failure evidence.
        if args._created_output and not (args.output / "failure.json").exists():
            dump(
                args.output / "failure.json",
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "partial_results_preserved": True,
                },
            )
        raise


if __name__ == "__main__":
    main()
