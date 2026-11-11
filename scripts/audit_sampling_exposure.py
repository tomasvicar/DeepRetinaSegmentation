"""TRAIN-only expected sampling exposure; no images, model, Torch or heldout masks.

Read the verified bool-label cache once for TRAIN records shared by V1/V2.
This computes pre-crop expectations, never actual delivered optimizer samples.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TASK_CHANNELS = {"vessels": ("vessel",), "av": ("vessel", "av", "av"), "disc_cup": ("disc", "cup")}
TASKS = ("vessel", "av", "disc", "cup")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def train_rows(rows):
    return [
        r
        for r in rows
        if r["split"] == "train"
        and r["task"] in TASK_CHANNELS
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    ]


def task_flags(row, target, valid, fov):
    valid = valid.astype(bool) & fov.astype(bool)[None]
    if target.shape != valid.shape or target.shape[0] != len(TASK_CHANNELS[row["task"]]):
        raise ValueError("Wrong label geometry: " + row["id"])
    tasks = {
        task for i, task in enumerate(TASK_CHANNELS[row["task"]]) if task != "av" and valid[i].any()
    }
    positive = {
        task
        for i, task in enumerate(TASK_CHANNELS[row["task"]])
        if task != "av" and ((target[i] >= 0.5) & valid[i]).any()
    }
    if row["task"] == "av":
        known = valid[1] & valid[2] & ((target[1] >= 0.5) ^ (target[2] >= 0.5))
        if known.any():
            tasks.add("av")
            positive.add("av")
    return tasks, positive


def exposure(
    rows, flags, positives, budget, frozen_method=None, *, vascular_priority=1.0, av_probability=0.5
):
    """Analytic research scenarios only; never used by the production sampler."""
    if vascular_priority not in {1, 1.5, 2} or av_probability not in {0.5, 0.75}:
        raise ValueError("Only the six predeclared research scenarios are allowed")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["photo_id"]].append(row)
    groups = [sorted(g, key=lambda r: r["id"]) for _, g in sorted(grouped.items())]
    if budget < len(groups):
        raise ValueError("Coverage-first budget is too small")
    memberships, selected, selected_positive, chosen_sources = [], [], [], []
    for group in groups:
        declared = {str(r["sampling_group"]) for r in group if "sampling_group" in r}
        if len(declared) > 1:
            raise ValueError("Conflicting sampling_group")
        source = next(iter(declared)) if declared else group[0]["dataset"]
        tasks = set().union(*(flags[r["id"]] for r in group))
        memberships.append({(source, task) for task in tasks})
        by_task = defaultdict(list)
        for row in group:
            by_task[row["task"]].append(row)
        vascular = [task for task in ("av", "vessels") if by_task[task]]
        probability, positive_probability, source_probability = Counter(), Counter(), Counter()
        for task in [*vascular, "disc_cup"]:
            options = by_task[task]
            if not options:
                continue
            task_probability = 1.0
            if task != "disc_cup" and len(vascular) == 2:
                task_probability = av_probability if task == "av" else 1 - av_probability
            for row in options:
                chance = task_probability / len(options)
                for name in flags[row["id"]]:
                    probability[name] += chance
                for name in positives[row["id"]]:
                    positive_probability[name] += chance
                source_probability[row["dataset"]] += chance
        selected.append(probability)
        selected_positive.append(positive_probability)
        chosen_sources.append(source_probability)
    populations = Counter(key for keys in memberships for key in keys)
    weights = [sum(1 / populations[key] for key in keys) for keys in memberships]
    if frozen_method:
        config = SimpleNamespace(sampling="dataset_task_balanced")
        fake = SimpleNamespace(
            groups=groups, rows=rows, _row_tasks=flags, config=config, support_summary=lambda: None
        )
        original = frozen_method(fake)
        if not all(
            math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14) for a, b in zip(weights, original)
        ) or len(original) != len(weights):
            raise ValueError("Weights differ from frozen sampling_weights")
    weights = [
        sum(
            (vascular_priority if key[1] in {"vessel", "av"} else 1.0) / populations[key]
            for key in keys
        )
        for keys in memberships
    ]
    total_weight = sum(weights)
    extras = budget - len(groups)
    expected = [1 + extras * w / total_weight for w in weights]
    task_report = {}
    for task in TASKS:
        opportunity = [float(any(key[1] == task for key in keys)) for keys in memberships]
        selected_p = [values[task] for values in selected]
        task_report[task] = {
            "available_photos": int(sum(opportunity)),
            "expected_photo_opportunities": sum(n * p for n, p in zip(expected, opportunity)),
            "coverage_pass_expected_selected_supervision": sum(selected_p),
            "weighted_extras_expected_selected_supervision": sum(
                (n - 1) * p for n, p in zip(expected, selected_p)
            ),
            "expected_selected_supervision_total": sum(n * p for n, p in zip(expected, selected_p)),
            "expected_selected_supervision_fraction": sum(
                n * p for n, p in zip(expected, selected_p)
            )
            / budget,
            "expected_selected_positive_class_presence": sum(
                n * values[task] for n, values in zip(expected, selected_positive)
            ),
        }
    owners = sorted({key[0] for keys in memberships for key in keys})
    source_report = []
    for source in owners:
        indices = [i for i, keys in enumerate(memberships) if any(key[0] == source for key in keys)]
        source_report.append(
            {
                "sampling_owner": source,
                "photos": len(indices),
                "weighted_extras_probability": sum(weights[i] for i in indices) / total_weight,
                "expected_photo_draws": sum(expected[i] for i in indices),
                "source_task_populations": {
                    task: populations[source, task]
                    for task in TASKS
                    if (source, task) in populations
                },
            }
        )
    annotation_sources = sorted({r["dataset"] for r in rows})
    return {
        "photos": len(groups),
        "records": len(rows),
        "epoch_samples": budget,
        "coverage_first_draws": len(groups),
        "weighted_extra_draws": extras,
        "membership_keys": len(populations),
        "sum_weights": total_weight,
        "base_weights_match_frozen_method": frozen_method is not None,
        "weights_match_frozen_method": frozen_method is not None and vascular_priority == 1,
        "observer_policy_matches_frozen": av_probability == 0.5,
        "vascular_priority": vascular_priority,
        "av_probability_when_both_bundles_exist": av_probability,
        "tasks": task_report,
        "sampling_sources": source_report,
        "expected_selected_annotation_bundles_by_source": {
            source: sum(n * p[source] for n, p in zip(expected, chosen_sources))
            for source in annotation_sources
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["manifest", "parent-manifest", "cache-root", "frozen-unified", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    for key in [
        "expected-manifest-sha256",
        "expected-cache-index-sha256",
        "expected-frozen-unified-sha256",
    ]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--epoch-samples", type=int, default=8192)
    p.add_argument(
        "--scenario-grid",
        action="store_true",
        help="Six predeclared analytic scenarios; no runtime sampling changes",
    )
    p.add_argument(
        "--task-flags-output",
        type=Path,
        help="Save reusable TRAIN-only task metadata; no label arrays",
    )
    p.add_argument("--reference-report", type=Path)
    p.add_argument("--expected-reference-sha256")
    a = p.parse_args()
    if a.scenario_grid and (not a.reference_report or not a.expected_reference_sha256):
        p.error("--scenario-grid requires a byte-pinned previous reference report")
    start = time.monotonic()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.task_flags_output and (
        a.task_flags_output.exists() or a.task_flags_output.resolve() == a.output.resolve()
    ):
        raise FileExistsError(a.task_flags_output)
    raw = a.manifest.read_bytes()
    index_raw = (a.cache_root / "index.json").read_bytes()
    code = a.frozen_unified.read_bytes()
    for actual, wanted in [
        (sha(raw), a.expected_manifest_sha256),
        (sha(index_raw), a.expected_cache_index_sha256),
        (sha(code), a.expected_frozen_unified_sha256),
    ]:
        if actual != wanted:
            raise ValueError("Input fingerprint mismatch")
    rows = json.loads(raw)
    old_raw = a.parent_manifest.read_bytes()
    old_rows = json.loads(old_raw)
    lookup = {r["id"]: r for r in rows}
    if any(lookup.get(r["id"]) != r for r in old_rows):
        raise ValueError("Parent rows are not retained exactly")
    index = json.loads(index_raw)
    if (
        index["canonical_manifest_sha256"] != sha(raw)
        or not index["construction"]["exact_equality_verified"]
    ):
        raise ValueError("Cache construction/manifest mismatch")
    tree = ast.parse(code)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedDataset")
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sampling_weights"
    )
    namespace = {"Counter": Counter}
    # Execute only the single trusted method after verifying the caller-supplied source SHA.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[method], type_ignores=[]), "<pinned frozen sampling_weights>", "exec"
        ),
        namespace,
    )
    flags = {}
    positives = {}
    read_cache = {}
    total_bytes = 0
    for number, row in enumerate(train_rows(rows), 1):
        mapping = index["source_to_derived"][row["mask"]]
        relative = mapping["derived"]
        path = (a.cache_root / relative).resolve()
        if not path.is_relative_to(a.cache_root.resolve()):
            raise ValueError("Unsafe cache path")
        key = (relative, row["task"])
        if key not in read_cache:
            contents = path.read_bytes()
            entry = index["files"][relative]
            if len(contents) != entry["bytes"] or sha(contents) != entry["sha256"]:
                raise ValueError("Cache file byte mismatch")
            total_bytes += len(contents)
            with np.load(io.BytesIO(contents), allow_pickle=False) as z:
                target = z["target"]
                valid = z["valid"]
                fov = z["fov"] if "fov" in z else np.ones(target.shape[-2:], bool)
                read_cache[key] = task_flags(row, target, valid, fov)
        flags[row["id"]], positives[row["id"]] = read_cache[key]
        if number % 500 == 0:
            print(
                json.dumps(
                    {"train_records_checked": number, "elapsed_seconds": time.monotonic() - start}
                ),
                flush=True,
            )
    reports = {
        name: exposure(
            train_rows(rs), flags, positives, a.epoch_samples, namespace["sampling_weights"]
        )
        for name, rs in [("v1", old_rows), ("v2", rows)]
    }
    result = {
        "schema_version": 1,
        "scope": "Expected TRAIN-only pre-crop exposure, not actual delivered counts",
        "manifest_sha256": sha(raw),
        "parent_manifest_sha256": sha(old_raw),
        "cache_index_sha256": sha(index_raw),
        "canonical_fingerprint_from_index": index["canonical_fingerprint"],
        "frozen_unified_sha256": sha(code),
        "train_label_files_decoded": len(read_cache),
        "compressed_train_label_bytes_read": total_bytes,
        "heldout_masks_decoded": 0,
        "rgb_images_decoded": 0,
        "reports": reports,
        "task_deltas_v2_minus_v1": {
            task: {
                key: reports["v2"]["tasks"][task][key] - reports["v1"]["tasks"][task][key]
                for key in [
                    "expected_selected_supervision_total",
                    "expected_selected_supervision_fraction",
                ]
            }
            for task in TASKS
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    if a.scenario_grid:
        reference_raw = a.reference_report.read_bytes()
        if sha(reference_raw) != a.expected_reference_sha256:
            raise ValueError("Previous report fingerprint mismatch")
        reference = json.loads(reference_raw)
        for pin in [
            "manifest_sha256",
            "parent_manifest_sha256",
            "cache_index_sha256",
            "frozen_unified_sha256",
        ]:
            if result[pin] != reference[pin]:
                raise ValueError("Previous report input identity differs")
        for name in ("v1", "v2"):
            for key in (
                "tasks",
                "sampling_sources",
                "expected_selected_annotation_bundles_by_source",
            ):
                if result["reports"][name][key] != reference["reports"][name][key]:
                    raise ValueError(
                        "Baseline expectations do not reproduce previous reference exactly"
                    )
        result["reference_report_sha256"] = sha(reference_raw)
        result["baseline_expectations_reproduced_exactly"] = True
        result["scenarios"] = [
            exposure(
                train_rows(rows),
                flags,
                positives,
                a.epoch_samples,
                namespace["sampling_weights"],
                vascular_priority=priority,
                av_probability=probability,
            )
            for priority in (1.0, 1.5, 2.0)
            for probability in (0.5, 0.75)
        ]
        result["elapsed_seconds"] = time.monotonic() - start
    if a.task_flags_output:
        task_metadata = {
            "schema_version": 1,
            "scope": "TRAIN-only record task availability; no GT arrays",
            "manifest_sha256": sha(raw),
            "cache_index_sha256": sha(index_raw),
            "frozen_unified_sha256": sha(code),
            "record_tasks": {key: sorted(value) for key, value in sorted(flags.items())},
        }
        with a.task_flags_output.open("x") as f:
            json.dump(task_metadata, f, sort_keys=True, separators=(",", ":"), allow_nan=False)
            f.write("\n")
        result["train_task_flags_sha256"] = sha(a.task_flags_output.read_bytes())
        result["train_task_flags_records"] = len(flags)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps({"output": str(a.output), "elapsed_seconds": result["elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
