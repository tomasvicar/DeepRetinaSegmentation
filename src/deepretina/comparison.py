"""Paired, grouped uncertainty for saved original/unified prediction counts.

No model inference or ground-truth masks are opened here. Historical exposure
and validation selection remain qualifications of the supplied reports.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANNELS = ("vessel", "artery", "vein", "disc", "cup")


def _score(counts):
    # counts: model, annotation source, output, TP/FP/FN.
    denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    dice = np.divide(
        2 * counts[..., 0],
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator != 0,
    )
    available = np.isfinite(dice)
    macro_heads = np.divide(
        np.nansum(dice, axis=1),
        available.sum(axis=1),
        out=np.full((2, 5), np.nan),
        where=available.sum(axis=1) != 0,
    )
    pooled = counts.sum(axis=1)
    denominator = 2 * pooled[..., 0] + pooled[..., 1] + pooled[..., 2]
    pooled_heads = np.divide(
        2 * pooled[..., 0], denominator, out=np.full((2, 5), np.nan), where=denominator != 0
    )
    supported = np.isfinite(macro_heads)
    macro = np.divide(
        np.nansum(macro_heads, axis=1),
        supported.sum(axis=1),
        out=np.full(2, np.nan),
        where=supported.sum(axis=1) != 0,
    )
    return np.concatenate([macro[:, None], pooled_heads, macro_heads], axis=1)


def _counts(value):
    numbers = [value[key] for key in ("tp", "fp", "fn", "valid_pixels")]
    if any(type(number) is not int or number < 0 for number in numbers):
        raise ValueError("Confusion counts must be nonnegative integers")
    if sum(numbers[:3]) > numbers[3]:
        raise ValueError("Confusion counts exceed the valid domain")
    return np.asarray(numbers, dtype=np.int64)


def compare_reports(
    reference_path,
    candidate_path,
    manifest_path,
    *,
    repeats=2000,
    seed=42,
    include_sources=False,
    include_av=False,
):
    """Compare identical photo/GT support with source-stratified group bootstrap.

    Split groups preserve recoverable patient and photograph aliases. Groups with
    opaque patient identity remain photograph/source groups, not proven patients.
    All heads and both pipelines share every bootstrap draw.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 100:
        raise ValueError("Use at least100 integer bootstrap draws")
    paths = [Path(reference_path), Path(candidate_path)]
    report_bytes = [path.read_bytes() for path in paths]
    reports = [json.loads(blob) for blob in report_bytes]
    split = reports[0]["split"]
    if split not in {"val", "test"} or reports[1]["split"] != split:
        raise ValueError("Reports must share one validation/test partition")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if any(report.get("manifest_sha256") != manifest_sha for report in reports):
        raise ValueError("Both reports must identify the supplied manifest SHA256")
    manifest = json.loads(manifest_bytes)
    if isinstance(manifest, dict):
        manifest = manifest["records"]
    partitions = {}
    identifiers = set()
    for row in manifest:
        if row["id"] in identifiers:
            raise ValueError("Duplicate manifest annotation ID")
        identifiers.add(row["id"])
        for key in ("photo_id", "source_id", "split_group_id"):
            identity = row.get(key)
            if identity is not None:
                group = (key, identity)
                if group in partitions and partitions[group] != row["split"]:
                    raise ValueError("Manifest identity group crosses partitions")
                partitions[group] = row["split"]
    rows = [
        row
        for row in manifest
        if row["split"] == split
        and row["task"] in {"vessels", "av", "disc_cup"}
        and row.get("usable_for_training", True)
        and not row.get("exclude_from_training", False)
    ]
    if not rows:
        raise ValueError("No usable annotated photographs in this partition")
    metadata = defaultdict(list)
    for row in rows:
        metadata[row["photo_id"]].append(row)
    aligned = []
    for report in reports:
        entries = report["per_photo"]
        lookup = {entry["photo_id"]: entry for entry in entries}
        if len(lookup) != len(entries) or set(lookup) != set(metadata):
            raise ValueError("Each report must contain each partition photo exactly once")
        aligned.append(lookup)
    photo_ids = sorted(metadata)
    groups = defaultdict(list)
    for photo_id in photo_ids:
        variants = metadata[photo_id]
        group_ids = {row.get("split_group_id") for row in variants}
        if len(group_ids) != 1 or None in group_ids:
            raise ValueError("Each photo requires one explicit reviewed split_group_id")
        groups[next(iter(group_ids))].append(photo_id)
    sources = sorted(
        {
            source
            for lookup in aligned
            for entry in lookup.values()
            for source in entry["channel_datasets"].values()
        }
    )
    source_indices = {source: index for index, source in enumerate(sources)}
    group_ids = sorted(groups)
    grouped = np.zeros((2, len(group_ids), len(sources), 5, 3), dtype=np.int64)
    totals = np.zeros((2, 5, 4), dtype=np.int64)
    av_grouped = (
        np.zeros((2, len(group_ids), len(sources), 4), dtype=np.int64) if include_av else None
    )
    strata = defaultdict(list)
    source_totals = np.zeros((2, len(sources), 5, 4), dtype=np.int64)
    source_support = defaultdict(lambda: {"photos": set(), "groups": set()})
    for group_index, group_id in enumerate(group_ids):
        signature = tuple(
            sorted({aligned[0][photo_id]["dataset"] for photo_id in groups[group_id]})
        )
        strata[signature].append(group_index)
        for photo_id in groups[group_id]:
            left, right = [lookup[photo_id] for lookup in aligned]
            expected_ids = {row["id"] for row in metadata[photo_id]}
            if any(set(entry["record_ids"]) != expected_ids for entry in (left, right)):
                raise ValueError("Annotation variants differ from the supplied manifest")
            if (
                left["channel_datasets"] != right["channel_datasets"]
                or left["dataset"] != right["dataset"]
            ):
                raise ValueError("Paired source attribution differs")
            if include_av:
                av_pair = []
                for entry in (left, right):
                    a, v = [_counts(entry["channels"][name]) for name in ("artery", "vein")]
                    eligible = int(a[0] + a[2] + v[0] + v[2])
                    raw = entry.get("av_conditional_counts", {})
                    values = [
                        raw.get(k, 0)
                        for k in (
                            "correct",
                            "scored_pixels",
                            "eligible_pixels",
                            "end_to_end_correct",
                        )
                    ]
                    if any(type(value) is not int or value < 0 for value in values):
                        raise ValueError("Invalid conditional AV integer counts")
                    correct, scored, known, end = values
                    if (
                        known != eligible
                        or not 0 <= end <= correct <= scored <= known
                        or end != int(a[0] + v[0])
                        or correct > known - int(a[1] + v[1])
                        or a[0] + v[1] > a[0] + a[2]
                        or v[0] + a[1] > v[0] + v[2]
                    ):
                        raise ValueError(
                            "Conditional AV counts contradict exclusive decoder/GT support"
                        )
                    av_pair.append(values)
                if av_pair[0][1:3] != av_pair[1][1:3]:
                    raise ValueError("Paired AV scored/eligible GT domains differ")
                if av_pair[0][2]:
                    source = left["channel_datasets"].get("artery")
                    if source is None or source != left["channel_datasets"].get("vein"):
                        raise ValueError("AV conditional source attribution differs")
                    for model in range(2):
                        av_grouped[model, group_index, source_indices[source]] += av_pair[model]
            for channel, name in enumerate(CHANNELS):
                counts = [_counts(entry["channels"][name]) for entry in (left, right)]
                totals[:, channel] += counts
                if (
                    counts[0][3] != counts[1][3]
                    or counts[0][0] + counts[0][2] != counts[1][0] + counts[1][2]
                ):
                    raise ValueError("Paired valid/positive ground-truth support differs")
                source = left["channel_datasets"].get(name)
                if source is None:
                    if any(value.any() for value in counts):
                        raise ValueError(
                            "Supported output is missing annotation source attribution"
                        )
                    continue
                if include_sources and counts[0][3]:
                    source_support[source, name]["photos"].add(photo_id)
                    source_support[source, name]["groups"].add(group_id)
                    compatible = {
                        "vessel": {"vessels", "av"},
                        "artery": {"av"},
                        "vein": {"av"},
                        "disc": {"disc_cup"},
                        "cup": {"disc_cup"},
                    }
                    if not any(
                        row["dataset"] == source and row["task"] in compatible[name]
                        for row in metadata[photo_id]
                    ):
                        raise ValueError(
                            "Source/head attribution is not eligible for this manifest photo"
                        )
                for model in range(2):
                    source_totals[model, source_indices[source], channel] += counts[model]
                    grouped[model, group_index, source_indices[source], channel] += counts[model][
                        :3
                    ]
    point = _score(grouped.sum(axis=1))
    for model, report in enumerate(reports):
        if "macro_dataset_mean_dice" in report:
            declared = report["macro_dataset_mean_dice"]
            matches = (
                not np.isfinite(point[model, 0])
                if declared is None
                else np.isclose(declared, point[model, 0], rtol=0, atol=1e-12)
            )
            if not matches:
                raise ValueError("Reported macro score disagrees with per-photo confusion counts")
        if "channels" in report:
            for channel, name in enumerate(CHANNELS):
                if not np.array_equal(_counts(report["channels"][name]), totals[model, channel]):
                    raise ValueError("Reported pooled counts disagree with per-photo counts")
    source_differences = (
        np.empty((repeats, len(sources), 5), dtype=np.float64) if include_sources else None
    )
    av_differences = (
        np.empty((repeats, len(sources) + 1, 2), dtype=np.float64) if include_av else None
    )

    def av_rates(counts):
        merged = np.concatenate([counts.sum(axis=1, keepdims=True), counts], axis=1)
        numerator = merged[..., [0, 3]].astype(np.float64)
        denominator = merged[..., [1, 2]]
        return np.divide(
            numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0
        )

    rng = np.random.default_rng(seed)
    differences = np.empty((repeats, 11), dtype=np.float64)
    for repeat in range(repeats):
        draw = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        drawn = grouped[:, draw].sum(axis=1)
        score = _score(drawn)
        differences[repeat] = score[1] - score[0]
        if include_av:
            rates = av_rates(av_grouped[:, draw].sum(axis=1))
            av_differences[repeat] = rates[1] - rates[0]
        if include_sources:
            denominator = 2 * drawn[..., 0] + drawn[..., 1] + drawn[..., 2]
            dice = np.divide(
                2 * drawn[..., 0],
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator != 0,
            )
            source_differences[repeat] = dice[1] - dice[0]

    def result(index):
        values = differences[:, index]
        values = values[np.isfinite(values)]
        return {
            "reference": float(point[0, index]) if np.isfinite(point[0, index]) else None,
            "candidate": float(point[1, index]) if np.isfinite(point[1, index]) else None,
            "candidate_minus_reference": float(point[1, index] - point[0, index])
            if np.isfinite(point[:, index]).all()
            else None,
            "paired_percentile_95ci": np.quantile(values, [0.025, 0.975]).tolist()
            if len(values)
            else None,
            "defined_bootstrap_draws": len(values),
        }

    report = {
        "schema_version": 1,
        "comparison_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": split,
        "photos": len(photo_ids),
        "split_groups": len(groups),
        "strata": {"|".join(key): len(value) for key, value in strata.items()},
        "manifest_sha256": manifest_sha,
        "reference_report_sha256": hashlib.sha256(report_bytes[0]).hexdigest(),
        "candidate_report_sha256": hashlib.sha256(report_bytes[1]).hexdigest(),
        "reference_checkpoint_sha256": reports[0].get("checkpoint_sha256"),
        "candidate_checkpoint_sha256": reports[1].get("checkpoint_sha256"),
        "bootstrap": {
            "draws": repeats,
            "seed": seed,
            "unit": "reviewed split_group_id",
            "stratification": "signature of canonical datasets in the whole group",
        },
        "macro_dataset_mean_dice": result(0),
        "channels": {
            name: {"pooled_dice": result(1 + i), "macro_dataset_dice": result(6 + i)}
            for i, name in enumerate(CHANNELS)
        },
        "independent_quality_improvement_claim": False,
        "qualifications": [
            "Validation results used for tuning are exploratory, including their confidence intervals",
            "Historical original-model exposure and documented preprocessing/backend differences remain qualifications",
            "Split groups preserve known patients/aliases; opaque patient IDs are not proven independent",
            "Identical input hashes and valid/positive counts support pairing; this script does not rerun mask inference",
            "Percentile intervals are conditional on these checkpoints and do not measure training-seed uncertainty",
            "Aggregate gains cannot establish improvement of each individual task",
        ],
    }

    if include_sources:

        def details(counts):
            tp, fp, fn, valid = map(int, counts)

            def ratio(numerator, denominator):
                return numerator / denominator if denominator else None

            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "valid_pixels": valid,
                "dice": ratio(2 * tp, 2 * tp + fp + fn),
                "iou": ratio(tp, tp + fp + fn),
                "precision": ratio(tp, tp + fp),
                "recall": ratio(tp, tp + fn),
            }

        report["by_dataset"] = {}
        for si, source in enumerate(sources):
            channels = {}
            for ci, name in enumerate(CHANNELS):
                left, right = [details(source_totals[model, si, ci]) for model in range(2)]
                values = source_differences[:, si, ci]
                values = values[np.isfinite(values)]
                support = source_support[source, name]
                channels[name] = {
                    "reference": left,
                    "candidate": right,
                    "candidate_minus_reference": {
                        key: right[key] - left[key]
                        if left[key] is not None and right[key] is not None
                        else None
                        for key in left
                    },
                    "paired_dice_percentile_95ci": np.quantile(values, [0.025, 0.975]).tolist()
                    if len(values)
                    else None,
                    "defined_bootstrap_draws": len(values),
                    "supported_photos": len(support["photos"]),
                    "supported_split_groups": len(support["groups"]),
                }
            report["by_dataset"][source] = channels
        report["pooled_counts"] = {}
        for ci, name in enumerate(CHANNELS):
            left, right = [details(totals[model, ci]) for model in range(2)]
            report["pooled_counts"][name] = {
                "reference": left,
                "candidate": right,
                "candidate_minus_reference": {
                    key: right[key] - left[key]
                    if left[key] is not None and right[key] is not None
                    else None
                    for key in left
                },
            }
        report["source_intervals_qualification"] = (
            "Same paired source-stratified split-group draws as primary metric; descriptive simultaneous collection, no multiplicity correction or unknown-patient independence claim"
        )
    if include_av:
        counts = av_grouped.sum(axis=1)
        rates = av_rates(counts)

        def av_result(source_index):
            values = {}
            for metric_index, name in enumerate(("conditional_accuracy", "end_to_end_accuracy")):
                finite = av_differences[:, source_index, metric_index]
                finite = finite[np.isfinite(finite)]
                point = rates[:, source_index, metric_index]
                values[name] = {
                    "reference": float(point[0]) if np.isfinite(point[0]) else None,
                    "candidate": float(point[1]) if np.isfinite(point[1]) else None,
                    "candidate_minus_reference": float(point[1] - point[0])
                    if np.isfinite(point).all()
                    else None,
                    "paired_percentile_95ci": np.quantile(finite, [0.025, 0.975]).tolist()
                    if len(finite)
                    else None,
                    "defined_bootstrap_draws": len(finite),
                }
            pooled = counts.sum(axis=1) if source_index == 0 else counts[:, source_index - 1]
            values["counts"] = {
                name: dict(
                    zip(
                        ("correct", "scored_pixels", "eligible_pixels", "end_to_end_correct"),
                        map(int, pooled[model]),
                        strict=True,
                    )
                )
                for model, name in enumerate(("reference", "candidate"))
            }
            return values

        report["av_accuracy"] = {
            **av_result(0),
            "by_dataset": {source: av_result(index + 1) for index, source in enumerate(sources)},
        }
        for model, original in enumerate(reports):
            for metric_index, keys in enumerate(
                (("av_conditional_accuracy",), ("av_end_to_end_accuracy", "av_accuracy"))
            ):
                for key in keys:
                    if (
                        key in original
                        and original[key] is not None
                        and not np.isclose(
                            original[key], rates[model, 0, metric_index], rtol=0, atol=1e-12
                        )
                    ):
                        raise ValueError("Declared AV accuracy contradicts paired per-photo counts")
    return report
