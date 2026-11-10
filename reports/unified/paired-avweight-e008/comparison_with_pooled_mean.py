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
    return np.concatenate(
        [macro[:, None], pooled_heads, macro_heads, np.nanmean(pooled_heads, axis=1)[:, None]],
        axis=1,
    )


def _counts(value):
    numbers = [value[key] for key in ("tp", "fp", "fn", "valid_pixels")]
    if any(type(number) is not int or number < 0 for number in numbers):
        raise ValueError("Confusion counts must be nonnegative integers")
    if sum(numbers[:3]) > numbers[3]:
        raise ValueError("Confusion counts exceed the valid domain")
    return np.asarray(numbers, dtype=np.int64)


def compare_reports(reference_path, candidate_path, manifest_path, *, repeats=2000, seed=42):
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
    strata = defaultdict(list)
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
                for model in range(2):
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
    rng = np.random.default_rng(seed)
    differences = np.empty((repeats, 12), dtype=np.float64)
    for repeat in range(repeats):
        draw = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        score = _score(grouped[:, draw].sum(axis=1))
        differences[repeat] = score[1] - score[0]

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

    return {
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
        "pooled_mean_dice": result(11),
        "channels": {
            name: {"pooled_dice": result(1 + i), "macro_dataset_dice": result(6 + i)}
            for i, name in enumerate(CHANNELS)
        },
        "independent_quality_improvement_claim": False,
        "qualifications": [
            "Validation results used for tuning are exploratory, including their confidence intervals",
            "Historical original-model exposure and raw MATLAB proxy limitations remain unresolved",
            "Split groups preserve known patients/aliases; opaque patient IDs are not proven independent",
            "Identical input hashes and valid/positive counts support pairing; this script does not rerun mask inference",
            "Percentile intervals are conditional on these checkpoints and do not measure training-seed uncertainty",
            "Aggregate gains cannot establish improvement of each individual task",
        ],
    }
