"""Pinned metadata/task-flag sampler check; never opens images/GT/model tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from deepretina.unified import CoverageSampler, UnifiedConfig, UnifiedDataset

MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
FLAGS = "62a4dd1497b8455ac499952b1191fd7fee1e72f4913f8c2ae492ced088049c17"
CACHE_INDEX = "db0ce309ee5ecbf18fa76b7dc8e34dee400b1ea656fff5e99accd204d6dce8ac"
FROZEN = "3260a00bc03b44544556adf5090a412318963fe1b121f7627f265c2fe3aae72d"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(manifest, taskflags, output):
    root = Path(__file__).parents[2]
    reference = root / "tests/reference/vascular_sampling/frozen6b.py"
    spec = importlib.util.spec_from_file_location("frozen6b_sampling_actual", reference)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    provenance = json.loads(reference.with_name("provenance.json").read_text())
    if sha(reference.read_bytes()) != provenance["fixture_sha256"]:
        raise ValueError("Frozen reference changed")
    raw, flags_raw = manifest.read_bytes(), taskflags.read_bytes()
    if sha(raw) != MANIFEST or sha(flags_raw) != FLAGS:
        raise ValueError("Actual input metadata/flags bytes differ")
    flags = json.loads(flags_raw)
    if (
        flags["manifest_sha256"] != MANIFEST
        or flags["cache_index_sha256"] != CACHE_INDEX
        or flags["frozen_unified_sha256"] != FROZEN
    ):
        raise ValueError("Original TRAIN diagnostic pins differ")
    rows = [
        r
        for r in json.loads(raw)
        if r["split"] == "train"
        and r["task"] in {"vessels", "av", "disc_cup"}
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    ]
    if len(rows) != 5700 or {r["id"] for r in rows} != set(flags["record_tasks"]):
        raise ValueError("Exact5700TRAIN record IDs differ")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["photo_id"]].append(row)
    groups = [sorted(values, key=lambda r: r["id"]) for _, values in sorted(grouped.items())]
    if len(groups) != 2529:
        raise ValueError("Exact2529TRAIN photos differ")
    value = object.__new__(UnifiedDataset)
    value.config = UnifiedConfig()
    value.config.validate()
    value.groups = groups
    value._row_tasks = {k: set(v) for k, v in flags["record_tasks"].items()}
    value.support_summary = lambda: None
    old_weights = old.sampling_weights(value)
    weights = value.sampling_weights()
    a, b = np.asarray(old_weights, np.float64), np.asarray(weights, np.float64)
    if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
        raise ValueError("Actual TRAIN default weights changed bits")
    sampler = []
    for seed in [0, 42, 717]:
        left, right = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
        for epoch in range(1, 4):
            x = list(old.CoverageSampler(old_weights, 8192, left, True))
            y = list(CoverageSampler(weights, 8192, right, True))
            if x != y or not torch.equal(left.get_state(), right.get_state()):
                raise ValueError("Actual TRAIN sampler/RNG changed")
            sampler.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "draws": 8192,
                    "indices_sha256": sha(np.asarray(x, np.int64).tobytes()),
                    "rng_sha256": sha(left.get_state().numpy().tobytes()),
                    "identical": True,
                }
            )
    value.config.vascular_sampling_weight = 2
    value.config.validate()
    priority = value.sampling_weights()
    oracle = []
    populations = defaultdict(int)
    memberships = []
    for group in groups:
        source = next(
            iter({str(r["sampling_group"]) for r in group if "sampling_group" in r}),
            group[0]["dataset"],
        )
        keys = {(source, t) for row in group for t in value._row_tasks[row["id"]]}
        memberships.append(keys)
        for key in keys:
            populations[key] += 1
    for keys in memberships:
        oracle.append(
            sum(
                (2 if task in {"vessel", "av"} else 1) / populations[(source, task)]
                for source, task in keys
            )
        )
    if priority != oracle:
        raise ValueError("Priority2 differs from vascular-key oracle")
    result = {
        "kind": "actual_train_flags_sampler_compatibility",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random/unset"),
        "manifest_sha256": MANIFEST,
        "taskflags_sha256": FLAGS,
        "taskflags_source_commit": "312cd431f2c98b221f45967a7b308290144a610f",
        "cache_index_sha256": CACHE_INDEX,
        "frozen_unified_sha256": FROZEN,
        "records": 5700,
        "photos": 2529,
        "membership_keys": len(populations),
        "default_weight_bits_identical": True,
        "default_float64_weights_sha256": sha(a.tobytes()),
        "default_sum_weights": float(a.sum()),
        "sampler_rng_sequences": sampler,
        "priority2_oracle_identical": True,
        "priority2_float64_weights_sha256": sha(np.asarray(priority, np.float64).tobytes()),
        "priority2_sum_weights": float(sum(priority)),
        "new_unified_sha256": sha((root / "src/deepretina/unified.py").read_bytes()),
        "scope": "Only already measured TRAIN support flags and original manifest JSON metadata; no labels/RGB/VAL/TEST arrays/model/training read or run here",
        "expectations_not_actual_training_counts": True,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "taskflags", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.taskflags, args.output)
