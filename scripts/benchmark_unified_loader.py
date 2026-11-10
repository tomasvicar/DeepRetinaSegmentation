"""Compare a frozen loader with this checkout, using exact paired CPU samples.

No optimizer, GPU, data mutation or cache eviction is used. This measures the
sample pipeline, not end-to-end epoch throughput. Reference modules must be from
a compatible checkout and are loaded under a separate package module name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

import deepretina.unified as candidate


def load_reference(path):
    name = "deepretina._loader_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paired_sample(dataset, index, seed):
    dataset._image_cache.clear()
    dataset._cache_bytes = 0
    dataset._augmentation_generator = None
    torch.manual_seed(seed)
    started = time.perf_counter()
    sample = dataset[index]
    seconds = time.perf_counter() - started
    generator = dataset._augmentation_generator
    return sample, seconds, torch.get_rng_state(), generator.get_state() if generator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photos-per-source", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only-mild-patch", action="store_true")
    args = parser.parse_args()
    if args.photos_per_source < 1 or args.repeats < 1:
        parser.error("Positive photo count and repeats required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = load_reference(args.reference_module)
    config = {"device": "cpu", "workers": 0, "purpose": "full_data"}
    original = reference.UnifiedDataset(
        args.manifest, "train", reference.UnifiedConfig(**config), return_trace=True
    )
    sources = sorted({row["dataset"] for row in original.rows})
    indices = []
    seen = set()
    for source in sources:
        choices = [
            i
            for i, group in enumerate(original.groups)
            if i not in seen and any(row["dataset"] == source for row in group)
        ]
        count = min(args.photos_per_source, len(choices))
        selected = [choices[k * len(choices) // count] for k in range(count)]
        indices.extend(selected)
        seen.update(selected)
    records = []
    order_rng = random.Random(20261110)
    combinations = [
        ("mild", "patch"),
        ("basic", "patch"),
        ("strong", "patch"),
        ("basic", "whole"),
        ("mild", "whole"),
        ("strong", "whole"),
    ]
    if args.only_mild_patch:
        combinations = combinations[:1]
    for augmentation, view in combinations:
        settings = dict(config, augmentation=augmentation, train_view=view)
        datasets = {
            "reference": reference.UnifiedDataset(
                args.manifest, "train", reference.UnifiedConfig(**settings), return_trace=True
            ),
            "candidate": candidate.UnifiedDataset(
                args.manifest, "train", candidate.UnifiedConfig(**settings), return_trace=True
            ),
        }
        selected = indices if (augmentation, view) == ("mild", "patch") else indices[::4]
        for repeat in range(args.repeats):
            for index in selected:
                methods = list(datasets)
                order_rng.shuffle(methods)
                results = {
                    method: paired_sample(datasets[method], index, 1700 + index + repeat)
                    for method in methods
                }
                left, right = results["reference"], results["candidate"]
                assert all(torch.equal(a, b) for a, b in zip(left[0][:3], right[0][:3]))
                assert left[0][3:] == right[0][3:]
                assert torch.equal(left[2], right[2])
                assert (left[3] is None and right[3] is None) or torch.equal(left[3], right[3])
                records.append(
                    {
                        "photo_id": original.photo_ids[index],
                        "augmentation": augmentation,
                        "view": view,
                        "repeat": repeat,
                        "reference_seconds": left[1],
                        "candidate_seconds": right[1],
                        "bitwise_equal": True,
                        "rng_equal": True,
                    }
                )
            print(f"Verified {augmentation}/{view}, pass {repeat + 1}", flush=True)
    summary = []
    for repeat in range(args.repeats):
        rows = [
            r
            for r in records
            if r["augmentation"] == "mild" and r["view"] == "patch" and r["repeat"] == repeat
        ]
        before = sum(r["reference_seconds"] for r in rows)
        after = sum(r["candidate_seconds"] for r in rows)
        summary.append(
            {
                "repeat": repeat,
                "photos": len(rows),
                "reference_seconds": before,
                "candidate_seconds": after,
                "throughput_ratio": before / after,
            }
        )
    report = {
        "scope": "CPU sample pipeline; not worker/GPU epoch throughput",
        "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_module_sha256": hashlib.sha256(args.reference_module.read_bytes()).hexdigest(),
        "candidate_module_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "threads": 2,
        "sources": sources,
        "distinct_photos": len(indices),
        "paired_samples": len(records),
        "identical_samples_and_rng": len(records),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
