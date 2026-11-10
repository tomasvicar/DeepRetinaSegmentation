"""CPU-only paired DataLoader gate; no models, optimization, or corpus writes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import random
import resource
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import deepretina.unified as candidate
from deepretina.derived_cache import VerifiedCache


class RNGTrace(Dataset):
    """Observe worker RNG without drawing from it or altering the sample."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = self.dataset[index]
        base = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        generator = base._augmentation_generator
        return (
            *result,
            hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
            if generator is not None
            else "none",
            torch.utils.data.get_worker_info().id,
        )


def load_reference(path):
    name = "deepretina._workers_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def loader(module, dataset, weights, seed, *, indices=None, rng_trace=False, sample_count=None):
    dataset._image_cache.clear()
    dataset._cache_bytes = 0
    dataset._verified_aliases.clear()
    dataset._augmentation_generator = None
    selected = Subset(dataset, indices) if indices is not None else dataset
    selected_weights = [weights[i] for i in indices] if indices is not None else weights
    if rng_trace:
        selected = RNGTrace(selected)
    generator = torch.Generator().manual_seed(seed)
    sampler = module.CoverageSampler(
        selected_weights,
        sample_count or (len(indices) if indices is not None else 8192),
        generator,
        True,
    )
    return DataLoader(
        selected, batch_size=4, sampler=sampler, num_workers=2, generator=generator
    ), generator


def shutdown(iterator):
    # Close our own bounded benchmark iterators, including prefetched samples.
    iterator._shutdown_workers()


def equal_batches(left, right):
    assert len(left) == len(right)
    for first, second in zip(left, right, strict=True):
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second), "Tensor/order mismatch"
        else:
            assert first == second, "Observer trace or worker RNG mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measured-batches", type=int, default=64)
    parser.add_argument("--derived-cache", type=Path)
    parser.add_argument("--expected-cache-fingerprint")
    parser.add_argument("--expected-canonical-fingerprint")
    parser.add_argument("--allow-canonical-fallback", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 3 and args.warmup_batches >= 1 and args.measured_batches >= 8
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference = load_reference(args.reference_module)
    modules = {"reference": reference, "candidate": candidate}
    frozen = json.loads(args.plan.read_text())["candidates"]["baseline"]
    settings = dict(frozen, device="cpu", workers=2)
    datasets, weights, startup = {}, {}, {}
    cache = None
    if args.derived_cache is not None:
        cache = VerifiedCache(
            args.derived_cache,
            args.manifest,
            expected_cache_fingerprint=args.expected_cache_fingerprint,
            expected_canonical_fingerprint=args.expected_canonical_fingerprint,
            allow_canonical_fallback=args.allow_canonical_fallback,
        )
    for method, module in modules.items():
        started = time.perf_counter()
        datasets[method] = module.UnifiedDataset(
            args.manifest,
            "train",
            module.UnifiedConfig(**settings),
            return_trace=True,
            **({"derived_cache": cache} if method == "candidate" and cache is not None else {}),
        )
        weights[method] = datasets[method].sampling_weights()
        startup[method] = time.perf_counter() - started
        print(
            json.dumps({"stage": "startup", "method": method, "seconds": startup[method]}),
            flush=True,
        )
    assert datasets["reference"].photo_ids == datasets["candidate"].photo_ids
    assert weights["reference"] == weights["candidate"]
    original = datasets["reference"]
    sources = sorted({r["dataset"] for r in original.rows})
    chosen = set()
    for source in sources:
        choices = [
            i
            for i, rows in enumerate(original.groups)
            if i not in chosen and any(r["dataset"] == source for r in rows)
        ]
        chosen.update(choices[:: max(1, len(choices) // 2)][:2])
    if cache is not None:
        selected = set(cache.index["coverage"]["photo_ids"])
        chosen = {i for i, photo in enumerate(original.photo_ids) if photo in selected}
        cache.require_coverage([row for i in chosen for row in original.groups[i]])
        assert len(chosen) == len(selected), "Cache benchmark selects train photographs only"
    else:
        while len(chosen) % 4:
            chosen.add(next(i for i in range(len(original)) if i not in chosen))
    indices = sorted(chosen)
    gates = []
    for setting, (augmentation, view) in enumerate(
        [(a, v) for v in ["patch", "whole"] for a in ["basic", "mild", "strong"]]
    ):
        for method, module in modules.items():
            datasets[method].config = module.UnifiedConfig(
                **dict(settings, augmentation=augmentation, train_view=view)
            )
        built = [
            loader(
                modules[m], datasets[m], weights[m], 1700 + setting, indices=indices, rng_trace=True
            )
            for m in modules
        ]
        iters = [iter(item[0]) for item in built]
        count = 0
        try:
            for left, right in zip(*iters, strict=True):
                equal_batches(left, right)
                count += int(left[0].shape[0])
            assert torch.equal(built[0][1].get_state(), built[1][1].get_state())
        finally:
            for iterator in iters:
                shutdown(iterator)
        gates.append(
            {
                "augmentation": augmentation,
                "view": view,
                "paired_samples": count,
                "tensors_traces_worker_rng_parent_rng_equal": True,
            }
        )
        print(json.dumps({"stage": "equivalence", **gates[-1]}), flush=True)

    for method, module in modules.items():
        datasets[method].config = module.UnifiedConfig(**settings)
    order_rng = random.Random(20261110)
    records = []
    for repeat in range(args.repeats):
        order = list(modules)
        order_rng.shuffle(order)
        results, kept = {}, {}
        for method in order:
            dl, generator = loader(
                modules[method],
                datasets[method],
                weights[method],
                42 + repeat,
                indices=indices if cache is not None else None,
                sample_count=8192,
            )
            started = time.perf_counter()
            iterator = iter(dl)
            try:
                for _ in range(args.warmup_batches):
                    next(iterator)
                warmed = time.perf_counter()
                batches = [next(iterator) for _ in range(args.measured_batches)]
                ended = time.perf_counter()
                kept[method] = batches
                results[method] = {
                    "iterator_and_warmup_seconds": warmed - started,
                    "measured_seconds": ended - warmed,
                    "samples_per_second": 4 * args.measured_batches / (ended - warmed),
                    "parent_rng": generator.get_state().clone(),
                }
            finally:
                shutdown(iterator)
        for left, right in zip(kept["reference"], kept["candidate"], strict=True):
            equal_batches(left, right)
        assert torch.equal(
            results["reference"].pop("parent_rng"), results["candidate"].pop("parent_rng")
        )
        record = {
            "repeat": repeat,
            "execution_order": order,
            **results,
            "throughput_ratio": results["reference"]["measured_seconds"]
            / results["candidate"]["measured_seconds"],
            "measured_batches_and_traces_exact": True,
        }
        measured_indices = [int(i) for batch in kept["reference"] for i in batch[3]]
        record["distinct_photos"] = len(set(measured_indices))
        record["sources"] = sorted(
            {row["dataset"] for i in measured_indices for row in original.groups[i]}
        )
        record["retained_tensor_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for batches in kept.values()
            for batch in batches
            for tensor in batch
            if isinstance(tensor, torch.Tensor)
        )
        records.append(record)
        del kept
        print(json.dumps({"stage": "timing", **record}), flush=True)
    report = {
        "scope": "CPU DataLoader delivery only; no model/GPU/epoch speed claim",
        "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_module_sha256": hashlib.sha256(args.reference_module.read_bytes()).hexdigest(),
        "candidate_module_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "frozen_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "runtime": {
            "torch": str(torch.__version__),
            "parent_threads": 2,
            "workers": 2,
            "batch_size": 4,
            "worker_threads": 1,
            "prefetch_factor": 2,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "training_config": dataclasses.asdict(datasets["candidate"].config),
        "derived_cache": cache.identity if cache is not None else None,
        "cache_coverage": cache.index["coverage"] if cache is not None else None,
        "startup_seconds_excluded": startup,
        "gate_sources": sources,
        "gate_photos": len(indices),
        "gates": gates,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.measured_batches,
        "timing": records,
        "max_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
