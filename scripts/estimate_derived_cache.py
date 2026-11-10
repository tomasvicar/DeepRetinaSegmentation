"""Estimate full train/validation cache capacity from a verified bounded sample."""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from deepretina.derived_cache import VerifiedCache


def estimate(manifest, cache, expected_cache, expected_canonical):
    manifest = Path(manifest)
    opened = VerifiedCache(
        cache,
        manifest,
        expected_cache_fingerprint=expected_cache,
        expected_canonical_fingerprint=expected_canonical,
    )
    rows = [row for row in json.loads(manifest.read_text()) if row["split"] in {"train", "val"}]
    photos = {
        photo["photo_id"]: photo
        for photo in json.loads((manifest.parent / "photos.json").read_text())
    }
    images = {row["image"]: row["photo_id"] for row in rows}
    # NPY header length depends only on decimal shape widths; compute every real shape.
    rgb_bytes = 0
    for photo_id in images.values():
        shape = tuple(photos[photo_id]["prepared_shape"])
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            buffer, {"descr": "|u1", "fortran_order": False, "shape": shape}
        )
        rgb_bytes += len(buffer.getvalue()) + math.prod(shape)
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[(row["dataset"], row["task"])][row["mask"]] = (
            (manifest.parent / row["mask"]).stat().st_size
        )
    estimates = []
    mappings = opened.index["source_to_derived"]
    for (dataset, task), masks in sorted(groups.items()):
        sampled = [path for path in masks if path in mappings]
        if not sampled:
            raise ValueError(f"Sample lacks dataset/task capacity evidence: {dataset}/{task}")
        sample_original = sum(masks[path] for path in sampled)
        sample_derived = sum(
            opened.index["files"][mappings[path]["derived"]]["bytes"] for path in sampled
        )
        original_bytes = sum(masks.values())
        estimates.append(
            {
                "dataset": dataset,
                "task": task,
                "full_mask_references": len(masks),
                "sample_masks": len(sampled),
                "sample_original_bytes": sample_original,
                "sample_derived_bytes": sample_derived,
                "original_bytes": original_bytes,
                "estimated_derived_bytes": round(original_bytes * sample_derived / sample_original),
            }
        )
    labels_estimate = sum(row["estimated_derived_bytes"] for row in estimates)
    return {
        "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "train+val; metadata-only full-corpus sizing, no full cache built",
        "canonical_fingerprint": expected_canonical,
        "sample_cache_fingerprint": expected_cache,
        "photos": len({row["photo_id"] for row in rows}),
        "records": len(rows),
        "rgb_source_references": len(images),
        "rgb_npy_bytes_before_identical_source_dedup": rgb_bytes,
        "labels_estimated_bytes": labels_estimate,
        "total_estimated_bytes": rgb_bytes + labels_estimate,
        "suggested_free_space_bytes": math.ceil((rgb_bytes + labels_estimate) * 1.3),
        "limitations": [
            "Label compression ratio extrapolated per dataset/task from the 64-photo sample; not an exact bound.",
            "RGB estimate is exact before byte-identical source-file dedup; all aliases counted.",
            "Index/directory filesystem overhead and transient per-file encoding memory are additional.",
            "Soft labels outside this sample remain float and may compress differently.",
        ],
        "by_source_task": estimates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-cache-fingerprint", required=True)
    parser.add_argument("--expected-canonical-fingerprint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = estimate(
        args.manifest,
        args.cache,
        args.expected_cache_fingerprint,
        args.expected_canonical_fingerprint,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "by_source_task"}))


if __name__ == "__main__":
    main()
