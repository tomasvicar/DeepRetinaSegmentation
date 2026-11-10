"""Audit native TRAIN additions against frozen corpus-v1 identities, no training.

Output is metadata/resource inventory only, NOT a ready training manifest. No
original or v1 image/mask is modified, no v1/test mask is decoded, and physical
geometry is not silently inferred for DRIONS/GAMMA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from deepretina.adapters.additional import drions, gamma, idrid_disc, resources
from deepretina.sources import Source

V1_MANIFEST = "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"


def pixel_sha256(image):
    h = hashlib.sha256()
    h.update(np.asarray(image.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return h.hexdigest()


def proposal(data_root, v1_manifest, progress=None):
    payload = Path(v1_manifest).read_bytes()
    if hashlib.sha256(payload).hexdigest() != V1_MANIFEST:
        raise ValueError("Expected frozen corpus-v1 manifest529")
    v1 = json.loads(payload)
    by_photo = defaultdict(list)
    by_group = defaultdict(set)
    for row in v1:
        by_photo[row["photo_id"]].append(row)
        by_group[(row["dataset"], row["source_id"])].add(row["split"])
    rows = []
    used = {}
    archives = {}
    test_names = []
    for dataset, filename, adapter in [
        ("drions", "DRIONS-DB.zip", drions),
        ("gamma", "GAMMA.zip", gamma),
        ("idrid_disc", "IDRiD.zip", idrid_disc),
    ]:
        path = Path(data_root) / filename
        archives[dataset] = {"filename": filename, "bytes": path.stat().st_size}
        with Source(path) as source:
            if dataset == "idrid_disc":
                test_names = [
                    n
                    for n in resources(source, "IDRiD")
                    if "/b. Testing Set/5. Optic Disc/" in n and n.endswith("_OD.tif")
                ]

            def emit(
                stem,
                split,
                image,
                target,
                valid=None,
                fov=None,
                source_id=None,
                task=None,
                metadata=None,
                dataset=dataset,
            ):
                if split != "train":
                    raise ValueError("This proposal is TRAIN annotations only")
                digest = pixel_sha256(image)
                photo = "sha256:" + digest
                matches = by_photo.get(photo, [])
                heldout = any(r["split"] in {"val", "test"} for r in matches)
                group_dataset = source_id.split(":", 1)[0]
                group_matches = by_group.get((group_dataset, source_id), set())
                if group_matches & {"val", "test"}:
                    heldout = True
                refs = []
                for name in [metadata["source_image"], metadata["source_annotation"]]:
                    key = dataset + ":" + name
                    if key not in used:
                        raw = source.bytes(name)
                        used[key] = {
                            "dataset": dataset,
                            "resource": name,
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    refs.append(key)
                rows.append(
                    {
                        "id": dataset + "_" + stem,
                        "dataset": dataset,
                        "source_id": source_id,
                        "photo_id": photo,
                        "source_split": split,
                        "proposed_split": "quarantined_v1_holdout_alias" if heldout else "train",
                        "shape": list(image.shape[:2]),
                        "task": task,
                        "target_positive_pixels": [int((c > 0).sum()) for c in target],
                        "valid_pixels": [int(image.shape[0] * image.shape[1])] * target.shape[0]
                        if valid is None
                        else [int(c.sum()) for c in valid],
                        "v1_aliases": [
                            {
                                "id": r["id"],
                                "dataset": r["dataset"],
                                "source_id": r["source_id"],
                                "split": r["split"],
                            }
                            for r in matches
                        ],
                        "v1_known_group_partitions": sorted(group_matches),
                        "resources": refs,
                        "metadata": metadata,
                        "training_ready": False,
                        "reason": "Metadata proposal only; native annotations verified, corpus-v2 geometry/composition must be reviewed and frozen separately",
                    }
                )

            capture = emit

            def emit(*args, _capture=capture, **kwargs):
                _capture(*args, **kwargs)
                if progress is not None:
                    progress(rows[-1])

            if dataset == "idrid_disc":
                adapter(source, emit, splits=("train",))
            else:
                adapter(source, emit)
    counts = Counter(r["dataset"] for r in rows)
    return {
        "schema_version": 1,
        "purpose": "native_train_addition_proposal_not_training_manifest",
        "v1_manifest_sha256": V1_MANIFEST,
        "v1_original_row_count": len(v1),
        "v1_memberships_changed": False,
        "v1_masks_opened": False,
        "test_annotations_decoded": False,
        "candidate_records": rows,
        "resources": list(used.values()),
        "source_archives": archives,
        "counts": {
            "annotation_variants": dict(counts),
            "unique_photos": {
                d: len({r["photo_id"] for r in rows if r["dataset"] == d}) for d in counts
            },
            "v1_exact_photo_alias_variants": sum(bool(r["v1_aliases"]) for r in rows),
            "quarantined_holdout_alias_variants": sum(r["proposed_split"] != "train" for r in rows),
        },
        "official_test_annotation_filename_inventory_only": {"idrid_disc": sorted(test_names)},
        "limits": [
            "Decoded RGB exact hashes do not detect re-encoding/resizing or unknown cross-dataset patient identities.",
            "All new unknown-patient samples remain TRAIN-only; official IDRiD TEST names are inventoried but unopened.",
            "DRIONS/GAMMA physical angles unresolved; no generic45-degree substitution or changed v1 experiment.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--v1-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.with_suffix(".jsonl").open("x") as journal:

        def progress(row):
            journal.write(json.dumps(row) + "\n")
            journal.flush()

        result = proposal(a.data_root, a.v1_manifest, progress)
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
