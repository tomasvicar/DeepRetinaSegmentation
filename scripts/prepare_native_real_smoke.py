"""Copy four complete canonical TRAIN/VAL photos for native functional smoke only.

No resize, crop, relabelling, split change, TEST input or model execution.
The resulting small corpus is not a full-data training or quality experiment.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from deepretina.integrity import prepared_content_fingerprint

PARENT = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(parent, output):
    parent, output = Path(parent), Path(output)
    if sha(parent / "manifest.json") != PARENT:
        raise ValueError("Require the declared original canonical manifest")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Require a new subset directory")
    rows = json.loads((parent / "manifest.json").read_text())
    selected = []
    for split, dataset, task in (
        ("train", "rite", "av"),
        ("val", "rite", "av"),
        ("train", "drishti", "disc_cup"),
        ("val", "drishti", "disc_cup"),
    ):
        matches = sorted(
            (r for r in rows if (r["split"], r["dataset"], r["task"]) == (split, dataset, task)),
            key=lambda row: row["id"],
        )
        if not matches:
            raise ValueError(f"Missing declared source: {split}/{dataset}/{task}")
        selected.append(matches[0]["photo_id"])
    if len(set(selected)) != 4:
        raise ValueError("Require four distinct photographs")
    subset = [r for r in rows if r["photo_id"] in selected]
    if any(r["split"] not in {"train", "val"} for r in subset):
        raise ValueError("TEST/cross-split photograph forbidden")
    photos = [
        p for p in json.loads((parent / "photos.json").read_text()) if p["photo_id"] in selected
    ]
    paths = {r[key] for r in subset for key in ("image", "mask")}
    paths.update(p[key] for p in photos for key in ("image", "fov"))
    paths.add("geometry-policy.json")
    for relative in paths:
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not (parent / path).resolve().is_relative_to(parent.resolve())
        ):
            raise ValueError("Input path escapes parent")
    output.mkdir(parents=True)
    copied = {}
    for relative in sorted(paths):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(parent / relative)
        shutil.copyfile(parent / relative, target)
        if sha(target) != before or sha(parent / relative) != before:
            raise ValueError("Copy/source byte mismatch")
        copied[relative] = before
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(subset, indent=2) + "\n")
    (output / "photos.json").write_text(json.dumps(photos, indent=2) + "\n")
    manifest_sha = sha(manifest)
    (output / "_SUCCESS").write_text(manifest_sha + "\n")
    fingerprint = prepared_content_fingerprint(manifest)
    proof = {
        "kind": "four_photo_native_functional_smoke_subset_not_quality_or_full_data",
        "parent_manifest_sha256": PARENT,
        "manifest_sha256": manifest_sha,
        "prepared_content_sha256": fingerprint,
        "fingerprint_schema": 2,
        "preview_only": False,
        "photographs": len(photos),
        "annotation_records": len(subset),
        "record_ids": [r["id"] for r in subset],
        "copied_input_sha256": copied,
        "rows_and_photo_metadata_unchanged": True,
        "split_assignments_unchanged": True,
        "TEST_inputs_copied_or_decoded": False,
        "NN_executed": False,
    }
    (output / "integrity.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent, args.output), indent=2))
