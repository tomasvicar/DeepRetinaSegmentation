"""Fail-closed audit of proposed held-out photographs against original checkpoints.

Synthetic legacy indices cannot be mapped from clean filenames: require an
explicit reviewed crosswalk. This tool audits metadata and never reads masks,
changes splits, or claims that a historical test partition was untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(manifest_path, index_paths, crosswalk_path):
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest["records"] if isinstance(manifest, dict) else manifest
    crosswalk = json.loads(Path(crosswalk_path).read_text())
    if crosswalk.get("schema_version") != 1:
        raise ValueError("Crosswalk requires schema_version=1")
    maps = {}
    for item in crosswalk["mappings"]:
        key = (item["photo_id"], item["family"])
        if key in maps:
            raise ValueError(f"Duplicate crosswalk key: {key}")
        maps[key] = item
    indices = {name: json.loads(Path(path).read_text()) for name, path in index_paths.items()}
    photos = {}
    for row in rows:
        if not row.get("photo_id"):
            raise ValueError("Explicit photo_id is required; source_id may describe a patient")
        photo = photos.setdefault(row["photo_id"], {"records": [], "splits": set()})
        photo["records"].append(row.get("id"))
        photo["splits"].add(row["split"])
    reports = []
    for photo_id, photo in sorted(photos.items()):
        checks = {}
        for family, index in indices.items():
            item = maps.get((photo_id, family))
            check = {"gradient_unseen": False, "historically_untouched": False}
            if not item or item.get("verified") is not True or not item.get("evidence"):
                check["status"] = "unrecoverable_or_unverified_mapping"
            else:
                ids = item.get("legacy_image_ids", [])
                if not ids:
                    if item.get("verified_absent_from_corpus") is True:
                        check.update(
                            status="verified_absent",
                            gradient_unseen=True,
                            historically_untouched=True,
                        )
                    else:
                        check["status"] = "absence_not_proven"
                else:
                    missing = sorted(set(ids) - set(index["images"]))
                    if missing:
                        check.update(status="invalid_legacy_ids", missing=missing)
                    else:
                        splits = sorted({index["images"][name]["split"] for name in ids})
                        # Original optimization used historical test scores. All indexed
                        # partitions count as exposure for a fully untouched comparison.
                        check.update(
                            status="mapped",
                            legacy_splits=splits,
                            gradient_unseen="train" not in splits,
                            historically_untouched=False,
                        )
                check["evidence"] = item["evidence"]
            checks[family] = check
        split_collision = len(photo["splits"]) != 1
        clean_heldout = photo["splits"] == {"test"}
        reports.append(
            {
                "photo_id": photo_id,
                "record_ids": photo["records"],
                "splits": sorted(photo["splits"]),
                "split_collision": split_collision,
                "clean_heldout": clean_heldout,
                "families": checks,
                "gradient_unseen_all": bool(checks)
                and not split_collision
                and clean_heldout
                and all(c["gradient_unseen"] for c in checks.values()),
                "historically_untouched_all": bool(checks)
                and not split_collision
                and clean_heldout
                and all(c["historically_untouched"] for c in checks.values()),
            }
        )
    return {
        "schema_version": 1,
        "manifest_sha256": sha256(manifest_path),
        "crosswalk_sha256": sha256(crosswalk_path),
        "indices_sha256": {name: sha256(path) for name, path in index_paths.items()},
        "policy": "Unknown identity fails closed; historical valid/test are tuning-exposed",
        "limitations": [
            "A/V original split is unrecoverable unless separately evidenced",
            "Aliases and paired eyes must be grouped before supplying photo_id",
            "No scientific improvement is established by an eligibility audit",
        ],
        "photos": reports,
        "counts": {
            "photos": len(reports),
            "gradient_unseen_all": sum(p["gradient_unseen_all"] for p in reports),
            "historically_untouched_all": sum(p["historically_untouched_all"] for p in reports),
            "split_collisions": sum(p["split_collision"] for p in reports),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--legacy-index", action="append", default=[], metavar="FAMILY=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = dict(item.split("=", 1) for item in args.legacy_index)
    if not indices:
        parser.error("At least one --legacy-index FAMILY=PATH is required")
    report = audit(args.manifest, indices, args.crosswalk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
