"""Append explicitly reviewed native sources to a copied immutable corpus v1.

No source data or parent corpus is modified. Existing rows and pixel files remain
exactly preserved; all changes are confined to a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from deepretina.adapters.additional import drions, gamma, idrid_disc
from deepretina.adapters.biomisa import FIXED_POLICY, biomisa
from deepretina.integrity import verify_prepared_content
from deepretina.sources import Source

_spec = importlib.util.spec_from_file_location(
    "_v2_verified_parent_writer", Path(__file__).with_name("prepare_unified_data.py")
)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

PARENT_FINGERPRINT = "61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def clone_parent(parent, output, expected=PARENT_FINGERPRINT):
    """CoW copies share disk extents, never writable parent inodes."""
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("Choose a new independent corpus-v2 output directory")
    if parent.is_symlink():
        raise ValueError("Parent must resolve to a local canonical directory")
    verify_prepared_content(parent / "manifest.json", expected=expected)
    for path in parent.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Parent contains a symlink: {path}")
    output.mkdir(parents=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", "--", str(parent) + "/.", str(output)], check=True
    )
    count = 0
    for path in parent.rglob("*"):
        if not path.is_file():
            continue
        copy = output / path.relative_to(parent)
        if (path.stat().st_dev, path.stat().st_ino) == (copy.stat().st_dev, copy.stat().st_ino):
            raise ValueError(f"Writable inode shared with parent: {path}")
        count += 1
    verify_prepared_content(output / "manifest.json", expected=expected)
    # A read-only parent is expected. Change owner modes only on independent copies.
    for directory in [output, *(p for p in output.rglob("*") if p.is_dir())]:
        directory.chmod(directory.stat().st_mode | 0o700)
    for name in (
        "manifest.json",
        "photos.json",
        "composition.json",
        "preparation.json",
        "excluded.json",
        "integrity.json",
        "_SUCCESS",
    ):
        p = output / name
        if p.exists():
            p.chmod(p.stat().st_mode | 0o200)
            backup = output / "parent-v1-metadata"
            backup.mkdir(exist_ok=True)
            shutil.copy2(p, backup / name)
    (output / "_SUCCESS").unlink()
    (output / "integrity.json").unlink()
    return {
        "method": "cp-a-reflink-auto; copy fallback permitted",
        "parent_files_checked_distinct_inode": count,
        "parent_fingerprint": expected,
    }


def resolve_partitions(parent_rows, rows):
    """Keep parent membership fixed; quarantine new TRAIN aliases of its holdouts."""
    parent_ids = {r["id"] for r in parent_rows}
    union = {}

    def find(x):
        union.setdefault(x, x)
        if union[x] != x:
            union[x] = find(union[x])
        return union[x]

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            union[max(a, b)] = min(a, b)

    for r in rows:
        join(("source", r["source_id"]), ("photo", r["photo_id"]))
    groups = {}
    for r in rows:
        groups.setdefault(find(("source", r["source_id"])), []).append(r)
    quarantine = []
    for group in groups.values():
        frozen = {r["split"] for r in group if r["id"] in parent_ids}
        if len(frozen) > 1:
            raise ValueError("New alias joins multiple frozen parent partitions")
        declared = {
            r.get("metadata", {}).get("official_split") for r in group if r["id"] not in parent_ids
        } & {"val", "test"}
        if len(declared) > 1:
            raise ValueError("Conflicting new official heldout partitions")
        frozen_groups = {
            r.get("split_group_id")
            for r in group
            if r["id"] in parent_ids and r.get("split_group_id")
        }
        if len(frozen_groups) > 1:
            raise ValueError("New alias bridges multiple frozen parent split groups")
        if frozen and declared - set(frozen):
            raise ValueError("New official heldout alias conflicts with frozen parent membership")
        fixed = (
            next(iter(frozen))
            if frozen
            else "test"
            if "test" in declared
            else "val"
            if "val" in declared
            else "train"
        )
        for r in group:
            if r["id"] in parent_ids:
                continue
            original = r["metadata"].get("source_split", r["split"])
            if fixed != "train" and (
                original in {"train", "unsplit"}
                or r["metadata"].get("split_policy") == "train_only_without_patient_crosswalk"
            ):
                quarantine.append(
                    {
                        "id": r["id"],
                        "reason": "new training annotation aliases protected heldout source/photo group",
                        "protected_split": fixed,
                        "mask": r["mask"],
                        "source_id": r["source_id"],
                        "photo_id": r["photo_id"],
                    }
                )
                continue
            r["split"] = fixed
            r["split_group_id"] = (
                next(iter(frozen_groups))
                if frozen_groups
                else (
                    "sha256:"
                    + hashlib.sha256(
                        "|".join(sorted({x["source_id"] for x in group})).encode()
                    ).hexdigest()
                )
            )
    excluded = {r["id"] for r in quarantine}
    kept = [r for r in rows if r["id"] not in excluded]
    if [r for r in kept if r["id"] in parent_ids] != parent_rows:
        raise ValueError("Frozen parent rows changed")
    return kept, quarantine


GEOMETRY_POLICY_SHA256 = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"


def angle_for(policy, dataset, shape):
    """Select exact native camera shape before the explicitly nominal fallback."""
    if dataset not in policy["policies"]:
        raise ValueError("No explicit reviewed angle for dataset: " + dataset)
    entry = deepcopy(policy["policies"][dataset])
    shape_key = f"{shape[0]}x{shape[1]}"
    if "per_shape" in entry:
        matched = shape_key in entry["per_shape"]
        entry = deepcopy(entry["per_shape"][shape_key] if matched else entry["default"])
        attribution = "native_shape_rule" if matched else "explicit_unseen_shape_nominal_fallback"
    else:
        attribution = "dataset_acquisition_description_or_explicit_nominal_research_policy"
    angle = entry["field_of_view_degrees"]
    if isinstance(angle, bool) or not math.isfinite(float(angle)) or not 0 < float(angle) <= 180:
        raise ValueError("Reviewed field angle must be finite, positive and explicit")
    entry.update(
        policy_id=policy["policy_id"],
        native_shape_hw=list(shape[:2]),
        camera_attribution_method=attribution,
    )
    return entry


def append(
    parent,
    output,
    data_root,
    policy_path,
    *,
    biomisa_policy,
    expected_parent=PARENT_FINGERPRINT,
    expected_policy=GEOMETRY_POLICY_SHA256,
):
    policy_bytes = Path(policy_path).read_bytes()
    policy = json.loads(policy_bytes)
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    if policy_sha != expected_policy:
        raise ValueError("Geometry policy differs from explicit reviewed fingerprint")
    if biomisa_policy != FIXED_POLICY:
        raise ValueError("Corpus-v2 requires explicit reviewed BIOMISA filename-role policy")
    verify_prepared_content(Path(parent) / "manifest.json", expected=expected_parent)
    parent_rows = json.loads((Path(parent) / "manifest.json").read_text())
    parent_photos = json.loads((Path(parent) / "photos.json").read_text())
    parent_photo_index = {p["photo_id"]: p for p in parent_photos}
    parent_files = {r[k] for r in parent_rows for k in ("image", "mask")} | {
        p[k] for p in parent_photos for k in ("image", "fov")
    }
    # Keep path/byte proof independent from content fingerprint and parent inode identity.
    parent_hashes = {}
    hash_cache = {}
    for rel in sorted(parent_files):
        p = Path(parent) / rel
        st = p.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in hash_cache:
            with p.open("rb") as f:
                hash_cache[key] = hashlib.file_digest(f, "sha256").hexdigest()
        parent_hashes[rel] = hash_cache[key]
    copy_proof = clone_parent(parent, output, expected_parent)
    writer = base.UnifiedWriter(output, data_root)
    writer.rows = deepcopy(parent_rows)
    writer.ids = {r["id"] for r in parent_rows}
    writer.photos = {p["raw_rgb_sha256"]: deepcopy(p) for p in parent_photos}
    for p in writer.photos.values():
        p["annotations"] = [a["id"] for a in p["annotations"]]
    resources = {}
    archive_metadata = {}
    progress = []
    for dataset, filename, adapter in [
        ("drions", "DRIONS-DB.zip", drions),
        ("gamma", "GAMMA.zip", gamma),
        ("idrid_disc", "IDRiD.zip", idrid_disc),
        ("biomisa", "BIOMISA.zip", biomisa),
    ]:
        source_path = Path(data_root) / filename
        before = source_path.stat()
        archive_metadata[dataset] = {
            "filename": filename,
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
        with Source(source_path) as source:

            def emit(
                stem,
                split,
                image,
                target,
                valid=None,
                fov=None,
                source_id=None,
                task="disc_cup",
                metadata=None,
                dataset=dataset,
                source=source,
                filename=filename,
            ):
                selected = angle_for(policy, dataset, image.shape)
                base.FIELDS[dataset] = float(selected["field_of_view_degrees"])
                details = deepcopy(metadata or {})
                parent_photo = parent_photo_index.get("sha256:" + base.pixel_sha256(image))
                if parent_photo is not None:
                    if parent_photo["field_of_view_degrees"] != selected["field_of_view_degrees"]:
                        raise ValueError(
                            "New requested angle conflicts with inherited frozen parent geometry; explicit alias review required"
                        )
                    details["geometry_inherited_from_parent"] = {
                        "photo_id": parent_photo["photo_id"],
                        "field_of_view_degrees": parent_photo["field_of_view_degrees"],
                        "geometry": parent_photo["geometry"],
                        "fov_source": parent_photo["fov_source"],
                        "application": "Requested rule angle agrees; frozen parent FOV/crop/canonical geometry is retained, not recomputed",
                    }
                details["native_physical_calibration"] = details.pop("physical_calibration", None)
                details.update(
                    physical_calibration=selected,
                    geometry_policy_sha256=policy_sha,
                    policy_sha256=policy_sha,
                    parent_corpus_fingerprint=expected_parent,
                    angle_source=selected.get("status"),
                    angle_assumed=selected["nominal_assumption"],
                    physical_scale_anchor="original max(H,W) before FOV crop; historical single_net contract",
                )
                for ref in [
                    {"path": details["source_image"], "kind": "source_rgb"},
                    *details.get("annotation_resources", []),
                ]:
                    key = dataset + ":" + ref["path"]
                    if key not in resources:
                        b = source.bytes(ref["path"])
                        resources[key] = {
                            "dataset": dataset,
                            "archive": filename,
                            "resource": ref["path"],
                            "bytes": len(b),
                            "sha256": hashlib.sha256(b).hexdigest(),
                            "kind": ref["kind"],
                        }
                row = writer.emit(
                    dataset=dataset,
                    stem=stem,
                    split=split,
                    image=image,
                    target=target,
                    valid=valid,
                    fov=fov,
                    source_id=source_id,
                    task=task,
                    metadata=details,
                )
                if row is None:
                    raise ValueError(f"New annotation lost all supervision: {dataset}/{stem}")
                geometry = row["metadata"]["original_geometry"]
                box = geometry["crop_position"]
                row["metadata"]["FOV_diameter_pixels_and_estimator_from_actual_geometry"] = {
                    "diameter_pixels": None,
                    "estimator": "No FOV circle estimated by legacy single_net max-dimension coordinate contract",
                    "actual_scale_anchor_pixels": max(image.shape[:2]),
                    "actual_source_FOV_bbox_hw": [box[1] - box[0] + 1, box[3] - box[2] + 1],
                    "rescale_factor": geometry["rescale_factor"],
                    "fov_source": writer.photos[row["metadata"]["raw_rgb_sha256"]]["fov_source"],
                    "retinal_angle_measured": False,
                }
                if row["photo_id"] not in parent_photo_index:
                    writer.photos[row["metadata"]["raw_rgb_sha256"]]["geometry_policy"] = {
                        "policy_sha256": policy_sha,
                        **selected,
                    }
                with np.load(Path(output) / row["mask"], allow_pickle=False) as n:
                    for c in range(target.shape[0]):
                        if (
                            (valid is None or valid[c].any())
                            and target[c].any()
                            and not (n["target"][c] > 0).any()
                        ):
                            raise ValueError(
                                f"Positive annotation vanished in geometry: {row['id']}/{c}"
                            )
                progress.append(row["id"])
                print(
                    json.dumps(
                        {
                            "stage": "append",
                            "dataset": dataset,
                            "new_variants": len(progress),
                            "id": row["id"],
                        }
                    ),
                    flush=True,
                )

            if dataset == "biomisa":
                adapter(source, emit, label_policy=FIXED_POLICY)
            else:
                adapter(source, emit)
        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("New source archive changed during preparation: " + filename)
    rows, quarantine = resolve_partitions(parent_rows, writer.rows)
    by_id = {r["id"]: r for r in rows}
    parent_ids = {r["id"] for r in parent_rows}
    photos = []
    for p in writer.photos.values():
        ids = [i for i in p["annotations"] if i in by_id]
        if not ids:
            continue
        variants = [by_id[i] for i in ids]
        states = {r["split"] for r in variants}
        if len(states) != 1:
            raise ValueError("Photograph crosses partitions")
        if p["photo_id"] in parent_photo_index and all(i in parent_ids for i in ids):
            photos.append(deepcopy(parent_photo_index[p["photo_id"]]))
            continue
        p["split"] = states.pop()
        p["source_ids"] = sorted({r["source_id"] for r in variants})
        p["annotations"] = [
            {k: r[k] for k in ("id", "dataset", "task", "mask")}
            | {
                "annotator": r["metadata"].get("annotator"),
                "official_split": r["metadata"].get("official_split"),
            }
            for r in variants
        ]
        photos.append(p)
    manifest = Path(output) / "manifest.json"
    write_json(manifest, rows)
    write_json(Path(output) / "photos.json", sorted(photos, key=lambda p: p["photo_id"]))
    composition = {
        "parent_corpus_fingerprint": expected_parent,
        "profile": "legacy25_raw",
        "pixels_per_degree": 25,
        "annotations": len(rows),
        "photos": len(photos),
        "counts": [
            {"dataset": k[0], "split": k[1], "task": k[2], "annotations": n}
            for k, n in sorted(Counter((r["dataset"], r["split"], r["task"]) for r in rows).items())
        ],
        "split_policy": "Parent rows immutable; new sources TRAIN-only except original IDRiD TEST; holdout aliases protected",
        "geometry_policy_sha256": policy_sha,
        "preview_only": False,
    }
    write_json(Path(output) / "composition.json", composition)
    (Path(output) / "geometry-policy.json").write_bytes(policy_bytes)
    write_json(Path(output) / "v2-source-resources.json", list(resources.values()))
    write_json(Path(output) / "v2-quarantine.json", quarantine)
    write_json(
        Path(output) / "preparation.json",
        {
            "parent": expected_parent,
            "copy": copy_proof,
            "producer_code_sha256": {
                name: hashlib.sha256((Path(__file__).parents[1] / name).read_bytes()).hexdigest()
                for name in [
                    "scripts/prepare_corpus_v2.py",
                    "scripts/prepare_unified_data.py",
                    "src/deepretina/sources.py",
                    "src/deepretina/integrity.py",
                    "src/deepretina/legacy/preprocessing.py",
                    "src/deepretina/adapters/additional.py",
                    "src/deepretina/adapters/biomisa.py",
                    "uv.lock",
                ]
            },
            "runtime_packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pillow", "scipy", "scikit-image", "opencv-python-headless"]
            },
            "geometry_policy_sha256": policy_sha,
            "new_annotations_attempted": len(progress),
            "source_archives": archive_metadata,
            "new_resource_inventory_sha256": hashlib.sha256(
                (Path(output) / "v2-source-resources.json").read_bytes()
            ).hexdigest(),
            "physical_geometry_measured_universally": False,
            "biomisa_policy": FIXED_POLICY,
        },
    )
    target_hash_cache = {}
    for rel, expected in parent_hashes.items():
        path = Path(output) / rel
        st = path.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in target_hash_cache:
            with path.open("rb") as f:
                target_hash_cache[key] = hashlib.file_digest(f, "sha256").hexdigest()
        actual = target_hash_cache[key]
        if actual != expected:
            raise ValueError(f"Copied parent pixel file changed: {rel}")
    if verify_prepared_content(Path(parent) / "manifest.json", expected_parent) != expected_parent:
        raise ValueError("Parent identity changed")
    (Path(output) / "_SUCCESS").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")
    audit = base.audit_prepared(output)
    write_json(
        Path(output) / "parent-preservation.json",
        {
            "parent_fingerprint": expected_parent,
            "rows_preserved_exactly": len(parent_rows),
            "pixel_files_preserved_byte_exact": len(parent_hashes),
            "source_parent_verified_after_build": True,
            "copy": copy_proof,
            "new_fingerprint": audit["prepared_content_sha256"],
        },
    )
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "data-root", "geometry-policy"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--biomisa-label-policy", choices=[FIXED_POLICY], required=True)
    p.add_argument(
        "--expected-parent-fingerprint",
        default=PARENT_FINGERPRINT,
        help="Explicit audited parent content SHA256; default preserves the historical freeze",
    )
    p.add_argument("--expected-geometry-policy-sha256", default=GEOMETRY_POLICY_SHA256)
    a = p.parse_args()
    result = append(
        a.parent,
        a.output,
        a.data_root,
        a.geometry_policy,
        biomisa_policy=a.biomisa_label_policy,
        expected_parent=a.expected_parent_fingerprint,
        expected_policy=a.expected_geometry_policy_sha256,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
