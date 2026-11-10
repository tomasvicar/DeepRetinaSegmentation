"""Prepare all suitable supervised sources on the verified legacy25 RGB grid.

The adapter registry decodes original annotations; this writer owns canonical
photograph identity, geometry, variants, patient/alias split safety and inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from deepretina.integrity import FINGERPRINT_SCHEMA, prepared_content_fingerprint
from deepretina.legacy.preprocessing import (
    DATASET_FOV_DEGREES,
    OFFICIAL_FOV_DATASETS,
    ImageGeometry,
    prepare_single_net,
)
from deepretina.prepare import prepare
from deepretina.sources import Source

DEFAULT_DATASETS = (
    "drive",
    "rite",
    "hrf",
    "drhagis",
    "refuge",
    "drishti",
    "chasedb1",
    "iostar",
    "aria",
    "avrdb",
    "g1020",
    "onhsd",
    "riga",
    "rimone",
    "stare",
    "uoadr",
    "erlangen",
)
FIELDS = {**DATASET_FOV_DEGREES, "rimone": 20.0, "erlangen": 20.0}
TASK_CHANNELS = {"vessels": 1, "av": 3, "disc_cup": 2}


def pixel_sha256(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_splits(rows, seed=42):
    """Union patient groups with exact photograph aliases before any partition."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        union("source:" + row["source_id"], "photo:" + row["photo_id"])
    groups = defaultdict(list)
    for row in rows:
        groups[find("source:" + row["source_id"])].append(row)
    pools = defaultdict(list)
    conflicts = []
    official_validation_datasets = {
        r["dataset"] for r in rows if r["metadata"].get("official_split") == "val"
    }
    for items in groups.values():
        sources = sorted({r["source_id"] for r in items})
        key = "|".join(sources)
        official = {r["metadata"].get("official_split") for r in items} & {"train", "val", "test"}
        if len(official) > 1:
            conflicts.append(
                {
                    "source_ids": sources,
                    "official_splits": sorted(official),
                    "resolution": "most restrictive partition: test > val > train",
                }
            )
        fixed = "test" if "test" in official else "val" if "val" in official else None
        for row in items:
            row["split_group_id"] = "sha256:" + hashlib.sha256(key.encode()).hexdigest()
        if fixed:
            for row in items:
                row["split"] = fixed
            continue
        family = (
            sources[0].split(":", 1)[0]
            if len({s.split(":", 1)[0] for s in sources}) == 1
            else "cross_dataset"
        )
        pool = "train" if "train" in official else "unsplit"
        pools[(family, pool)].append((key, items))
    for (family, pool), groups_in_pool in pools.items():
        ordered = sorted(
            groups_in_pool,
            key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
        )
        n = len(ordered)
        n_val = max(1, round(n * 0.2)) if n >= 3 else 0
        n_test = max(1, round(n * 0.2)) if pool == "unsplit" and n >= 3 else 0
        if pool == "train" and family in official_validation_datasets:
            n_val = 0
        for i, (_, items) in enumerate(ordered):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            for row in items:
                row["split"] = split
    partitions = {}
    for row in rows:
        for key in [("source", row["source_id"]), ("photo", row["photo_id"])]:
            if key in partitions and partitions[key] != row["split"]:
                raise ValueError(f"Source/photo leaked across partitions: {key}")
            partitions[key] = row["split"]
        if row["metadata"].get("official_split") == "test" and row["split"] != "test":
            raise ValueError("Official test data moved out of test")
    return conflicts


class UnifiedWriter:
    def __init__(self, output, data_root, *, pixels_per_degree=25.0, preview=False):
        self.output, self.data_root = Path(output), Path(data_root)
        self.ppd = float(pixels_per_degree)
        self.profile = f"legacy{self.ppd:g}_raw"
        self.preview = preview
        self.photos = {}
        self.source_photos = {}
        self.recent = OrderedDict()
        self.rows = []
        self.excluded = []
        self.ids = set()
        self.shared_fov_sources = []

    def _reject(self, dataset, stem, task, reason, metadata):
        self.excluded.append(
            {
                "dataset": dataset,
                "stem": stem,
                "task": task,
                "reason": reason,
                "source_image": metadata.get("source_image", metadata.get("original_resource")),
                "quality": metadata.get("quality"),
                "source_annotations": metadata.get("annotation_resources", []),
            }
        )

    def _rite_fov(self, source_image, split):
        archive = self.data_root / "DRIVE.zip"
        if not archive.is_file():
            return None
        stem = Path(source_image).stem
        folder = "test" if split == "test" else "training"
        suffix = f"{folder}/mask/{stem}_mask.gif"
        with Source(archive) as source:
            path = source.find(suffix, optional=True)
            if path is None:
                return None
            mask = source.read(path) > 0
        self.shared_fov_sources.append({"archive": str(archive.resolve()), "member": path})
        return mask

    def _photo(self, dataset, image, fov, source_id, metadata, split):
        source_image = metadata.get("source_image") or metadata.get("original_resource")
        if not source_image:
            raise ValueError(
                f"{dataset}: a supervised annotation lacks exact source_image provenance"
            )
        resource = (dataset, source_image)
        digest = self.source_photos.get(resource)
        if digest is None:
            digest = pixel_sha256(image)
            self.source_photos[resource] = digest
        photo = self.photos.get(digest)
        source_ref = {
            "dataset": dataset,
            "resource": source_image,
            "source_id": source_id,
            "source_split": metadata.get("source_split", split),
            "official_split": metadata.get("official_split"),
            "declared_field_of_view_degrees": FIELDS[dataset],
        }
        if photo is None:
            image_fov = fov if dataset in OFFICIAL_FOV_DATASETS | {"rimone", "erlangen"} else None
            if dataset == "rite" and image_fov is None:
                image_fov = self._rite_fov(source_image, split)
            prepared = prepare_single_net(
                image,
                dataset=dataset,
                fov=image_fov,
                field_of_view_degrees=FIELDS[dataset],
                pixels_per_degree=self.ppd,
                normalized=False,
            )
            relative = f"images/{digest}.png"
            Image.fromarray(prepared.image).save(self.output / relative)
            (self.output / "fovs").mkdir(exist_ok=True)
            fp = f"fovs/{digest}.png"
            Image.fromarray(prepared.fov.astype(np.uint8) * 255).save(self.output / fp)
            photo = {
                "photo_id": "sha256:" + digest,
                "image": relative,
                "fov": fp,
                "raw_rgb_sha256": digest,
                "raw_shape": list(image.shape),
                "prepared_rgb_sha256": pixel_sha256(prepared.image),
                "prepared_shape": list(prepared.image.shape),
                "geometry": prepared.geometry.as_dict(),
                "canonical_dataset": dataset,
                "pixels_per_degree": self.ppd,
                "field_of_view_degrees": FIELDS[dataset],
                "preprocessing_profile": self.profile,
                "normalized": False,
                "fov_source": prepared.fov_source,
                "sources": [],
                "annotations": [],
            }
            if dataset in {"rimone", "erlangen"}:
                photo["calibration_note"] = (
                    "Field angle from historical MATLAB loader; additional source outside original single_net Python14. "
                    "RIM-ONE retains original full stereo canvas as scale anchor, then crops left-supervised bbox; physical angular calibration needs review."
                    if dataset == "rimone"
                    else "Field angle20deg from historical MATLAB loader; full-frame FOV explicitly supplied."
                )
            self.photos[digest] = photo
            self._cache(digest, prepared.fov, prepared.geometry)
            if len(self.photos) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "prepare",
                            "photos": len(self.photos),
                            "annotations": len(self.rows),
                            "dataset": dataset,
                        }
                    ),
                    flush=True,
                )
        if source_ref not in photo["sources"]:
            photo["sources"].append(source_ref)
        if digest in self.recent:
            prepared_fov, geometry = self.recent.pop(digest)
            self.recent[digest] = (prepared_fov, geometry)
        else:
            with Image.open(self.output / photo["fov"]) as im:
                prepared_fov = np.asarray(im) > 0
            info = photo["geometry"]
            geometry = ImageGeometry(
                tuple(info["original_shape"]),
                tuple(info["crop_position"]),
                info["rescale_factor"],
                tuple(info["prepared_shape"]),
            )
            self._cache(digest, prepared_fov, geometry)
        # Per-group image aliases let the shared reader retain its strict source
        # path invariant. Pixel identity and final union splitting join aliases.
        group_hash = hashlib.sha256(source_id.encode()).hexdigest()[:12]
        alias = f"images/{digest}_{group_hash}.png"
        if not (self.output / alias).exists():
            os.link(self.output / photo["image"], self.output / alias)
        return photo, prepared_fov, geometry, alias

    def _cache(self, digest, fov, geometry):
        self.recent[digest] = (fov, geometry)
        while len(self.recent) > 3:
            self.recent.popitem(last=False)

    def emit(
        self,
        *,
        dataset,
        stem,
        split,
        image,
        target,
        valid=None,
        fov=None,
        source_id=None,
        task="vessels",
        metadata=None,
    ):
        details = dict(metadata or {})
        if task not in TASK_CHANNELS or target is None:
            return self._reject(
                dataset, stem, task, "not a dense vessel/AV/disc/cup annotation", details
            )
        image = np.asarray(image)
        target = np.asarray(target, dtype=np.float32)
        expected = (TASK_CHANNELS[task], *image.shape[:2])
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or target.shape != expected
        ):
            raise ValueError(f"{dataset}/{stem}: invalid original RGB/GT geometry")
        if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
            raise ValueError(f"{dataset}/{stem}: invalid original categorical target")
        valid = np.ones_like(target, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if valid.shape != target.shape:
            raise ValueError(f"{dataset}/{stem}: invalid original validity geometry")
        if not valid.any():
            return self._reject(
                dataset, stem, task, "source annotation quarantined/all channels invalid", details
            )
        details.setdefault("source_split", split)
        if dataset in {"drive", "rite", "drishti", "refuge"}:
            details.setdefault("official_split", split)
        source_id = source_id or f"{dataset}:{stem}"
        photo, prepared_fov, geometry, image_path = self._photo(
            dataset, image, fov, source_id, details, split
        )
        prepared_target = np.stack([geometry.prepare_mask(a) for a in target]).astype(np.float32)
        prepared_valid = np.stack([geometry.prepare_mask(a) for a in valid]).astype(bool)
        prepared_valid &= prepared_fov[None]
        if fov is not None:
            prepared_valid &= geometry.prepare_mask(np.asarray(fov) > 0)[None]
        if not prepared_valid.any():
            return self._reject(
                dataset, stem, task, "no valid supervision after canonical geometry", details
            )
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
        ident = f"{dataset}_{stem}_{task}"
        if ident in self.ids:
            raise ValueError(f"Duplicate annotation ID: {ident}")
        self.ids.add(ident)
        mp = f"masks/{ident}.npz"
        np.savez_compressed(
            self.output / mp, target=prepared_target, valid=prepared_valid, fov=prepared_fov
        )
        details.update(
            {
                "preprocessing_profile": self.profile,
                "pixels_per_degree": self.ppd,
                "source_image": details.get("source_image", details.get("original_resource")),
                "original_geometry": geometry.as_dict(),
                "raw_rgb_sha256": photo["raw_rgb_sha256"],
                "canonical_field_of_view_degrees": photo["field_of_view_degrees"],
                "annotation_resources_copied": False,
            }
        )
        row = {
            "id": ident,
            "dataset": dataset,
            "split": split,
            "source_id": source_id,
            "photo_id": photo["photo_id"],
            "image": image_path,
            "mask": mp,
            "task": task,
            "usable_for_training": True,
            "original_size": list(image.shape[:2]),
            "preprocessing_profile": self.profile,
            "pixels_per_degree": self.ppd,
            "field_of_view_degrees": photo["field_of_view_degrees"],
            "preparation_preview": self.preview,
            "metadata": details,
        }
        photo["annotations"].append(ident)
        self.rows.append(row)
        return row

    def finalize(self, seed=42):
        conflicts = safe_splits(self.rows, seed)
        self.rows.sort(key=lambda r: r["id"])
        by_id = {row["id"]: row for row in self.rows}
        photos = []
        for photo in self.photos.values():
            variants = [by_id[key] for key in photo["annotations"]]
            if not variants:
                continue
            states = {r["split"] for r in variants}
            assert len(states) == 1
            photo["split"] = states.pop()
            photo["source_ids"] = sorted({r["source_id"] for r in variants})
            photo["annotations"] = [
                {
                    "id": r["id"],
                    "dataset": r["dataset"],
                    "task": r["task"],
                    "mask": r["mask"],
                    "annotator": r["metadata"].get("annotator"),
                    "official_split": r["metadata"].get("official_split"),
                }
                for r in variants
            ]
            photos.append(photo)
        photos.sort(key=lambda r: r["photo_id"])
        counts = Counter((r["dataset"], r["task"], r["split"]) for r in self.rows)
        report = {
            "profile": self.profile,
            "pixels_per_degree": self.ppd,
            "normalized": False,
            "tensor_range": "uint8 RGB/255 -0.5",
            "preview_only": self.preview,
            "annotations": len(self.rows),
            "photos": len(photos),
            "excluded": len(self.excluded),
            "counts": [
                {"dataset": k[0], "task": k[1], "split": k[2], "annotations": v}
                for k, v in sorted(counts.items())
            ],
            "exclusion_counts": {
                str(k): v
                for k, v in Counter((r["dataset"], r["reason"]) for r in self.excluded).items()
            },
            "official_alias_conflicts": conflicts,
            "extra_fov_sources": self.shared_fov_sources,
            "identity_policy": "SHA256 of decoded original RGB shape and pixels; exact aliases share first canonical geometry",
            "split_policy": "patient source_id and exact-photo union; official test>val>train; grouped SHA25620%val/20%test for unsplit;20%val from official train unless official val exists",
            "annotations_portable": True,
            "raw_auxiliary_resources_copied": False,
            "source_records_are_variants": True,
        }
        provenance_path = self.output / "preparation.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            provenance["reader_max_size_argument_ignored_by_raw_emitter"] = provenance.pop(
                "max_size", None
            )
            provenance["max_size"] = None
            provenance["preprocessing_profile"] = self.profile
            provenance["pixels_per_degree"] = self.ppd
            provenance["normalized"] = False
            _json(provenance_path, provenance)
        _json(self.output / "manifest.json", self.rows)
        _json(self.output / "photos.json", photos)
        _json(self.output / "composition.json", report)
        _json(self.output / "excluded.json", self.excluded)
        (self.output / "_SUCCESS").write_text(
            hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest() + "\n"
        )
        return report


def prepare_unified_data(
    data_root, output, datasets=DEFAULT_DATASETS, *, seed=42, limit=None, pixels_per_degree=25.0
):
    datasets = list(datasets)
    unknown = set(datasets) - set(DEFAULT_DATASETS)
    if unknown:
        raise ValueError(f"Not dense supervised sources for five outputs: {sorted(unknown)}")
    writer = UnifiedWriter(
        output, data_root, pixels_per_degree=pixels_per_degree, preview=limit is not None
    )
    prepare(
        Path(data_root), Path(output), datasets, seed=seed, limit=limit, raw_emitter=writer.emit
    )
    return writer.finalize(seed)


def audit_prepared(output):
    """Check completed portable data and count actual valid support per head.

    This reads every retained NPZ. Counts distinguish annotation variants from
    physical photos, so six observers never masquerade as six photographs.
    """
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (output / "_SUCCESS").read_text().strip() != manifest_sha:
        raise ValueError("Incomplete preparation or manifest completion hash mismatch")
    rows = json.loads(manifest_path.read_text())
    photos = json.loads((output / "photos.json").read_text())
    photo_index = {photo["photo_id"]: photo for photo in photos}
    if len(photo_index) != len(photos) or set(photo_index) != {r["photo_id"] for r in rows}:
        raise ValueError("Photograph index does not match the manifest")
    indexed_annotations = [item["id"] for photo in photos for item in photo["annotations"]]
    if sorted(indexed_annotations) != sorted(row["id"] for row in rows):
        raise ValueError("Photograph annotation variants do not match the manifest")
    channels = {
        "vessels": ("vessel",),
        "av": ("vessel", "artery", "vein"),
        "disc_cup": ("disc", "cup"),
    }
    counts, partitions, image_shapes, image_hashes, profiles = {}, {}, {}, {}, set()
    ids = set()
    for row in sorted(rows, key=lambda item: item["id"]):
        if row["id"] in ids:
            raise ValueError(f"Duplicate annotation: {row['id']}")
        ids.add(row["id"])
        photo = photo_index[row["photo_id"]]
        if (
            photo["split"] != row["split"]
            or row["source_id"] not in photo["source_ids"]
            or photo["geometry"] != row["metadata"]["original_geometry"]
            or photo["raw_rgb_sha256"] != row["metadata"]["raw_rgb_sha256"]
        ):
            raise ValueError(f"Photograph provenance differs from its annotation: {row['id']}")
        profiles.add((row["preprocessing_profile"], row["pixels_per_degree"]))
        for key in ("source_id", "photo_id"):
            group = (key, row[key])
            if partitions.setdefault(group, row["split"]) != row["split"]:
                raise ValueError(f"Cross-partition {key}: {row[key]}")
        official = row["metadata"].get("official_split")
        if official == "test" and row["split"] != "test":
            raise ValueError(f"Official test moved: {row['id']}")
        if official == "val" and row["split"] == "train":
            raise ValueError(f"Official validation used for training: {row['id']}")
        image_path = output / row["image"]
        stat = image_path.stat()
        file_key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if file_key not in image_shapes:
            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"Prepared image must be RGB: {image_path}")
                image_shapes[file_key] = (image.height, image.width)
            image_hashes[file_key] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if list(image_shapes[file_key]) != photo["prepared_shape"][:2]:
            raise ValueError(f"Prepared photograph geometry differs from its index: {row['id']}")
        photo_key = ("photo_hash", row["photo_id"])
        if partitions.setdefault(photo_key, image_hashes[file_key]) != image_hashes[file_key]:
            raise ValueError(f"Photograph aliases have different prepared RGB: {row['photo_id']}")
        mask_path = output / row["mask"]
        with np.load(mask_path, allow_pickle=False) as arrays:
            target, valid, fov = arrays["target"], arrays["valid"], arrays["fov"]
            expected = (len(channels[row["task"]]), *image_shapes[file_key])
            if target.shape != expected or valid.shape != expected or fov.shape != expected[1:]:
                raise ValueError(f"Invalid prepared geometry: {row['id']}")
            if target.dtype != np.float32 or valid.dtype != bool or fov.dtype != bool:
                raise ValueError(f"Invalid prepared dtypes: {row['id']}")
            if not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
                raise ValueError(f"Nonbinary prepared target: {row['id']}")
            if not valid.any() or (valid & ~fov[None]).any():
                raise ValueError(f"Invalid or empty prepared supervision: {row['id']}")
            if row["task"] == "av":
                known = valid[1] & valid[2]
                if (
                    not np.array_equal(valid[1], valid[2])
                    or (known & ((target[1] + target[2]) != 1)).any()
                ):
                    raise ValueError(f"Invalid conditional AV support: {row['id']}")
            if row["task"] == "disc_cup" and (valid[0] & valid[1] & (target[1] > target[0])).any():
                raise ValueError(f"Cup outside supervised disc: {row['id']}")
            for index, channel in enumerate(channels[row["task"]]):
                key = (row["dataset"], row["split"], channel)
                count = counts.setdefault(
                    key,
                    {
                        "annotation_variants": 0,
                        "photos": set(),
                        "positive_photos": set(),
                        "valid_pixels": 0,
                        "positive_pixels": 0,
                    },
                )
                support = int(valid[index].sum())
                positive = int((valid[index] & (target[index] > 0)).sum())
                if support:
                    count["annotation_variants"] += 1
                    count["photos"].add(row["photo_id"])
                    count["valid_pixels"] += support
                    count["positive_pixels"] += positive
                if positive:
                    count["positive_photos"].add(row["photo_id"])
    if len(profiles) != 1:
        raise ValueError(f"Mixed preparation profiles: {profiles}")
    report = {
        "manifest_sha256": manifest_sha,
        "prepared_content_sha256": prepared_content_fingerprint(manifest_path),
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "annotations_checked": len(rows),
        "photos_checked": len({row["photo_id"] for row in rows}),
        "preview_only": any(row.get("preparation_preview", False) for row in rows),
        "head_counts": [
            {
                "dataset": key[0],
                "split": key[1],
                "channel": key[2],
                **{
                    name: len(value) if isinstance(value, set) else value
                    for name, value in count.items()
                },
            }
            for key, count in sorted(counts.items())
        ],
        "pixel_count_policy": "annotation-variant totals; photos counted once per source/split/head",
    }
    _json(output / "integrity.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="Verify every retained NPZ and count per-head support",
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        help="Functionality preview groups per dataset; never a full-data result",
    )
    args = parser.parse_args()
    if args.audit_existing:
        print(json.dumps(audit_prepared(args.output), indent=2))
        return
    if args.data_root is None:
        parser.error("--data-root is required for preparation")
    print(
        json.dumps(
            prepare_unified_data(
                args.data_root, args.output, args.datasets, seed=args.seed, limit=args.limit
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
