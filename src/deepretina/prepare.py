"""Prepare public annotations directly from ZIP files, without MATLAB or extraction.

Coordinates stay in the original image frame. RGB images are downsampled with
Lanczos, labels with nearest-neighbour; no dataset-specific field-angle warping.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters.labels import decode_av, decode_iostar_disc, decode_refuge
from .sources import Source

__all__ = [
    "ARCHIVES",
    "DATASETS",
    "Archive",
    "assign_splits",
    "decode_av",
    "decode_iostar_disc",
    "decode_refuge",
    "decoded_photo_id",
    "prepare",
]

# Public compatibility name used by existing callers.
Archive = Source

ARCHIVES = {
    "drive": "DRIVE.zip",
    "rite": "RITE.zip",
    "hrf": "HRF.zip",
    "drhagis": "DRHAGIS.zip",
    "refuge": "REFUGE.zip",
    "drishti": "Drishti-GS.zip",
    "chasedb1": "CHASEDB1.zip",
    "iostar": "IOSTAR.zip",
    "aria": "ARIA.zip",
    "avrdb": "AVRDB.zip",
    "g1020": "G1020.zip",
    "onhsd": "Onhsd.zip",
    "riga": "RIGA",
    "rimone": "RIM-ONE.zip",
    "stare": "STARE.zip",
    "uoadr": "UoA_DR.zip",
    "diaretdb0": "DiaRetDB0.zip",
    "diaretdb1": "DiaRetDB1.zip",
    "eophtha": "E-Ophtha.zip",
    "erlangen": "Erlangen.zip",
    "fire": "FIRE.zip",
    "heimed": "HEI-MED.zip",
    "idrid": "IDRiD.zip",
    "inspireavr": "INSPIRE-AVR.zip",
    "messidor": "MESSIDOR.zip",
    "rcrgbma": "RC-RGB-MA.zip",
    "review": "Review.zip",
    "ridb": "RIDB.zip",
    "roc": "ROC.zip",
    "vicavr": "VICAVR.zip",
    "ubmi": "UBMI",
}
DATASETS = tuple(ARCHIVES)
CORE_DATASETS = {"drive", "rite", "hrf", "drhagis", "refuge", "drishti", "chasedb1", "iostar"}


class _LimitReached(Exception):
    """Internal early termination of a bounded dataset preview."""


def _resize(a: np.ndarray, size: tuple[int, int], *, image: bool = False) -> np.ndarray:
    return np.asarray(
        Image.fromarray(a).resize(
            size, Image.Resampling.LANCZOS if image else Image.Resampling.NEAREST
        )
    )


def decoded_photo_id(image: np.ndarray) -> str:
    """SHA256 of original RGB uint8 dimensions and pixels, before any resizing."""
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Photo identity requires RGB uint8")
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(image.tobytes())
    return "sha256:" + digest.hexdigest()


def assign_splits(records: list[dict], seed: int) -> None:
    """Union patient/source groups and exact photos globally before partitioning.

    Photo identity controls the split, not image storage: distinct photographs
    of the same patient remain separate records and image resources.
    """
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

    for row in records:
        source = ("source", row["source_id"])
        find(source)
        if row.get("photo_id"):
            union(source, ("photo", row["photo_id"]))
        if row.get("image"):
            union(source, ("image", row["image"]))
    groups = defaultdict(list)
    for row in records:
        groups[find(("source", row["source_id"]))].append(row)
    pools = defaultdict(list)
    official_validation_families = {
        row["source_id"].split(":")[0]
        for row in records
        if row["split"] == "val" or row.get("metadata", {}).get("official_split") == "val"
    }
    for items in groups.values():
        sources = sorted({row["source_id"] for row in items})
        key = "|".join(sources)
        splits = {row["split"] for row in items}
        if splits - {"train", "val", "test", "unsplit"}:
            raise ValueError(f"Unsupported splits for {key}: {splits}")
        declared = {row.get("metadata", {}).get("official_split") for row in items}
        heldout = (splits | declared) & {"val", "test"}
        train_only = any(
            row.get("metadata", {}).get("split_policy") == "train_only_without_patient_crosswalk"
            for row in items
        )
        if train_only and heldout:
            raise ValueError(f"Train-only source aliases official held-out data: {key}")
        group_id = "sha256:" + hashlib.sha256(key.encode()).hexdigest()
        for row in items:
            row["split_group_id"] = group_id
        if train_only or heldout:
            # Any known heldout alias protects the entire connected patient/photo
            # component. Test is the stricter partition if official aliases disagree.
            fixed = "test" if "test" in heldout else "val" if heldout else "train"
            for row in items:
                row["split"] = fixed
            continue
        families = {source.split(":")[0] for source in sources}
        family = next(iter(families)) if len(families) == 1 else "cross_dataset"
        pool = "train" if "train" in splits or "train" in declared else "unsplit"
        pools[(family, pool)].append((key, items))
    for (family, pool), entries in pools.items():
        ordered = sorted(
            entries, key=lambda entry: hashlib.sha256(f"{seed}:{entry[0]}".encode()).hexdigest()
        )
        n = len(ordered)
        n_val = max(1, round(n * 0.2)) if n >= 3 else 0
        n_test = max(1, round(n * 0.2)) if pool == "unsplit" and n >= 3 else 0
        if pool == "train" and family in official_validation_families:
            n_val = 0
        for i, (_, items) in enumerate(ordered):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            for row in items:
                row["split"] = split


def prepare(
    data_root: Path,
    output: Path,
    datasets: list[str],
    max_size: int = 1024,
    seed: int = 42,
    *,
    limit: int | None = None,
    source_overrides: dict[str, Path] | None = None,
    raw_emitter=None,
) -> Path:
    """Write an independent prepared dataset and return its relative-path manifest.

    Each call requires a new or empty output directory, preventing accidental
    overwrites of datasets used by previous experiments. Sources are never changed.
    """
    data_root, output = Path(data_root), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Output must be a new or empty directory: {output}. Choose a new output path."
        )
    if max_size < 16:
        raise ValueError("max_size must be at least 16")
    datasets = list(dict.fromkeys(d.lower() for d in datasets))
    if not datasets or set(datasets) - set(DATASETS):
        raise ValueError(f"Choose one or more datasets from {', '.join(DATASETS)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    overrides = source_overrides or {}
    if set(overrides) - set(datasets):
        raise ValueError("Source overrides must refer to requested datasets")
    sources = {d: Path(overrides.get(d, data_root / ARCHIVES[d])) for d in datasets}
    for d, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{d}: expected {path}; use --source {d}=/path/to/data to override"
            )
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    records = []
    seen_ids = set()
    preview_groups = defaultdict(set)
    copied_annotations = {}
    image_resources = {}
    photo_resources = {}
    active_source = None

    def save(
        d,
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
        source_id = source_id or f"{d}:{stem}"
        if not source_id.startswith(d + ":") and not (
            d == "rite" and source_id.startswith("drive:")
        ):
            source_id = f"{d}:{source_id}"
        if (
            limit is not None
            and source_id not in preview_groups[d]
            and len(preview_groups[d]) >= limit
        ):
            raise _LimitReached
        preview_groups[d].add(source_id)
        if split not in {"train", "val", "test", "unsplit"}:
            raise ValueError(f"{d}/{stem}: invalid split {split!r}")
        if raw_emitter is not None:
            row = raw_emitter(
                dataset=d,
                stem=stem,
                split=split,
                image=image,
                target=target,
                valid=valid,
                fov=fov,
                source_id=source_id,
                task=task,
                metadata=dict(metadata or {}),
            )
            if row is not None:
                records.append(row)
            return
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{d}/{stem}: expected RGB uint8 image")
        height, width = image.shape[:2]
        scale = min(1.0, max_size / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if fov is None:
            fov = image.max(axis=-1) > 10
        if fov.shape != (height, width):
            raise ValueError(f"{d}/{stem}: FOV and image dimensions differ")
        fov = _resize(fov.astype(np.uint8), size) > 0
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
        ident = f"{d}_{stem}_{task}"
        if ident in seen_ids:
            raise ValueError(f"Duplicate output id {ident!r}; qualify image/annotator names")
        seen_ids.add(ident)
        details = dict(metadata or {})
        resource_key = details.get("source_image") or details.get("original_resource")
        image_key = (d, resource_key or stem)
        photo_id = photo_resources.get(image_key)
        if photo_id is None:
            photo_id = decoded_photo_id(image)
            photo_resources[image_key] = photo_id
        ip = image_resources.get(image_key)
        if ip is None:
            ip = f"images/{d}_{stem}.png"
            Image.fromarray(_resize(image, size, image=True)).save(output / ip)
            image_resources[image_key] = ip
        row = {
            "id": ident,
            "dataset": d,
            "split": split,
            "image": ip,
            "task": task,
            "source_id": source_id,
            "photo_id": photo_id,
            "original_size": [height, width],
            "preparation_max_size": max_size,
            "preparation_preview": limit is not None,
        }
        if target is not None:
            if task not in {"vessels", "av", "disc_cup"}:
                raise ValueError(f"{d}/{stem}: unsupported dense training task {task!r}")
            target = np.asarray(target, dtype=np.float32)
            expected = ({"vessels": 1, "av": 3, "disc_cup": 2}[task], height, width)
            if (
                target.shape != expected
                or not np.isfinite(target).all()
                or (target < 0).any()
                or (target > 1).any()
            ):
                raise ValueError(f"{d}/{stem}: invalid mask shape, range or finite values")
            if valid is None:
                valid = np.ones_like(target, dtype=bool)
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != expected:
                raise ValueError(f"{d}/{stem}: validity and target dimensions differ")
            target = np.stack([_resize(a, size) for a in target]).astype(np.float32)
            valid = np.stack([_resize(a.astype(np.uint8), size) > 0 for a in valid])
            valid &= fov[None]
            mp = f"masks/{ident}.npz"
            np.savez_compressed(output / mp, target=target, valid=valid, fov=fov)
            row.update(
                mask=mp,
                usable_for_training=bool(valid.any()),
                supervised_channels=valid.any(axis=(1, 2)).tolist(),
            )
        else:
            if task != "images":
                raise ValueError("Unlabelled images must use task='images'")
            (output / "fovs").mkdir(exist_ok=True)
            fp = f"fovs/{ident}.png"
            Image.fromarray(fov.astype(np.uint8) * 255).save(output / fp)
            row.update(fov=fp, usable_for_training=False)
        # Small auxiliary tables and unsupported label types remain portable.
        # Decoded raster labels already live in target/valid; preserve their source
        # identity without duplicating all full-resolution annotation photographs.
        for ref in details.get("annotation_resources", []):
            original = ref.get("path")
            if not original or active_source is None:
                continue
            ref["source_path"] = original
            is_raster = Path(original).suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".ppm",
            }
            if (
                is_raster
                and target is not None
                and row["usable_for_training"]
                and not ref.get("preserve", False)
            ):
                ref["preserved_as"] = "decoded target/valid arrays"
                continue
            key = (d, original)
            if key not in copied_annotations:
                (output / "annotations").mkdir(exist_ok=True)
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(original).name)
                relative = (
                    f"annotations/{d}_{hashlib.sha256(original.encode()).hexdigest()[:12]}_{name}"
                )
                (output / relative).write_bytes(active_source.bytes(original))
                copied_annotations[key] = relative
            ref["prepared_path"] = copied_annotations[key]
        if details:
            row["metadata"] = details
        records.append(row)

    extra = None
    try:
        if any(d in datasets for d in ("hrf", "drhagis")):
            extra_path = data_root / "drive-download-20261019T122612Z-1-001.zip"
            if extra_path.is_file():
                extra = Archive(extra_path)
        for d in datasets:
            z = Archive(sources[d])
            active_source = z
            try:

                def emit(*args, _dataset=d, **kwargs):
                    save(_dataset, *args, **kwargs)

                if d not in CORE_DATASETS:
                    from .adapters.auxiliary import ADAPTERS as auxiliary
                    from .adapters.segmentation import ADAPTERS as segmentation

                    {**auxiliary, **segmentation}[d](z, emit)
                else:
                    from .adapters.core import decode_core

                    decode_core(d, z, emit, extra)
            except _LimitReached:
                pass
            finally:
                z.close()
    finally:
        if extra:
            extra.close()
    if not records:
        raise ValueError("No annotated images matched the requested archives")
    assign_splits(records, seed)
    records.sort(key=lambda r: r["id"])
    manifest = output / "manifest.json"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(manifest)
    archive_paths = [sources[d] for d in datasets]
    if any(r["task"] == "av" and r["dataset"] in ("hrf", "drhagis") for r in records):
        archive_paths.append(data_root / "drive-download-20261019T122612Z-1-001.zip")
    provenance = {
        "schema_version": 1,
        "datasets": datasets,
        "max_size": max_size,
        "seed": seed,
        "preview_limit_per_dataset": limit,
        "preview_only": limit is not None,
        "split_policy": "global patient/source and original RGB-photo union; official heldout test>val; train-only conflicts rejected; SHA256 ranked20%validation/20%test for unsplit sources",
        "photo_identity": "sha256(original RGB uint8 shape as little-endian int64, then contiguous decoded pixels); computed before resizing and cached per original resource",
        "archive_identity_note": "Size, mtime and ZIP member CRCs are recorded; this is not a cryptographic content hash.",
        "archives": [],
    }
    for path in archive_paths:
        with Source(path) as source:
            provenance["archives"].append(source.identity())
    (output / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return manifest
