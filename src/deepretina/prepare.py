"""Prepare public annotations directly from ZIP files, without MATLAB or extraction.

Coordinates stay in the original image frame. RGB images are downsampled with
Lanczos, labels with nearest-neighbour; no dataset-specific field-angle warping.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DATASETS = ("drive", "rite", "hrf", "drhagis", "refuge", "drishti", "chasedb1", "iostar")
ARCHIVES = {
    "drive": "DRIVE.zip",
    "rite": "RITE.zip",
    "hrf": "HRF.zip",
    "drhagis": "DRHAGIS.zip",
    "refuge": "REFUGE.zip",
    "drishti": "Drishti-GS.zip",
    "chasedb1": "CHASEDB1.zip",
    "iostar": "IOSTAR.zip",
}


class Archive:
    """ZIP member index, ignoring Mac metadata and duplicate exported prefixes."""

    def __init__(self, path: Path):
        self.zip = zipfile.ZipFile(path)
        self.names = sorted(
            n
            for n in self.zip.namelist()
            if not n.endswith("/") and "__MACOSX" not in n and not Path(n).name.startswith(".")
        )

    def find(self, suffix: str, *, optional: bool = False) -> str | None:
        suffix = suffix.lower().lstrip("/")
        matches = [n for n in self.names if n.lower() == suffix or n.lower().endswith("/" + suffix)]
        if not matches:
            if optional:
                return None
            raise FileNotFoundError(f"Missing {suffix!r} in {self.zip.filename}")
        # Archives may include the same files once under the full Drive path and
        # once at the root. Prefer the shortest; do not double the dataset.
        return min(matches, key=lambda n: (len(n), n))

    def images(self, folder: str, extension: str) -> list[str]:
        folder = folder.lower().strip("/")
        names = {
            Path(n).name
            for n in self.names
            if ("/" + folder + "/") in ("/" + n.lower()) and n.lower().endswith(extension)
        }
        return sorted(names)

    def read(self, suffix: str, mode: str = "L") -> np.ndarray:
        name = self.find(suffix)
        with self.zip.open(name) as stream, Image.open(io.BytesIO(stream.read())) as image:
            return np.asarray(image.convert(mode)).copy()

    def close(self):
        self.zip.close()


def decode_av(rgb: np.ndarray, vessel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode RGB A/V labels; uncertain/crossing/unlabelled vessels are ignored.

    Artery and vein losses are supervised on confidently labelled vessels only.
    The independent vessel channel uses the original binary segmentation.
    """
    r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
    artery = (r > 180) & (g < 100) & (b < 100)
    vein = (b > 180) & (g < 100) & (r < 100)
    known = (artery | vein) & vessel
    target = np.stack([vessel, artery & vessel, vein & vessel]).astype(np.float32)
    valid = np.stack([np.ones_like(vessel, dtype=bool), known, known])
    return target, valid


def decode_refuge(mask: np.ndarray) -> np.ndarray:
    """REFUGE: 255 background, 128 disc rim, 0 cup (disc includes cup)."""
    return np.stack([mask < 150, mask < 50]).astype(np.float32)


def _resize(a: np.ndarray, size: tuple[int, int], *, image: bool = False) -> np.ndarray:
    return np.asarray(
        Image.fromarray(a).resize(
            size, Image.Resampling.LANCZOS if image else Image.Resampling.NEAREST
        )
    )


def assign_splits(records: list[dict], seed: int) -> None:
    """Group shared source images and patient eyes before making any split."""
    families = defaultdict(list)
    for record in records:
        families[record["source_id"].split(":")[0]].append(record)
    for rows in families.values():
        groups = defaultdict(list)
        for row in rows:
            groups[row["source_id"]].append(row)
        pools = defaultdict(list)
        for source, items in groups.items():
            official = {r["split"] for r in items}
            if len(official) != 1:
                raise ValueError(f"Conflicting official splits for {source}: {official}")
            pools[official.pop()].append(source)
        for pool in ("train", "unsplit"):
            ordered = sorted(
                pools[pool], key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest()
            )
            n = len(ordered)
            if not n:
                continue
            n_val = max(1, round(n * 0.2)) if n >= 3 else 0
            n_test = max(1, round(n * 0.2)) if pool == "unsplit" and n >= 3 else 0
            # Official REFUGE validation is already available.
            if pool == "train" and pools["val"]:
                n_val = 0
            for i, source in enumerate(ordered):
                split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
                for row in groups[source]:
                    row["split"] = split


def prepare(
    data_root: Path, output: Path, datasets: list[str], max_size: int = 1024, seed: int = 42
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
    for d in datasets:
        if not (data_root / ARCHIVES[d]).is_file():
            raise FileNotFoundError(data_root / ARCHIVES[d])
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    records = []

    def save(d, stem, split, image, target, valid=None, fov=None, source_id=None, task="vessels"):
        height, width = image.shape[:2]
        scale = min(1.0, max_size / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if target.shape[1:] != (height, width):
            raise ValueError(f"{d}/{stem}: image and mask dimensions differ")
        if valid is None:
            valid = np.ones_like(target, dtype=bool)
        if fov is None:
            fov = image.max(axis=-1) > 10
        image = _resize(image, size, image=True)
        target = np.stack([_resize(a.astype(np.float32), size) for a in target]).astype(np.float32)
        valid = np.stack([_resize(a.astype(np.uint8), size) > 0 for a in valid])
        fov = _resize(fov.astype(np.uint8), size) > 0
        valid &= fov[None]
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
        ident = f"{d}_{stem}_{task}"
        ip, mp = f"images/{d}_{stem}.png", f"masks/{ident}.npz"
        Image.fromarray(image).save(output / ip)
        np.savez_compressed(output / mp, target=target, valid=valid, fov=fov)
        records.append(
            {
                "id": ident,
                "dataset": d,
                "split": split,
                "image": ip,
                "mask": mp,
                "task": task,
                "source_id": source_id or f"{d}:{stem}",
                "original_size": [height, width],
                "preparation_max_size": max_size,
            }
        )

    extra = None
    try:
        if any(d in datasets for d in ("hrf", "drhagis")):
            extra_path = data_root / "drive-download-20261019T122612Z-1-001.zip"
            if extra_path.is_file():
                extra = Archive(extra_path)
        for d in datasets:
            z = Archive(data_root / ARCHIVES[d])
            try:
                if d in ("drive", "rite"):
                    for folder, split in (("training", "train"), ("test", "test")):
                        for name in z.images(f"{folder}/images", ".tif"):
                            stem = Path(name).stem
                            num = stem.split("_")[0]
                            image = z.read(f"{folder}/images/{name}", "RGB")
                            if d == "drive":
                                vessel = z.read(f"{folder}/1st_manual/{num}_manual1.gif") > 127
                                fov = z.read(f"{folder}/mask/{stem}_mask.gif") > 127
                                save(
                                    d,
                                    stem,
                                    split,
                                    image,
                                    vessel[None].astype(np.float32),
                                    fov=fov,
                                    source_id=f"drive:{num}",
                                )
                            else:
                                vessel = z.read(f"{folder}/vessel/{stem}.png") > 127
                                av = z.read(f"{folder}/av/{stem}.png", "RGB")
                                target, valid = decode_av(av, vessel)
                                save(
                                    d,
                                    stem,
                                    split,
                                    image,
                                    target,
                                    valid,
                                    source_id=f"drive:{num}",
                                    task="av",
                                )
                elif d in ("hrf", "drhagis"):
                    image_folder = "images" if d == "hrf" else "Fundus_Images"
                    for name in z.images(image_folder, ".jpg"):
                        stem = Path(name).stem
                        image = z.read(f"{image_folder}/{name}", "RGB")
                        vessel = (
                            z.read(
                                f"manual1/{stem}.tif"
                                if d == "hrf"
                                else f"Manual_Segmentations/{stem}_manual_orig.png"
                            )
                            > 127
                        )
                        fov = (
                            z.read(
                                f"mask/{stem}_mask.tif"
                                if d == "hrf"
                                else f"Mask_images/{stem}_mask_orig.png"
                            )
                            > 127
                        )
                        source = f"{d}:{stem.split('_')[0]}"
                        save(
                            d,
                            stem,
                            "unsplit",
                            image,
                            vessel[None].astype(np.float32),
                            fov=fov,
                            source_id=source,
                        )
                        av_suffix = f"{d.upper()}/clasified/{stem}.png"
                        if extra and extra.find(av_suffix, optional=True):
                            av = extra.read(av_suffix, "RGB")
                            if av.shape[:2] != vessel.shape:
                                raise ValueError(
                                    f"{d}/{stem}: new A/V annotation has unexpected size"
                                )
                            target, valid = decode_av(av, vessel)
                            save(d, stem, "unsplit", image, target, valid, fov, source, "av")
                elif d == "refuge":
                    for folder, mask_folder, split in (
                        ("Training400", "Annotation-Training400/Disc_Cup_Masks", "train"),
                        ("REFUGE-Validation400", "REFUGE-Validation400-GT/Disc_Cup_Masks", "val"),
                        ("Test400", "REFUGE-Test-GT/Disc_Cup_Masks", "test"),
                    ):
                        for name in z.images(folder, ".jpg"):
                            stem = Path(name).stem
                            # The two training categories and test categories are subdirectories.
                            matches = [
                                n
                                for n in z.names
                                if f"/{folder.lower()}/" in n.lower() and Path(n).name == name
                            ]
                            im_path = min(matches, key=len)
                            matches = [
                                n
                                for n in z.names
                                if f"/{mask_folder.lower()}/" in n.lower()
                                and Path(n).name.lower() == f"{stem}.bmp".lower()
                            ]
                            if not matches:
                                raise FileNotFoundError(f"REFUGE mask for {name}")
                            image = z.read(im_path, "RGB")
                            save(
                                d,
                                stem,
                                split,
                                image,
                                decode_refuge(z.read(min(matches, key=len))),
                                task="disc_cup",
                            )
                elif d == "drishti":
                    for folder, gt, split in (
                        ("Training", "GT", "train"),
                        ("Test", "Test_GT", "test"),
                    ):
                        for name in z.images(f"{folder}/Images", ".png"):
                            stem = Path(name).stem
                            image = z.read(f"{folder}/Images/{name}", "RGB")
                            base = f"{folder}/{gt}/{stem}/SoftMap/{stem}"
                            disc = z.read(base + "_ODsegSoftmap.png") > 127
                            cup = z.read(base + "_cupsegSoftmap.png") > 127
                            save(
                                d,
                                stem,
                                split,
                                image,
                                np.stack([disc | cup, cup]).astype(np.float32),
                                task="disc_cup",
                            )
                elif d == "chasedb1":
                    for name in z.images("CHASEDB1", ".jpg"):
                        stem = Path(name).stem
                        image = z.read(name, "RGB")
                        target = (z.read(stem + "_1stHO.png") > 127)[None].astype(np.float32)
                        save(d, stem, "unsplit", image, target, source_id=f"{d}:{stem[:-1]}")
                elif d == "iostar":
                    for name in z.images("image", ".jpg"):
                        stem = Path(name).stem
                        image = z.read(f"image/{name}", "RGB")
                        vessel = z.read(f"GT/{stem}_GT.tif") > 127
                        fov = z.read(f"mask/{stem}_Mask.tif") > 127
                        target, valid = decode_av(z.read(f"AV_GT/{stem}_AV.tif", "RGB"), vessel)
                        save(d, stem, "unsplit", image, vessel[None].astype(np.float32), fov=fov)
                        save(d, stem, "unsplit", image, target, valid, fov, task="av")
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
    archive_paths = [data_root / ARCHIVES[d] for d in datasets]
    if any(r["task"] == "av" and r["dataset"] in ("hrf", "drhagis") for r in records):
        archive_paths.append(data_root / "drive-download-20261019T122612Z-1-001.zip")
    provenance = {
        "schema_version": 1,
        "datasets": datasets,
        "max_size": max_size,
        "seed": seed,
        "split_policy": "source groups; official splits retained; SHA256 ranked 20% validation, 20% test for unsplit sources",
        "archive_identity_note": "Size, mtime and ZIP member CRCs are recorded; this is not a cryptographic content hash.",
        "archives": [],
    }
    for path in archive_paths:
        stat = path.stat()
        with zipfile.ZipFile(path) as archive:
            members = [
                {"name": info.filename, "size": info.file_size, "crc32": info.CRC}
                for info in archive.infolist()
                if not info.is_dir()
            ]
        provenance["archives"].append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "members": members,
            }
        )
    (output / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return manifest
