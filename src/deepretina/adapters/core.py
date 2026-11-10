"""Raw readers for the eight initial supervised datasets.

Readers emit original-frame arrays; geometry, storage and split policies belong
in the caller. Official split provenance is attached before deriving validation.
"""

from pathlib import Path

import numpy as np

from .labels import decode_av, decode_iostar_disc, decode_refuge


def decode_core(dataset, source, emit, extra=None):
    d, z = dataset, source
    image_path = None

    def save(_dataset, stem, split, image, target, *args, metadata=None, **kwargs):
        details = dict(metadata or {})
        details.setdefault("source_image", z.find(image_path))
        details.setdefault("source_split", split)
        if d in {"drive", "rite", "refuge", "drishti"}:
            details.setdefault("official_split", split)
        emit(stem, split, image, target, *args, metadata=details, **kwargs)

    if d in ("drive", "rite"):
        for folder, split in (("training", "train"), ("test", "test")):
            for name in z.images(f"{folder}/images", ".tif"):
                stem = Path(name).stem
                num = stem.split("_")[0]
                image_path = f"{folder}/images/{name}"
                image = z.read(image_path, "RGB")
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
                        metadata={
                            "annotator": "first",
                            "source_image": f"{folder}/images/{name}",
                            "source_annotation": f"{folder}/1st_manual/{num}_manual1.gif",
                        },
                    )
                    secondary = z.find(f"{folder}/2nd_manual/{num}_manual2.gif", optional=True)
                    if secondary:
                        vessel2 = z.read(secondary) > 127
                        save(
                            d,
                            stem + "_observer2",
                            split,
                            image,
                            vessel2[None].astype(np.float32),
                            fov=fov,
                            source_id=f"drive:{num}",
                            metadata={
                                "annotator": "second",
                                "source_image": f"{folder}/images/{name}",
                                "source_annotation": secondary,
                            },
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
            image_path = f"{image_folder}/{name}"
            image = z.read(image_path, "RGB")
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
                    f"mask/{stem}_mask.tif" if d == "hrf" else f"Mask_images/{stem}_mask_orig.png"
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
                    raise ValueError(f"{d}/{stem}: new A/V annotation has unexpected size")
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
                image_path = im_path
                image = z.read(image_path, "RGB")
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
                image_path = f"{folder}/Images/{name}"
                image = z.read(image_path, "RGB")
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
            image_path = name
            image = z.read(image_path, "RGB")
            target = (z.read(stem + "_1stHO.png") > 127)[None].astype(np.float32)
            save(
                d,
                stem,
                "unsplit",
                image,
                target,
                source_id=f"{d}:{stem[:-1]}",
                metadata={
                    "annotator": "first",
                    "source_image": name,
                    "source_annotation": stem + "_1stHO.png",
                },
            )
            target2 = (z.read(stem + "_2ndHO.png") > 127)[None].astype(np.float32)
            save(
                d,
                stem + "_observer2",
                "unsplit",
                image,
                target2,
                source_id=f"{d}:{stem[:-1]}",
                metadata={
                    "annotator": "second",
                    "source_image": name,
                    "source_annotation": stem + "_2ndHO.png",
                },
            )
    elif d == "iostar":
        for name in z.images("image", ".jpg"):
            stem = Path(name).stem
            image_path = f"image/{name}"
            image = z.read(image_path, "RGB")
            vessel = z.read(f"GT/{stem}_GT.tif") > 127
            fov = z.read(f"mask/{stem}_Mask.tif") > 127
            target, valid = decode_av(z.read(f"AV_GT/{stem}_AV.tif", "RGB"), vessel)
            shared_image = {"source_image": f"image/{name}"}
            save(
                d,
                stem,
                "unsplit",
                image,
                vessel[None].astype(np.float32),
                fov=fov,
                metadata={**shared_image, "source_annotation": f"GT/{stem}_GT.tif"},
            )
            save(
                d,
                stem,
                "unsplit",
                image,
                target,
                valid,
                fov,
                task="av",
                metadata={**shared_image, "source_annotation": f"AV_GT/{stem}_AV.tif"},
            )
            od_path = f"mask_OD/{stem}_ODMask.tif"
            disc = decode_iostar_disc(z.read(od_path) > 127)
            disc_target = np.stack([disc, np.zeros_like(disc)]).astype(np.float32)
            disc_valid = np.ones_like(disc_target, dtype=bool)
            disc_valid[1] = False
            save(
                d,
                stem + "_disc",
                "unsplit",
                image,
                disc_target,
                disc_valid,
                fov,
                source_id=f"iostar:{stem}",
                task="disc_cup",
                metadata={
                    "source_image": f"image/{name}",
                    "source_annotation": od_path,
                    "cup_annotation": "not supplied",
                },
            )
