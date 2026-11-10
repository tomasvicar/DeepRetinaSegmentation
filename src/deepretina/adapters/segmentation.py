"""Legacy vessel and optic nerve annotation adapters.

These functions decode source annotations only; the shared writer owns resizing,
validity/FOV intersection, provenance and image-group splits.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.io import loadmat

TRAIN_ONLY_PATIENT_POLICY = {
    "riga": {
        "split_policy": "train_only_without_patient_crosswalk",
        "split_policy_evidence": {
            "reason": "Paired eyes are documented, but the seven image folders supply no patient crosswalk",
            "source": "https://uwspace.uwaterloo.ca/bitstreams/29ac8bdf-6c5a-4b1f-ae6e-95d8e60dd510/download",
            "location": "Ahmed Almazroa thesis section3.2, PDFpage71",
        },
    },
    "uoadr": {
        "split_policy": "train_only_without_patient_crosswalk",
        "split_policy_evidence": {
            "reason": "The source paper states that patient information is unavailable after de-identification",
            "source": "UoA_DR.zip/Comparative Analysis of University of Auckland DR Database.pdf",
        },
    },
}


def _base(path):
    return path.replace("\\", "/").split("/")[-1]


def _paths(source, pattern):
    """Deduplicate archive exports using their dataset-relative path suffix."""
    found = {}
    for name in source.names:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            key = match.group(0).lower()
            if key not in found or len(name) < len(found[key]):
                found[key] = name
    return [found[k] for k in sorted(found)]


def _binary(source, name, *, invert=False, channel=None):
    a = source.read(name, "RGB" if channel is not None else "L")
    if channel is not None:
        a = a[..., channel]
    return a < 128 if invert else a > 127


def _polygon(points, shape, *, offset=0):
    points = np.asarray(points, dtype=float) - offset
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Annotation requires at least three finite x,y coordinates")
    canvas = Image.new("1", (shape[1], shape[0]))
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=1)
    return np.asarray(canvas, dtype=bool)


def _disc_channels(disc, cup=None):
    zero = np.zeros_like(disc, dtype=bool)
    target = np.stack([disc, zero if cup is None else cup]).astype(np.float32)
    valid = np.ones_like(target, dtype=bool)
    if cup is None:
        valid[1] = False
    return target, valid


def _largest(mask):
    labels, n = ndi.label(mask, np.ones((3, 3), bool))
    if not n:
        raise ValueError("Empty contour annotation")
    return labels == (1 + np.bincount(labels.ravel())[1:].argmax())


def _close_disk(mask, radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ndi.binary_closing(mask, structure=(x * x + y * y <= radius * radius))


def aria_source_group(name):
    """Conservatively join cohort/number views despite transport counters."""
    match = re.search(r"aria_([acd])_(\d+)(?=[_.]|$)", _base(name), re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot recover the ARIA cohort/number source group: {name}")
    return f"aria:{match[1].lower()}:{int(match[2])}"


def aria(source, emit):
    for name in _paths(source, r"aria_[acd]_markups/aria_[acd]_markups/[^/]+\.tif$"):
        stem = _base(name).rsplit(".", 1)[0]
        image = source.read(name, "RGB")
        group = aria_source_group(name)
        old_group = ":".join(stem.split("_")[:3])
        identity = {
            "source_image_stem": stem,
            "original_source_id": old_group
            if old_group.startswith("aria:")
            else "aria:" + old_group,
            "source_group_basis": "conservative ARIA cohort/number base; transport counter and view tokens ignored",
        }
        for observer in ("BDP", "BSS"):
            mask_name = (
                name.replace("_markups", "_markup_vessel").rsplit(".", 1)[0] + f"_{observer}.tif"
            )
            vessel = source.read(mask_name) > 0
            emit(
                stem + "_" + observer,
                "unsplit",
                image,
                vessel[None].astype(np.float32),
                source_id=group,
                metadata={
                    "source_image": name,
                    "annotator": observer,
                    "source_annotation": mask_name,
                    **identity,
                },
            )
        if group.split(":")[1] == "a":
            continue
        base = name.replace("_markups", "_markupdiscfovea").rsplit(".", 1)[0]
        candidates = [base + suffix for suffix in ("_dfs.tif", ".tif", "_dfd.tif")]
        mask_name = next(
            (source.find(n, optional=True) for n in candidates if source.find(n, optional=True)),
            None,
        )
        if mask_name is None:
            raise FileNotFoundError(f"ARIA disc markup missing for {name}")
        outline = source.read(mask_name) > 0
        disc = _largest(ndi.binary_fill_holes(_close_disk(outline, 21)))
        target, valid = _disc_channels(disc)
        emit(
            stem + "_disc",
            "unsplit",
            image,
            target,
            valid=valid,
            source_id=group,
            task="disc_cup",
            metadata={
                "source_image": name,
                "annotation": "disc-fovea outline: disk21 closing, hole-fill, largest component",
                "source_annotation": mask_name,
                **identity,
            },
        )


def avrdb(source, emit):
    for name in _paths(source, r"IM\d+/IM\d+\.jpg$"):
        stem = _base(name).rsplit(".", 1)[0]
        folder = name.rsplit("/", 1)[0]
        image = source.read(name, "RGB")
        vessel = _binary(source, folder + "/" + stem + "--vessels.jpg", invert=True, channel=0)
        emit(
            stem,
            "unsplit",
            image,
            vessel[None].astype(np.float32),
            valid=vessel[None],
            source_id=stem,
            metadata={
                "source_image": name,
                "source_annotation": folder + "/" + stem + "--vessels.jpg",
                "annotation_support": "sparse vessel lines: unmarked pixels ignored",
            },
        )

        def find_variant(variants, folder=folder, stem=stem):
            for variant in variants:
                path = source.find(folder + "/" + stem + "--" + variant + ".jpg", optional=True)
                if path:
                    return path
            raise FileNotFoundError(f"AVRDB class annotation missing: {stem}, {variants}")

        artery_name = find_variant(("artery", "arteries", "atertries"))
        vein_name = find_variant(("veins", "vein", "veinds", "veisn"))
        artery = _binary(source, artery_name, invert=True, channel=2) & vessel
        vein = _binary(source, vein_name, invert=True, channel=0) & vessel
        known = artery ^ vein
        target = np.stack([vessel, artery & known, vein & known]).astype(np.float32)
        valid = np.stack([vessel, known, known])
        emit(
            stem + "_av",
            "unsplit",
            image,
            target,
            valid=valid,
            source_id=stem,
            task="av",
            metadata={
                "source_image": name,
                "artery_annotation": artery_name,
                "vein_annotation": vein_name,
                "crossings": "ignored",
            },
        )


def decode_g1020(annotation, shape):
    """Rasterize LabelMe coordinates; resolve only geometrically nested mislabels."""
    masks = {"disc": [], "cup": []}
    for item in annotation.get("shapes", []):
        label = item["label"].lower()
        if label in masks:
            masks[label].append(_polygon(item["points"], shape))
    corrections = []
    # Three source files label both nested disc and cup polygons as `disc`.
    if len(masks["disc"]) == 2 and not masks["cup"]:
        ordered = sorted(masks["disc"], key=np.count_nonzero, reverse=True)
        if (ordered[1] & ~ordered[0]).sum() > max(2, ordered[1].sum() * 0.01):
            return (
                np.zeros((2, *shape), dtype=np.float32),
                np.zeros((2, *shape), dtype=bool),
                ["requires_review: duplicate disc polygons are not nested; original JSON retained"],
            )
        masks = {"disc": [ordered[0]], "cup": [ordered[1]]}
        corrections.append("nested inner duplicate disc relabelled cup")
    if any(len(v) > 1 for v in masks.values()):
        raise ValueError("Ambiguous duplicate G1020 disc/cup polygon labels")
    target = np.zeros((2, *shape), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    for index, label in enumerate(("disc", "cup")):
        if masks[label]:
            target[index] = masks[label][0]
            valid[index] = True
    if valid[0].any() and valid[1].any():
        outside = (target[1] > 0) & (target[0] == 0)
        corrections.append(f"cup outside disc pixels excluded from cup validity: {outside.sum()}")
        valid[1, outside] = False
    return target, valid, corrections


def g1020_patient_crosswalk(source):
    """Require the bundled patient/eye table; image numbers are not patients."""
    resource = source.find("G1020.csv")
    payload = source.bytes(resource)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"patientID", "imageID", "Left/Right", "imageName"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("G1020.csv is missing required patient/image/eye columns")
    mapping = {}
    for row in reader:
        image_id, patient = row["imageID"].strip(), row["patientID"].strip()
        if not image_id or not patient or image_id in mapping:
            raise ValueError("G1020.csv contains missing or duplicate patient/image identities")
        mapping[image_id] = {
            "patient_id": patient,
            "patient_group_id": "g1020:patient:" + patient,
            "source_image_id": image_id,
            "source_image_name": row["imageName"].strip(),
            "laterality": row["Left/Right"].strip(),
            "patient_crosswalk": {
                "resource": resource,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
    if not mapping:
        raise ValueError("G1020.csv has no patient identities")
    return mapping


def g1020(source, emit):
    patients = g1020_patient_crosswalk(source)
    for name in _paths(source, r"image_\d+\.jpg$"):
        stem = _base(name).rsplit(".", 1)[0]
        if _base(name) not in patients:
            raise ValueError(f"G1020 image is absent from its patient crosswalk: {name}")
        identity = patients[_base(name)]
        image = source.read(name, "RGB")
        annotation_name = name.rsplit(".", 1)[0] + ".json"
        target, valid, corrections = decode_g1020(
            json.loads(source.text(annotation_name)), image.shape[:2]
        )
        emit(
            stem,
            "unsplit",
            image,
            target,
            valid=valid,
            source_id=identity["patient_group_id"],
            task="disc_cup",
            metadata={
                "source_image": name,
                "source_annotation": annotation_name,
                "corrections": corrections,
                "coordinates": "LabelMe image x,y; zero based; Pillow polygon rasterization",
                "original_source_id": "g1020:" + stem,
                **identity,
            },
        )


def decode_onhsd(center, radii, shape):
    center, radii = np.asarray(center).ravel(), np.asarray(radii).ravel()
    if (
        len(center) != 2
        or len(radii) != 24
        or not np.isfinite(center).all()
        or not np.isfinite(radii).all()
        or (radii <= 0).any()
    ):
        raise ValueError("ONHSD requires a finite center and24 positive radial distances")
    angles = np.arange(1, 25) * np.pi / 12
    points = center[None] - 1 + np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
    if len(np.unique(np.round(points), axis=0)) < 24:
        raise ValueError("ONHSD degenerate repeated radial boundary points")
    return _polygon(np.round(points), shape)


def onhsd(source, emit):
    for name in _paths(source, r"Images/[^/]+\.bmp$"):
        stem = _base(name).rsplit(".", 1)[0]
        image = source.read(name, "RGB")
        base = re.sub(
            r"(^|/)Images/[^/]+$",
            lambda m, stem=stem: m[1] + "Clinicians/" + stem,
            name,
            flags=re.IGNORECASE,
        )
        center = loadmat(io.BytesIO(source.bytes(base + "_C.mat")))["ONHCentre"]
        source_id = stem.rsplit("-", 1)[0]
        for observer in ("AnsuONH", "Bob", "David", "Lee"):
            annotation_name = base + "_" + observer + ".mat"
            radii = loadmat(io.BytesIO(source.bytes(annotation_name)))["ONHEdge"]
            try:
                disc = decode_onhsd(center, radii, image.shape[:2])
                target, valid = _disc_channels(disc)
                quality = "valid"
            except ValueError as error:
                # Keep the image and provenance; do not turn a missing annotation
                # into an all-background training example.
                target, valid = _disc_channels(np.zeros(image.shape[:2], bool))
                valid[:] = False
                quality = str(error)
            emit(
                stem + "_" + observer,
                "unsplit",
                image,
                target,
                valid=valid,
                source_id=source_id,
                task="disc_cup",
                metadata={
                    "source_image": name,
                    "annotator": observer,
                    "source_annotation": annotation_name,
                    "center_annotation": base + "_C.mat",
                    "quality": quality,
                    **TRAIN_ONLY_PATIENT_POLICY["riga"],
                },
            )


def _riga_closed_pair(ink):
    labels, n = ndi.label(ink, np.ones((3, 3), bool))
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    candidates = []
    bounds = ndi.find_objects(labels)
    for index in np.argsort(areas)[::-1][: min(n, 12)]:
        if areas[index] < 20:
            continue
        region = bounds[index - 1]
        component = labels[region] == index
        filled = ndi.binary_fill_holes(np.pad(component, 1))[1:-1, 1:-1]
        if filled.sum() < component.sum() * 1.5:
            continue
        full = np.zeros(labels.shape, dtype=bool)
        full[region] = filled
        candidates.append(full)
    candidates.sort(key=np.count_nonzero, reverse=True)
    for i, disc in enumerate(candidates):
        for cup in candidates[i + 1 :]:
            outside = (cup & ~disc).sum()
            if outside <= max(2, cup.sum() * 0.01) and cup.sum() < disc.sum() * 0.98:
                return disc, cup & disc, int(outside)
    return None


def recover_riga(original, marked):
    """Recover nested closed outlines; close only bounded JPEG contour gaps.

    The legacy MATLAB loader also uses diamond closing. The smallest successful
    radius is recorded; pairs that remain open or merge are never fabricated.
    """
    if original.shape != marked.shape:
        raise ValueError("RIGA prime and annotated image dimensions differ")
    ink = (original[..., 1].astype(np.int16) - marked[..., 1].astype(np.int16)) > 15
    yy, xx = np.nonzero(ink)
    if not len(yy):
        raise ValueError("RIGA annotation has no two recoverable nested closed contours")
    # Six zero pixels cover the support of dilation+erosion at max radius3.
    # This is an exact nonzero bounding-box crop, never a guessed optic-disc ROI.
    region = (
        slice(max(0, int(yy.min()) - 6), min(ink.shape[0], int(yy.max()) + 7)),
        slice(max(0, int(xx.min()) - 6), min(ink.shape[1], int(xx.max()) + 7)),
    )
    cropped = ink[region]
    for radius in range(4):
        if radius:
            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            repaired = ndi.binary_closing(cropped, structure=(np.abs(x) + np.abs(y) <= radius))
        else:
            repaired = cropped
        pair = _riga_closed_pair(repaired)
        if pair is not None:
            disc_crop, cup_crop, outside = pair
            disc = np.zeros(ink.shape, dtype=bool)
            cup = np.zeros(ink.shape, dtype=bool)
            disc[region], cup[region] = disc_crop, cup_crop
            return (
                disc,
                cup,
                {
                    "method": "green-difference>15;8-connected closed outlines; nested filled areas",
                    "closing_diamond_radius": radius,
                    "cup_pixels_outside_disc": outside,
                },
            )
    raise ValueError("RIGA annotation has no two recoverable nested closed contours")


def riga(source, emit):
    for name in _paths(
        source, r"(?:MESSIDOR|BinRushed[^/]*|MagrabiaMale|MagrabiFemale)/[^/]*prime\.(?:jpg|tif)$"
    ):
        leaf = _base(name)
        base = (
            name.rsplit("/", 1)[0]
            + "/"
            + re.sub(r"prime\.(jpg|tif)$", "", leaf, flags=re.IGNORECASE)
        )
        folder = name.rsplit("/", 2)[-2]
        stem = folder + "_" + re.sub(r"prime\.(jpg|tif)$", "", leaf, flags=re.IGNORECASE)
        image = source.read(name, "RGB")
        for observer in range(1, 7):
            annotation = next(
                (
                    source.find(base + f"-{observer}." + ext, optional=True)
                    for ext in ("jpg", "tif")
                    if source.find(base + f"-{observer}." + ext, optional=True)
                ),
                None,
            )
            if annotation is None:
                raise FileNotFoundError(f"RIGA observer{observer} missing for {name}")
            marked = source.read(annotation, "RGB")
            try:
                disc, cup, quality = recover_riga(image, marked)
                target, valid = _disc_channels(disc, cup)
            except ValueError as error:
                target, valid = _disc_channels(np.zeros(image.shape[:2], bool))
                valid[:] = False
                quality = {"error": str(error), "requires_review": True}
            emit(
                stem + f"_observer{observer}",
                "unsplit",
                image,
                target,
                valid=valid,
                source_id=stem,
                task="disc_cup",
                metadata={
                    "source_image": name,
                    "annotator": observer,
                    "source_annotation": annotation,
                    "quality": quality,
                },
            )


def rimone(source, emit):
    for name in _paths(source, r"(?:Healthy|Glaucoma and suspects)/Stereo Images/[^/]+\.jpg$"):
        stem = _base(name).rsplit(".", 1)[0]
        image = source.read(name, "RGB")
        base = name.rsplit("/Stereo Images/", 1)[0]
        fov = np.ones(image.shape[:2], dtype=bool)
        fov[:, image.shape[1] // 2 :] = False
        group = re.sub(r"-[LR](?:-\d+)?$", "", stem, flags=re.IGNORECASE)
        for observer, folder, suffix in (
            ("average", "Average_masks", "-Avg"),
            ("expert1", "Expert1_masks", "-exp1"),
            ("expert2", "Expert2_masks", "-exp2"),
        ):
            prefix = stem if observer == "average" else stem + "-1"
            disc = _binary(source, f"{base}/{folder}/{prefix}-Disc{suffix}.png")
            cup = _binary(source, f"{base}/{folder}/{prefix}-Cup{suffix}.png")
            target, valid = _disc_channels(disc, cup)
            # Annotation disagreements remain visible, excluded only where the
            # source cup lies outside its corresponding source disc.
            outside = cup & ~disc
            valid[1, outside] = False
            emit(
                stem + "_" + observer,
                "unsplit",
                image,
                target,
                valid=valid,
                fov=fov,
                source_id=group,
                task="disc_cup",
                metadata={
                    "source_image": name,
                    "annotator": observer,
                    "stereo_annotation": "left half only",
                    "annotation_resources": [
                        {
                            "path": f"{base}/{folder}/{prefix}-{label}{suffix}.png",
                            "kind": label.lower(),
                        }
                        for label in ("Disc", "Cup")
                    ],
                    "cup_outside_disc_pixels": int(outside.sum()),
                },
            )


def stare(source, emit):
    for name in _paths(source, r"all-images/im\d+\.ppm$"):
        stem = _base(name).rsplit(".", 1)[0]
        image = source.read(name, "RGB")
        found = False
        for observer in ("ah", "vk"):
            annotation = source.find(f"labels-{observer}/{stem}.{observer}.ppm", optional=True)
            if annotation:
                found = True
                vessel = source.read(annotation) > 0
                emit(
                    stem + "_" + observer,
                    "unsplit",
                    image,
                    vessel[None].astype(np.float32),
                    source_id=stem,
                    metadata={
                        "source_image": name,
                        "annotator": observer,
                        "source_annotation": annotation,
                    },
                )
        if not found:
            emit(
                stem,
                "unsplit",
                image,
                None,
                source_id=stem,
                task="images",
                metadata={"source_image": name, "annotation": "image only; no vessel mask"},
            )


def uoadr(source, emit):
    for name in _paths(source, r"\d+/\d+\.jpg$"):
        stem = _base(name).rsplit(".", 1)[0]
        if name.rsplit("/", 2)[-2] != stem:
            continue
        image = source.read(name, "RGB")
        base = name.rsplit(".", 1)[0]
        vessel_num, disc_num = (2, 1) if int(stem) == 72 else (1, 2)
        vessel = _binary(source, base + f".{vessel_num}.jpg", channel=0)
        disc = _binary(source, base + f".{disc_num}.jpg", channel=0)
        center_marker = _binary(source, base + ".3.jpg", channel=0)
        fov = image.max(axis=-1) > 10
        # Legacy timestamp suppression: ignored pixels, not invented background.
        fov[round(0.97 * image.shape[0]) :, : round(0.25 * image.shape[0])] = False
        emit(
            stem,
            "unsplit",
            image,
            vessel[None].astype(np.float32),
            fov=fov,
            source_id=stem,
            metadata={
                "source_image": name,
                "source_annotation": base + f".{vessel_num}.jpg",
                "annotation72_swap": int(stem) == 72,
                **TRAIN_ONLY_PATIENT_POLICY["uoadr"],
            },
        )
        # The bundled dataset paper identifies .3.jpg as an optic-disc center
        # marker. Historical loaders mistakenly called it a cup annotation.
        target, valid = _disc_channels(disc)
        yy, xx = np.nonzero(center_marker & fov)
        center_xy = [float(xx.mean()), float(yy.mean())] if len(xx) else None
        emit(
            stem + "_disc_cup",
            "unsplit",
            image,
            target,
            valid=valid,
            fov=fov,
            source_id=stem,
            task="disc_cup",
            metadata={
                "source_image": name,
                "cup_annotation": "not supplied; legacy .3.jpg is optic-disc center",
                "optic_disc_center_marker_centroid_xy": center_xy,
                "center_coordinate_frame": "original image; zero based x,y; centroid of supplied marker",
                "annotation_resources": [
                    {"path": base + f".{disc_num}.jpg", "kind": "disc"},
                    {"path": base + ".3.jpg", "kind": "optic_disc_center", "preserve": True},
                ],
                "annotation72_swap": int(stem) == 72,
                **TRAIN_ONLY_PATIENT_POLICY["uoadr"],
            },
        )


def _preserve_annotations(adapter):
    def run(source, emit):
        def annotated_emit(*args, metadata=None, **kwargs):
            metadata = dict(metadata or {})
            paths = [
                metadata[key]
                for key in (
                    "source_annotation",
                    "artery_annotation",
                    "vein_annotation",
                    "center_annotation",
                )
                if key in metadata
            ]
            if paths:
                metadata["annotation_resources"] = metadata.get("annotation_resources", []) + [
                    {"path": path, "kind": "source_annotation"} for path in paths
                ]
            emit(*args, metadata=metadata, **kwargs)

        adapter(source, annotated_emit)

    return run


ADAPTERS = {
    "aria": aria,
    "avrdb": avrdb,
    "g1020": g1020,
    "onhsd": onhsd,
    "riga": riga,
    "rimone": rimone,
    "stare": stare,
    "uoadr": uoadr,
}

ADAPTERS = {name: _preserve_annotations(adapter) for name, adapter in ADAPTERS.items()}
