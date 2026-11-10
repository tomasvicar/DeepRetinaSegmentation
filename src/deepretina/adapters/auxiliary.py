"""Legacy image collections and auxiliary labels, without invented segmentations.

Annotation resources refer to virtual paths in the input Source, in original image
coordinates. Only Erlangen manual vessel masks create supervised records.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
from scipy import ndimage

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _stem(path):
    return PurePosixPath(path).stem


def _unique(paths, key=None):
    """ZIP exports may include the same tree under two different prefixes."""
    selected = {}
    for path in sorted(paths, key=lambda p: (len(p), p)):
        selected.setdefault(key(path) if key else PurePosixPath(path).name.casefold(), path)
    return sorted(selected.values())


def _pick(source, predicate, description):
    paths = [
        p
        for p in source.names
        if predicate(p) and PurePosixPath(p).suffix.lower() in _IMAGE_EXTENSIONS
    ]
    paths = _unique(paths)
    if not paths:
        raise ValueError(
            f"{description}: no matching original images; check the dataset root/layout"
        )
    return paths


def _refs(paths, kind):
    return [
        {"path": p, "kind": kind, "coordinate_space": "original_image"}
        for p in _unique(paths, key=lambda p: "/".join(PurePosixPath(p).parts[-2:]).casefold())
    ]


def _emit_image(
    source, emit, path, *, dataset, split="unsplit", group=None, metadata=None, fov=None
):
    image = source.read(path, mode="RGB")
    details = {
        "original_resource": path,
        "original_shape": list(image.shape[:2]),
        **(metadata or {}),
    }
    emit(
        _stem(path),
        split,
        image,
        None,
        task="images",
        source_id=group or f"{dataset}:{_stem(path).lower()}",
        metadata=details,
        fov=fov,
    )


def _diaretdb(source, emit, version):
    folder = "diaretdb0_fundus_images" if version == 0 else "ddb1_fundusimages"
    paths = _pick(source, lambda p: f"/{folder}/" in "/" + p.lower(), f"DiaRetDB{version}")
    for path in paths:
        stem = _stem(path)
        number = int(re.search(r"(\d+)$", stem).group(1))
        diagnosis = (
            ("healthy" if number > 110 else "dr")
            if version == 0
            else ("healthy" if number in {49, 57, 60, 62, 72} else "npdr")
        )
        annotations = [
            p
            for p in source.names
            if "groundtruth" in p.lower() and _stem(p).casefold() == stem.casefold()
        ]
        _emit_image(
            source,
            emit,
            path,
            dataset=f"diaretdb{version}",
            metadata={
                "legacy_diagnosis": diagnosis,
                "annotation_resources": _refs(annotations, "lesion_groundtruth"),
                "label_note": "Lesion annotations are auxiliary; no vessel/disc/cup target is inferred.",
            },
        )


def diaretdb0(source, emit):
    _diaretdb(source, emit, 0)


def diaretdb1(source, emit):
    _diaretdb(source, emit, 1)


def eophtha(source, emit):
    paths = _pick(
        source,
        lambda p: (
            any(f"/{d}/" in "/" + p.lower() for d in ("ex", "ma", "healthy"))
            and "annotation" not in p.lower()
        ),
        "E-Ophtha",
    )
    for path in paths:
        parts = PurePosixPath(path).parts
        diagnosis = next(
            (x.lower() for x in parts if x.lower() in {"ex", "ma", "healthy"}), "unknown"
        )
        annotations = [
            p
            for p in source.names
            if "annotation" in p.lower() and _stem(p).casefold().startswith(_stem(path).casefold())
        ]
        _emit_image(
            source,
            emit,
            path,
            dataset="eophtha",
            group=f"eophtha:{parts[-2]}",
            metadata={
                "lesion_group": diagnosis,
                "annotation_resources": _refs(annotations, "lesion_mask"),
            },
        )


def _resize_mask(mask, shape):
    if mask.shape != shape:
        mask = (
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (shape[1], shape[0]), Image.Resampling.NEAREST
                )
            )
            > 0
        )
    return mask


def erlangen(source, emit):
    paths = _pick(
        source,
        lambda p: "/fundusimages/" in "/" + p.lower() and "fov" not in _stem(p).lower(),
        "Erlangen",
    )
    masks = _unique(
        [
            p
            for p in source.names
            if "/manually_corrected/" in "/" + p.lower()
            and p.lower().endswith((".tif", ".tiff", ".png"))
        ]
    )
    for path in paths:
        stem = _stem(path)
        base = re.sub(r"_(reg|registered.*)$", "", stem, flags=re.IGNORECASE)
        exact = [p for p in masks if _stem(p).lower().startswith(base.lower() + "_registered")]
        if not exact:
            # Original MATLAB distributed images collapse the visit component.
            exact = [
                p
                for p in masks
                if re.sub(
                    r"_(\d\d)_([LR])_registered.*", r"_\2", _stem(p), flags=re.IGNORECASE
                ).lower()
                == base.lower()
            ]
        if len(exact) > 1:
            raise ValueError(f"Erlangen: ambiguous manual vessel mask for {path}: {exact}")
        image = source.read(path, mode="RGB")
        subject = re.match(r"(Study_\d+_\d+)", stem, re.IGNORECASE)
        group = f"erlangen:{subject.group(1).lower() if subject else base.lower()}"
        metadata = {
            "original_resource": path,
            "annotation_resources": _refs(exact, "manual_vessel_mask"),
            "mask_channel": "green",
            "mask_resampling": "nearest",
        }
        if exact:
            green = source.read(exact[0], mode="RGB")[..., 1]
            mask = _resize_mask(green > green.max() / 2, image.shape[:2])
            emit(
                stem,
                "unsplit",
                image,
                mask[None].astype(np.float32),
                valid=np.ones((1, *mask.shape), bool),
                fov=np.ones(mask.shape, bool),
                source_id=group,
                metadata=metadata,
            )
        else:
            _emit_image(
                source,
                emit,
                path,
                dataset="erlangen",
                group=group,
                metadata=metadata,
                fov=np.ones(image.shape[:2], bool),
            )


def fire(source, emit):
    paths = _pick(
        source,
        lambda p: (
            "/images/" in "/" + p.lower()
            and re.fullmatch(r"[APS]\d+_[12]", _stem(p), re.IGNORECASE)
        ),
        "FIRE",
    )
    masks = _unique([p for p in source.names if p.lower().endswith("/masks/mask.png")])
    for path in paths:
        pair = _stem(path).rsplit("_", 1)[0]
        annotations = [
            p
            for p in source.names
            if PurePosixPath(p).name.lower() == f"control_points_{pair}_1_2.txt".lower()
        ]
        points = []
        if annotations:
            points = np.loadtxt(StringIO(source.text(annotations[0])), ndmin=2).tolist()
        fov = source.read(masks[0], mode="L") > 0 if masks else None
        _emit_image(
            source,
            emit,
            path,
            dataset="fire",
            group=f"fire:{pair.lower()}",
            fov=fov,
            metadata={
                "registration_pair": pair,
                "control_points_xy_image1_xy_image2": points,
                "annotation_resources": _refs(annotations, "registration_control_points"),
            },
        )


def heimed(source, emit):
    paths = _pick(
        source,
        lambda p: "/dmed/" in "/" + p.lower() and p.lower().endswith((".jpg", ".jpeg")),
        "HEI-MED",
    )
    for path in paths:
        stem = _stem(path)
        annotations = [
            p
            for p in source.names
            if PurePosixPath(p).parent == PurePosixPath(path).parent
            and (
                PurePosixPath(p).name.startswith(stem + ".")
                or PurePosixPath(p).name.startswith(stem + "_")
            )
            and p != path
        ]
        _emit_image(
            source,
            emit,
            path,
            dataset="heimed",
            metadata={
                "annotation_resources": _refs(annotations, "auxiliary_lesion_or_derived_resource"),
                "label_note": "GND/meta/map and derived vessel PNGs retained; no verified manual complete vessel target.",
            },
        )


def idrid(source, emit):
    paths = _pick(
        source,
        lambda p: "/c. localization/1. original images/" in "/" + p.lower(),
        "IDRiD localization",
    )
    csvs = _unique(
        [
            p
            for p in source.names
            if "/c. localization/2. groundtruths/" in "/" + p.lower() and p.lower().endswith(".csv")
        ]
    )
    locations = {}
    for path in csvs:
        kind = "optic_disc_center_xy" if "od_center" in path.lower() else "fovea_center_xy"
        for row in csv.reader(StringIO(source.text(path))):
            if len(row) >= 3 and re.fullmatch(r"IDRiD_\d+", row[0].strip(), re.IGNORECASE):
                try:
                    locations.setdefault(row[0].strip().lower(), {})[kind] = [
                        float(row[1]),
                        float(row[2]),
                    ]
                except ValueError as error:
                    raise ValueError(
                        f"IDRiD invalid coordinate row in {path}: {row[:3]}"
                    ) from error
    for path in paths:
        split = "test" if "testing set" in path.lower() else "train"
        references = [p for p in csvs if ("testing" in p.lower()) == (split == "test")]
        _emit_image(
            source,
            emit,
            path,
            dataset="idrid",
            split=split,
            metadata={
                "coordinates_original_xy": locations.get(_stem(path).lower(), {}),
                "annotation_resources": _refs(references, "localization_points"),
                "label_note": "Center coordinates do not define a disc or cup segmentation.",
            },
        )


def inspireavr(source, emit):
    paths = _pick(source, lambda p: "/org/" in "/" + p.lower(), "INSPIRE-AVR")
    gts = _unique([p for p in source.names if p.lower().endswith("/gt/gt.txt")])
    if not gts:
        raise ValueError("INSPIRE-AVR requires GT/GT.txt observer AVR measurements")
    measurements = {}
    for line in source.text(gts[0]).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].lower().endswith(".jpg"):
            measurements[parts[0].lower()] = [float(parts[1]), float(parts[2])]
    for path in paths:
        name = PurePosixPath(path).name.lower()
        if name not in measurements:
            raise ValueError(f"INSPIRE-AVR missing measurement for {path}")
        _emit_image(
            source,
            emit,
            path,
            dataset="inspireavr",
            metadata={
                "avr_observer_1_2": measurements[name],
                "annotation_resources": _refs(gts, "avr_measurements"),
                "label_note": "AVR scalar measurements are not artery/vein pixel masks.",
            },
        )


def messidor(source, emit):
    import xlrd

    paths = _pick(source, lambda p: p.lower().endswith((".tif", ".tiff")), "MESSIDOR")
    tables = _unique(
        [
            p
            for p in source.names
            if PurePosixPath(p).name.lower().startswith("annotation_base")
            and p.lower().endswith(".xls")
        ]
    )
    if not tables:
        raise ValueError("MESSIDOR requires Annotation_Base*.xls diagnosis tables")
    diagnoses = {}
    for path in tables:
        workbook = xlrd.open_workbook(file_contents=source.bytes(path))
        sheet = workbook.sheet_by_index(0)
        for row_index in range(1, sheet.nrows):
            row = sheet.row_values(row_index)
            if len(row) < 4 or not str(row[0]).lower().endswith((".tif", ".tiff")):
                continue
            name = str(row[0]).lower()
            value = {
                "retinopathy_grade": int(row[2]),
                "macular_edema_risk": int(row[3]),
                "resource": path,
            }
            if name in diagnoses and diagnoses[name] != value:
                raise ValueError(f"MESSIDOR conflicting diagnosis for {name}")
            diagnoses[name] = value
    for path in paths:
        name = PurePosixPath(path).name.lower()
        if name not in diagnoses:
            raise ValueError(f"MESSIDOR no diagnosis table row for {path}")
        diagnosis = diagnoses[name]
        _emit_image(
            source,
            emit,
            path,
            dataset="messidor",
            metadata={
                "retinopathy_grade": diagnosis["retinopathy_grade"],
                "macular_edema_risk": diagnosis["macular_edema_risk"],
                "annotation_resources": _refs([diagnosis["resource"]], "disease_grading_table"),
            },
        )


def rcrgbma(source, emit):
    paths = _pick(source, lambda p: "/original/" in "/" + p.lower(), "RC-RGB-MA")
    for path in paths:
        match = re.search(r"\((\d+)\)", _stem(path))
        if match is None:
            raise ValueError(f"RC-RGB-MA original image lacks numeric ID: {path}")
        number = match.group(1)
        masks = [
            p
            for p in source.names
            if "/ma masks/" in "/" + p.lower() and re.search(r"\(" + number + r"\)\.[^.]+$", p)
        ]
        _emit_image(
            source,
            emit,
            path,
            dataset="rcrgbma",
            metadata={
                "annotation_resources": _refs(masks, "microaneurysm_expert_mask"),
                "label_note": "Microaneurysm masks retained as lesion resources; not vessel masks.",
            },
        )


def review(source, emit):
    paths = _pick(
        source,
        lambda p: any(
            "/" + group + "/" in "/" + p.lower() for group in ("hris", "vdis", "clris", "kpis")
        ),
        "REVIEW",
    )
    for path in paths:
        subset = PurePosixPath(path).parent.name.upper()
        references = [
            p
            for p in source.names
            if "observer marking for " + subset.lower() in p.lower() and p.lower().endswith(".txt")
        ]
        image_number = int(re.search(r"(\d+)$", _stem(path)).group(1))
        observer_rows = []
        for reference in _unique(references):
            for line in source.text(reference).splitlines():
                row = line.split()
                if len(row) >= 15 and row[0].isdigit() and int(row[1]) == image_number:
                    observer_rows.append(
                        {
                            "serial": int(row[0]),
                            "image": int(row[1]),
                            "segment": int(row[2]),
                            "observer_right_left_xy": np.asarray(row[3:15], dtype=float)
                            .reshape(3, 4)
                            .tolist(),
                            "resource": reference,
                        }
                    )
        _emit_image(
            source,
            emit,
            path,
            dataset="review",
            metadata={
                "subset": subset,
                "vessel_edge_measurements_original": observer_rows,
                "annotation_resources": _refs(references, "sparse_vessel_width_edges"),
                "label_note": "Sparse observer vessel edges do not define a complete binary vessel map.",
            },
        )


def ridb(source, emit):
    paths = _pick(source, lambda p: p.lower().endswith((".jpg", ".jpeg")), "RIDB")
    for path in paths:
        identity = _stem(path).split("_")[0]
        _emit_image(
            source,
            emit,
            path,
            dataset="ridb",
            group=f"ridb:{identity.lower()}",
            metadata={
                "identity_group": identity,
                "label_note": "Recognition identity grouping; no segmentation labels.",
            },
        )


def roc(source, emit):
    paths = _pick(
        source, lambda p: re.fullmatch(r"image\d+_(training|test)", _stem(p), re.IGNORECASE), "ROC"
    )
    xmls = _unique(
        [p for p in source.names if p.lower().endswith("annotations-consensus-ma-only.xml")]
    )
    annotations = {}
    for path in xmls:
        tree = ET.fromstring(source.bytes(path))
        for element in tree.findall(".//annotations-per-image"):
            rows = []
            for annotation in element.findall("annotation"):
                mark = annotation.find("mark")
                if mark is None:
                    raise ValueError(f"ROC malformed annotation missing mark in {path}")
                rows.append(
                    {
                        "x": float(mark.attrib["x"]),
                        "y": float(mark.attrib["y"]),
                        "radius": float(mark.findtext("radius")),
                        "lesion": annotation.findtext("lesion"),
                        "probability": float(annotation.findtext("probability")),
                    }
                )
            annotations[element.attrib["imagename"].lower()] = rows
    for path in paths:
        split = "test" if _stem(path).lower().endswith("_test") else "train"
        _emit_image(
            source,
            emit,
            path,
            dataset="roc",
            split=split,
            metadata={
                "lesion_marks_original_xy": annotations.get(PurePosixPath(path).name.lower(), []),
                "annotation_available": split == "train" and bool(xmls),
                "annotation_resources": _refs(
                    xmls if split == "train" else [], "microaneurysm_point_annotations"
                ),
            },
        )


def vicavr(source, emit):
    paths = _pick(source, lambda p: "/images/original/" in "/" + p.lower(), "VICAVR")
    references = _unique(
        [
            p
            for p in source.names
            if re.fullmatch(r"expert[123]\.txt", PurePosixPath(p).name, re.IGNORECASE)
        ]
    )
    measurements = {}
    for path in references:
        expert = _stem(path).lower()
        for row in csv.reader(StringIO(source.text(path))):
            if len(row) != 6 or not row[0].strip().isdigit():
                continue
            number = int(row[0])
            vessel_class = int(row[5])
            measurements.setdefault(number, []).append(
                {
                    "expert": expert,
                    "vessel": int(row[1]),
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "caliber": float(row[4]),
                    "class": vessel_class,
                }
            )
    for path in paths:
        number = int(re.search(r"(\d+)$", _stem(path)).group(1))
        _emit_image(
            source,
            emit,
            path,
            dataset="vicavr",
            metadata={
                "point_measurements_original_xy": measurements.get(number, []),
                "class_encoding": {
                    "0": "unclassified",
                    "1": "artery",
                    "2": "vein",
                    "3": "undocumented_source_value",
                },
                "annotation_resources": _refs(references, "sparse_vessel_caliber_and_class"),
                "label_note": "Point classes and calibers remain sparse measurements; no dense AV masks are inferred.",
            },
        )


def ubmi(source, emit):
    paths = _pick(
        source,
        lambda p: (
            re.fullmatch(r"Gacr.*[LR]", _stem(p), re.IGNORECASE)
            and PurePosixPath(p).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        "UBMI: provide original Gacr* folders with Gacr*L.jpg/R.jpg and optional ImageAnalysis/VesselsSeg/*whole.png",
    )
    for path in paths:
        image = source.read(path, mode="RGB")
        stem = _stem(path)
        folder = str(PurePosixPath(path).parent)
        prefix = "" if folder == "." else folder + "/"
        masks = _unique(
            [
                p
                for p in source.names
                if p.startswith(prefix)
                and "/imageanalysis/vesselsseg/" in "/" + p.lower()
                and p.lower().endswith("whole.png")
            ]
        )
        if len(masks) > 1:
            raise ValueError(f"UBMI ambiguous whole vessel masks for {path}: {masks}")
        gray = image.astype(np.float32) @ np.asarray([0.2989, 0.5870, 0.1141], np.float32)
        # Historical special-case hysteresis thresholds (intensity range 0..255).
        low, high = (
            (2, 5)
            if "01_017_01_L" in stem
            else (1, 3)
            if "01_015_01_L" in stem
            else (6, 10)
            if "01_014_01_L" in stem
            else (3, 6)
        )
        fov = ndimage.binary_propagation(gray >= high, mask=gray >= low)
        fov = ndimage.binary_fill_holes(fov)
        labels, count = ndimage.label(fov)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            fov = labels == sizes.argmax()
        metadata = {
            "original_resource": path,
            "annotation_resources": _refs(masks, "existing_vessel_segmentation_for_inference"),
            "label_note": "Existing segmentations lack verified manual provenance and are not training ground truth.",
            "fov_method": "grayscale hysteresis, largest component, filled holes; no undocumented contrast transformation",
            "original_shape": list(image.shape[:2]),
        }
        subject = re.match(r"(Gacr_\d+_\d+)", stem, re.IGNORECASE)
        group = (
            "ubmi:"
            + (
                subject.group(1) if subject else re.sub(r"_[LR]$", "", stem, flags=re.IGNORECASE)
            ).lower()
        )
        if masks:
            mask = source.read(masks[0], mode="L") > 0
            if mask.shape != image.shape[:2]:
                raise ValueError(f"UBMI vessel mask geometry differs from image: {path}")
        _emit_image(source, emit, path, dataset="ubmi", group=group, fov=fov, metadata=metadata)


ADAPTERS = {
    "diaretdb0": diaretdb0,
    "diaretdb1": diaretdb1,
    "eophtha": eophtha,
    "erlangen": erlangen,
    "fire": fire,
    "heimed": heimed,
    "idrid": idrid,
    "inspireavr": inspireavr,
    "messidor": messidor,
    "rcrgbma": rcrgbma,
    "review": review,
    "ridb": ridb,
    "roc": roc,
    "vicavr": vicavr,
    "ubmi": ubmi,
}
