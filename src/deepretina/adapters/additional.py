"""Native annotations found by archive audit beyond the historical loader set.

No physical field angle is invented. The functions emit native RGB and labels;
physical legacy25 preparation requires an explicit reviewed calibration policy.
"""

from __future__ import annotations

import io
import re
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
from skimage.draw import polygon

TRAIN_ONLY = {
    "split_policy": "train_only_without_patient_crosswalk",
    "split_policy_evidence": {
        "reason": "No reliable patient crosswalk in local export; keep all new unknown-patient samples out of validation/test"
    },
    "physical_calibration": {
        "status": "requires_explicit_reviewed_angle",
        "field_of_view_degrees": None,
    },
}


def resources(source, dataset):
    """Logical resource -> transport aliases, without opening their contents."""
    grouped = {}
    for name in source.names:
        logical = (
            name[len(dataset) + 1 :] if name.lower().startswith(dataset.lower() + "/") else name
        )
        grouped.setdefault(logical, []).append(name)
    return {k: sorted(v, key=lambda n: (len(n), n)) for k, v in grouped.items()}


def resource(source, aliases):
    """Reject conflicting duplicated transport members for actually used files."""
    chosen = aliases[0]
    if len(aliases) > 1:
        data = source.bytes(chosen)
        if any(source.bytes(name) != data for name in aliases[1:]):
            raise ValueError(f"Conflicting annotation/image transport aliases: {aliases}")
    return chosen


def disc_from_points(text, shape):
    """Closed piecewise-linear polygon at one-based X/Y pixel centers."""
    points = np.loadtxt(io.StringIO(text), delimiter=",")
    if points.shape != (36, 2) or not np.isfinite(points).all():
        raise ValueError("DRIONS requires exactly36 finite X,Y contour points")
    if (points < 1).any() or (points[:, 0] > shape[1]).any() or (points[:, 1] > shape[0]).any():
        raise ValueError("DRIONS one-based contour lies outside native image")
    mask = np.zeros(shape, bool)
    rows, cols = polygon(points[:, 1] - 1, points[:, 0] - 1, shape)
    mask[rows, cols] = True
    if not mask.any():
        raise ValueError("DRIONS disc polygon is empty")
    return mask


def drions(source, emit):
    index = resources(source, "DRIONS-DB")
    images = sorted(k for k in index if re.fullmatch(r"images/image_\d{3}\.jpg", k, re.IGNORECASE))
    if not images:
        raise FileNotFoundError("DRIONS images/image_NNN.jpg not found")
    for logical in images:
        stem = PurePosixPath(logical).stem
        number = stem.rsplit("_", 1)[1]
        image_name = resource(source, index[logical])
        image = source.read(image_name, "RGB")
        for observer in (1, 2):
            key = f"experts_anotation/anotExpert{observer}_{number}.txt"
            if key not in index:
                raise FileNotFoundError(key)
            annotation = resource(source, index[key])
            disc = disc_from_points(source.text(annotation), image.shape[:2])
            target = np.stack([disc, np.zeros_like(disc)]).astype(np.float32)
            valid = np.stack([np.ones_like(disc), np.zeros_like(disc)])
            emit(
                f"{stem}_expert{observer}",
                "train",
                image,
                target,
                valid=valid,
                source_id=f"drions:{number}",
                task="disc_cup",
                metadata={
                    **TRAIN_ONLY,
                    "source_image": image_name,
                    "source_annotation": annotation,
                    "source_split": "unsplit",
                    "annotator": f"expert{observer}",
                    "annotation_resources": [{"path": annotation, "kind": "source_contour"}],
                    "coordinates": "one-based X/Y centers from bundled ShowDRIONS_DB.m",
                    "rasterization": "closed piecewise-linear polygon at integer pixel centers, no curve interpolation or observer averaging",
                    "cup_annotation": "absent; validity false",
                    "patient_identity_known": False,
                },
            )


def decode_gamma_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 128, 255]).all():
        raise ValueError("GAMMA expects native grayscale labels0=cup,128=rim,255=background")
    target = np.stack([mask != 255, mask == 0]).astype(np.float32)
    if not target[0].any() or not target[1].any() or not (mask == 255).any():
        raise ValueError("GAMMA disc/cup/background labels must all be present")
    return target


def gamma(source, emit):
    index = resources(source, "GAMMA")
    mask_keys = sorted(
        k
        for k in index
        if re.fullmatch(r"GAMMA_training data/training_data/Disc_Cup_Mask/\d{4}\.png", k)
    )
    if not mask_keys:
        raise FileNotFoundError("GAMMA official training Disc_Cup_Mask resources missing")
    for key in mask_keys:
        stem = PurePosixPath(key).stem
        image_key = f"GAMMA_training data/training_data/multi-modality_images/{stem}/{stem}.jpg"
        if image_key not in index:
            raise FileNotFoundError(image_key)
        image_name = resource(source, index[image_key])
        annotation = resource(source, index[key])
        image = source.read(image_name, "RGB")
        target = decode_gamma_mask(source.read(annotation, "L"))
        if target.shape[1:] != image.shape[:2]:
            raise ValueError(f"GAMMA mask/RGB geometry differs: {stem}")
        emit(
            stem,
            "train",
            image,
            target,
            source_id=f"gamma:{stem}",
            task="disc_cup",
            metadata={
                **TRAIN_ONLY,
                "source_image": image_name,
                "source_annotation": annotation,
                "source_split": "train",
                "official_split": "train",
                "annotation_resources": [{"path": annotation, "kind": "source_segmentation"}],
                "label_encoding": {"0": "cup", "128": "disc rim", "255": "background"},
                "patient_identity_known": False,
                "excluded_modalities": ["OCT slices", "Disc_Cup_Fovea_check overlays"],
            },
        )


def idrid_disc(source, emit, *, splits=("train", "test")):
    """Native OD masks from A.Segmentation, separate from localization points.

    `splits=("train",)` is used by the v2 proposal audit: it inventories official
    TEST resource names but never opens their labels or images.
    """
    if not set(splits) <= {"train", "test"}:
        raise ValueError("IDRiD dense splits must be train/test")
    index = resources(source, "IDRiD")
    pattern = r"A\. Segmentation/2\. All Segmentation Groundtruths/([ab])\. (Training|Testing) Set/5\. Optic Disc/(IDRiD_\d+)_OD\.tif"
    keys = sorted(k for k in index if re.fullmatch(pattern, k))
    if not keys:
        raise FileNotFoundError("IDRiD A.Segmentation optic-disc GT missing")
    for key in keys:
        match = re.fullmatch(pattern, key)
        split = "train" if match[1] == "a" else "test"
        if split not in splits:
            continue
        stem = match[3]
        folder = "a. Training Set" if split == "train" else "b. Testing Set"
        image_key = f"A. Segmentation/1. Original Images/{folder}/{stem}.jpg"
        if image_key not in index:
            raise FileNotFoundError(image_key)
        image_name = resource(source, index[image_key])
        annotation = resource(source, index[key])
        image = source.read(image_name, "RGB")
        # Native TIFF palette index1 is red, whose grayscale conversion is76.
        # Preserve the binary indices rather than thresholding rendered colors.
        with Image.open(io.BytesIO(source.bytes(annotation))) as native:
            label = np.asarray(native).copy()
            label_mode = native.mode
        binary = np.isin(label, [0, 1]).all() or np.isin(label, [0, 255]).all()
        if label.shape != image.shape[:2] or not binary:
            raise ValueError(f"IDRiD requires native matching binary OD mask: {stem}")
        disc = label > 0
        if not disc.any():
            raise ValueError(f"IDRiD OD annotation is empty: {stem}")
        target = np.stack([disc, np.zeros_like(disc)]).astype(np.float32)
        valid = np.stack([np.ones_like(disc), np.zeros_like(disc)])
        emit(
            stem,
            split,
            image,
            target,
            valid=valid,
            source_id=f"idrid:{stem.lower()}",
            task="disc_cup",
            metadata={
                "source_image": image_name,
                "source_annotation": annotation,
                "source_split": split,
                "official_split": split,
                "annotation_resources": [{"path": annotation, "kind": "source_segmentation"}],
                "scope": "A.Segmentation OD raster; existing idrid selector retains C.Localization points/images",
                "label_encoding": "native0=background, positive binary index=disc; cup unannotated",
                "annotation_native_mode": label_mode,
                "annotation_native_values": np.unique(label).tolist(),
                "patient_identity_known": False,
                "physical_calibration": {
                    "status": "published_dataset_acquisition_angle",
                    "field_of_view_degrees": 50.0,
                    "camera": "Kowa VX-10 alpha",
                    "source": "https://www.mdpi.com/2306-5729/3/3/25",
                },
            },
        )


ADAPTERS = {"drions": drions, "gamma": gamma, "idrid_disc": idrid_disc}
