"""Explicit BIOMISA fundus export-role policy; default keeps labels quarantined.

The measured local export has a consistent file-role inversion: named Cup is
outer disc and named Disc is inner cup. The opt-in mapping is fixed by filename,
never chosen per case by area. Source mask geometry and values are unchanged.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from .additional import TRAIN_ONLY, resource, resources

FIXED_POLICY = "named_cup_is_disc_named_disc_is_cup"


def native_binary(source, path):
    data = source.bytes(path)
    with Image.open(io.BytesIO(data)) as image:
        image_format, mode = image.format, image.mode
        a = np.asarray(image.convert("RGBA"))
    if image_format != "PNG" or not np.isin(a, [0, 255]).all():
        raise ValueError("Expected exact binary PNG content despite BIOMISA .jpg extension")
    if not np.array_equal(a[:, :, 0], a[:, :, 1]) or not np.array_equal(a[:, :, 1], a[:, :, 2]):
        raise ValueError("BIOMISA fundus mask must be binary grayscale, not diagnostic colors")
    mask = (a[:, :, 0] == 255) & (a[:, :, 3] > 0)
    if not mask.any() or mask.all():
        raise ValueError("BIOMISA foreground must be a nonempty proper region")
    return mask, {
        "path": path,
        "kind": "native_fundus_label",
        "sha256": hashlib.sha256(data).hexdigest(),
        "actual_format": image_format,
        "mode": mode,
        "positive_pixels": int(mask.sum()),
    }


def biomisa(source, emit, *, label_policy="quarantine"):
    """Emit45 native labelled-eye candidates under an explicit source-wide policy.

    No physical scaling and no field-angle default are applied. Default outputs
    have all validity false and retain every raw annotation reference. Caller
    must explicitly choose FIXED_POLICY to activate the fixed filename mapping.
    """
    if label_policy not in {"quarantine", FIXED_POLICY}:
        raise ValueError("Select quarantine or the explicit audited fixed filename-role policy")
    index = resources(source, "BIOMISA")
    pattern = r"OD/(Glaucoma|Healthy)/P_\d+/(Left|Right) Eye/Fundus_.+_(Disc|Cup)\.jpg"
    keys = sorted(k for k in index if re.fullmatch(pattern, k))
    if not keys:
        raise FileNotFoundError("BIOMISA native Fundus_*_Disc/Cup label resources not found")
    folders = sorted({str(PurePosixPath(k).parent) for k in keys})
    for folder in folders:
        raw = [
            k
            for k in index
            if k.startswith(folder + "/") and "_Color_" in k and k.lower().endswith(".jpg")
        ]
        if len(raw) != 1:
            raise ValueError(f"BIOMISA raw fundus match must be unambiguous: {folder}, {raw}")
        image_name = resource(source, index[raw[0]])
        image = source.read(image_name, "RGB")
        labels, references = {}, []
        for named_role in ("Cup", "Disc"):
            matches = [
                k
                for k in keys
                if str(PurePosixPath(k).parent) == folder and k.endswith("_" + named_role + ".jpg")
            ]
            if len(matches) > 1:
                raise ValueError(f"BIOMISA duplicated role: {folder}/{named_role}")
            if not matches:
                continue
            name = resource(source, index[matches[0]])
            mask, ref = native_binary(source, name)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"BIOMISA raw/label geometry differs: {name}")
            labels[named_role] = mask
            references.append({**ref, "source_filename_role": named_role})
        target = np.zeros((2, *image.shape[:2]), np.float32)
        valid = np.zeros_like(target, dtype=bool)
        if label_policy == FIXED_POLICY:
            # The roles are fixed globally. Containment validates; it never selects or repairs them.
            if "Cup" in labels:
                target[0] = labels["Cup"]
                valid[0] = True
            if "Disc" in labels:
                target[1] = labels["Disc"]
                valid[1] = True
            if valid[0].any() and valid[1].any() and (labels["Disc"] & ~labels["Cup"]).any():
                raise ValueError(
                    f"BIOMISA fixed role policy violated; no per-image sorting allowed: {folder}"
                )
        parts = folder.split("/")
        stem = "_".join(parts[1:]).replace(" ", "_")
        emit(
            stem,
            "train",
            image,
            target,
            valid=valid,
            task="disc_cup",
            source_id="biomisa:" + ":".join(parts[1:3]),
            metadata={
                **TRAIN_ONLY,
                "source_image": image_name,
                "source_split": "unsplit",
                "source_folder": folder,
                "source_group_basis": "source cohort/P_N directory; paired-eye grouping, cross-cohort identities not established",
                "annotation_resources": references,
                "annotation_policy": label_policy,
                "source_export_roles": {"named_Cup": "disc", "named_Disc": "cup"}
                if label_policy == FIXED_POLICY
                else None,
                "annotation_status": "explicit_review_required_source_wide_role_correction"
                if label_policy == FIXED_POLICY
                else "quarantined_filename_role_inversion",
                "role_sorting_by_area": False,
                "mask_values_or_geometry_modified": False,
                "cup_missing": "Disc" not in labels,
                "proposed_field_angle": {
                    "degrees": 45,
                    "status": "inferred_device_spec_only_not_measured_acquisition",
                    "dataset_paper": "https://doi.org/10.1109/ICoDT255437.2022.9787482",
                    "manufacturer_spec": "https://topconhealthcare.com/wp-content/uploads/sites/8/2020/07/3DOCT-2000.pdf",
                },
            },
        )
