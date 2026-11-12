"""Experimental probability-only grid bridge. No images, GT, models or FOV masking."""

import math
import re
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
from skimage.transform import resize

BACKEND = {"numpy": "2.5.4", "scipy": "1.18.1", "scikit-image": "0.26.0"}
CONVENTION = {
    "version": 2,
    "coordinates": "source=(target_index+0.5)*source_size/target_size-0.5 per axis",
    "interpolation": "bilinear/order1 on FP32 joint V/A/V/D/C planes",
    "edge": "nearest edge extension (clamp), skimage mode=edge",
    "anti_aliasing": False,
    "clip": False,
    "TTA": "already inverse-aligned in source prepared frame before bridge",
    "FOV": "no source or target FOV masking here; unchanged target25 scorer owns support",
    "identity": "exact25-to25 geometry returns independent bit-identical copy",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Grid:
    photo_id: str
    split_group_id: str
    split: str
    canonical_owner: tuple[str, str]
    original_shape: tuple[int, int]
    crop_position: tuple[int, int, int, int]
    prepared_shape: tuple[int, int]
    field_of_view_degrees: float
    pixels_per_degree: float
    preprocessing_profile: str
    geometry_policy_sha256: str
    manifest_sha256: str
    content_sha256: str
    parent_geometry_policy_sha256: str | None = None

    def validate(self):
        require(self.split in {"train", "val"}, "TRAIN/VAL only; no TEST bridge")
        for value in (self.photo_id, self.split_group_id):
            require(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
                "Photo/group identity required",
            )
        for value in (self.geometry_policy_sha256, self.manifest_sha256, self.content_sha256):
            require(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                "Exact policy/manifest/content pins required",
            )
        if self.pixels_per_degree == 35:
            require(
                isinstance(self.parent_geometry_policy_sha256, str)
                and re.fullmatch(r"[0-9a-f]{64}", self.parent_geometry_policy_sha256),
                "Source35 parent geometry policy pin required",
            )
        else:
            require(
                self.parent_geometry_policy_sha256 is None,
                "Historical25 has no new35 parent policy",
            )
        require(
            len(self.canonical_owner) == 2
            and all(isinstance(x, str) and x for x in self.canonical_owner),
            "Canonical owner required",
        )
        require(
            len(self.original_shape) == len(self.prepared_shape) == 2
            and all(type(n) is int and n > 0 for n in (*self.original_shape, *self.prepared_shape)),
            "Positive integer grid shapes required",
        )
        require(
            len(self.crop_position) == 4 and all(type(n) is int for n in self.crop_position),
            "Inclusive integer crop required",
        )
        y0, y1, x0, x1 = self.crop_position
        h, w = self.original_shape
        require(0 <= y0 <= y1 < h and 0 <= x0 <= x1 < w, "Crop escaped native frame")
        require(
            type(self.field_of_view_degrees) in (int, float)
            and math.isfinite(self.field_of_view_degrees)
            and self.field_of_view_degrees > 0,
            "Positive finite camera angle required",
        )
        require(
            type(self.pixels_per_degree) in (int, float) and self.pixels_per_degree in (25, 35),
            "Only explicit25/35 grid supported",
        )
        require(
            self.preprocessing_profile == f"legacy{self.pixels_per_degree:g}_raw",
            "RAW physical profile differs",
        )
        scale = self.pixels_per_degree * self.field_of_view_degrees / max(h, w)
        expected = tuple(max(1, round(n * scale)) for n in (y1 - y0 + 1, x1 - x0 + 1))
        require(
            self.prepared_shape == expected, "Prepared shape differs from native anchor/crop/ppd"
        )


def _resize_plane(plane, shape):
    return resize(
        plane, shape, order=1, mode="edge", preserve_range=True, anti_aliasing=False, clip=False
    ).astype(np.float32, copy=False)


def bridge(probability, source: Grid, target: Grid):
    """Map already inverse-TTA conditional probabilities onto fixed25 geometry.

    Identity binds native RGB photo/group/owner/crop/angle and equal25 policy. Different
    source/target manifest and content pins are retained, never equated/retagged.
    FOV arrays deliberately cannot be passed. Caller must use the original25 GT
    and valid/FOV in its unchanged scorer, with exact observer/source traces.
    """
    source.validate()
    target.validate()
    require(target.pixels_per_degree == 25, "Measurement target must be25")
    for field in (
        "photo_id",
        "split_group_id",
        "split",
        "canonical_owner",
        "original_shape",
        "crop_position",
        "field_of_view_degrees",
    ):
        require(
            getattr(source, field) == getattr(target, field),
            "Native geometry/lineage mismatch: " + field,
        )
    if source.pixels_per_degree == 35:
        require(
            source.geometry_policy_sha256 != target.geometry_policy_sha256,
            "Source35 requires an actual NEW geometry policy, never reused25 policy",
        )
        require(
            source.parent_geometry_policy_sha256 == target.geometry_policy_sha256,
            "Source35 parent geometry policy differs from target25 policy",
        )
    else:
        require(
            source.geometry_policy_sha256 == target.geometry_policy_sha256,
            "Identity25 actual geometry policy differs",
        )
    p = np.asarray(probability)
    require(
        p.dtype == np.float32 and p.shape == (5, *source.prepared_shape),
        "Five FP32 conditional channels on source grid required",
    )
    require(
        np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
        "Finite unit probabilities required",
    )
    require(np.allclose(p[1] + p[2], 1, rtol=0, atol=2e-6), "Conditional A/V must sum to one")
    require(
        {name: version(name) for name in BACKEND} == BACKEND, "Pinned interpolation backend differs"
    )
    if source.pixels_per_degree == 25:
        require(source.prepared_shape == target.prepared_shape, "Identity grid differs")
        return p.copy()  # No joint/conditional round trip, including signed zeros.
    joint = p.copy()
    joint[1:3] *= joint[0:1]
    out = np.stack([_resize_plane(plane, target.prepared_shape) for plane in joint])
    denominator = out[1:3].sum(axis=0, keepdims=True)
    out[1:3] = np.divide(
        out[1:3], denominator, out=np.full_like(out[1:3], 0.5), where=denominator > 0
    )
    return out
