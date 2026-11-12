"""Experimental native RAW35 producer with frozen TRAIN/VAL identities.

Planning reads JSON only. Execution requires a separate explicit bounded profile;
no trainer admission changes are made by this module.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import platform
import re
import resource
import shutil
import time
import zipfile
from collections import Counter, OrderedDict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters.additional import ADAPTERS as ADDITIONAL
from .adapters.auxiliary import erlangen
from .adapters.biomisa import FIXED_POLICY, biomisa
from .adapters.core import decode_core
from .adapters.segmentation import ADAPTERS
from .geometry_policy import geometry_rule, validate_geometry_policy
from .integrity import FINGERPRINT_SCHEMA, prepared_content_fingerprint
from .legacy.preprocessing import OFFICIAL_FOV_DATASETS, prepare_single_net
from .prepare import ARCHIVES, CORE_DATASETS, decoded_photo_id
from .sources import Source, canonical_path

MANIFEST_SHA = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
PHOTOS_SHA = "765709a1ddc08035565ebf3d54d414dd1f53a9882a5cfa9c0fd03811b8e48a4d"
PREPARATION_SHA = "e7899af17ac94153a64752a25b52fba6001da361ae7a2893750d4bf2493389e2"
CONTENT_SHA = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
POLICY_SHA = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
EXTRA = "drive-download-20261019T122612Z-1-001.zip"
DATASETS = frozenset(CORE_DATASETS | set(ADAPTERS) | set(ADDITIONAL) | {"biomisa", "erlangen"})
PROFILE = "experimental35_raw"
REPLAY_PYTHON = "3.12.13"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = root / relative
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and path.resolve().is_relative_to(root),
        "Path escapes output/reference root",
    )
    return path


def row_id(dataset, stem, task):
    return dataset + "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", stem) + "_" + task


def native_source_id(dataset, stem, source_id):
    """Preserve parent prepare normalization and direct-v2 IDRiD source IDs."""
    source_id = source_id or f"{dataset}:{stem}"
    if (
        dataset not in set(ADDITIONAL) | {"biomisa"}
        and not source_id.startswith(dataset + ":")
        and not (dataset == "rite" and source_id.startswith("drive:"))
    ):
        source_id = dataset + ":" + source_id
    return source_id


def dependencies(row):
    """Explicit native resource requests, including implicit original-core inputs."""
    d, m = row["dataset"], row["metadata"]
    image = m["source_image"]
    result = {(ARCHIVES.get(d, "BIOMISA.zip"), image, "RGB")}
    archive = ARCHIVES.get(d, "BIOMISA.zip")
    for ref in m.get("annotation_resources", []):
        result.add((archive, ref["path"], "annotation"))
    for key in ("source_annotation", "artery_annotation", "vein_annotation", "center_annotation"):
        if m.get(key):
            result.add((archive, m[key], "annotation"))
    stem = Path(image).stem

    def add(path, role="annotation", file=archive):
        result.add((file, path, role))

    if d in {"drive", "rite"}:
        folder = "test" if m["source_split"] == "test" else "training"
        num = stem.split("_")[0]
        if d == "drive":
            add(f"{folder}/1st_manual/{num}_manual1.gif")
            add(f"{folder}/mask/{stem}_mask.gif", "native_FOV")
        else:
            add(f"{folder}/vessel/{stem}.png")
            add(f"{folder}/av/{stem}.png")
    elif d in {"hrf", "drhagis"}:
        add(f"manual1/{stem}.tif" if d == "hrf" else f"Manual_Segmentations/{stem}_manual_orig.png")
        add(
            f"mask/{stem}_mask.tif" if d == "hrf" else f"Mask_images/{stem}_mask_orig.png",
            "native_FOV",
        )
        if row["task"] == "av":
            add(f"{d.upper()}/clasified/{stem}.png", file=EXTRA)
    elif d == "refuge":
        folder = {
            "train": "Annotation-Training400",
            "val": "REFUGE-Validation400-GT",
            "test": "REFUGE-Test-GT",
        }[m["source_split"]]
        add(f"{folder}/Disc_Cup_Masks/{stem}.bmp")
    elif d == "drishti":
        folder, gt = ("Training", "GT") if m["source_split"] == "train" else ("Test", "Test_GT")
        base = f"{folder}/{gt}/{stem}/SoftMap/{stem}"
        add(base + "_ODsegSoftmap.png")
        add(base + "_cupsegSoftmap.png")
    elif d == "chasedb1":
        add(stem + "_1stHO.png")
        add(stem + "_2ndHO.png")
    elif d == "iostar":
        for folder, suffix in (
            ("GT", "_GT"),
            ("mask", "_Mask"),
            ("AV_GT", "_AV"),
            ("mask_OD", "_ODMask"),
        ):
            add(f"{folder}/{stem}{suffix}.tif", "native_FOV" if folder == "mask" else "annotation")
    elif d == "g1020":
        add(m["patient_crosswalk"]["resource"], "patient_sidecar")
    return sorted(result)


class FrozenPlan:
    def __init__(self, rows, photos):
        require(len({r["id"] for r in rows}) == len(rows), "Duplicate frozen row")
        require(len({p["photo_id"] for p in photos}) == len(photos), "Duplicate frozen photo")
        all_photos = {p["photo_id"]: copy.deepcopy(p) for p in photos}
        self.test_inventory = [
            {"frozen25_record": copy.deepcopy(r), "materialized35": False}
            for r in rows
            if r["split"] == "test"
        ]
        require(all(r["split"] in {"train", "val", "test"} for r in rows), "Unknown split")
        self.rows = {r["id"]: copy.deepcopy(r) for r in rows if r["split"] in {"train", "val"}}
        self.photos = {r["photo_id"]: all_photos[r["photo_id"]] for r in self.rows.values()}
        for r in self.rows.values():
            p = self.photos[r["photo_id"]]
            require(r["dataset"] in DATASETS, "Unsupported retained source")
            require(r["split"] == p["split"], "Frozen photo split conflict")
            require(r["pixels_per_degree"] == p["pixels_per_degree"] == 25, "Expected frozen25")
            require(r["metadata"]["source_image"], "Missing native resource")
            require(
                p["sources"] and p["canonical_dataset"] == p["sources"][0]["dataset"],
                "Missing canonical owner",
            )
            require(
                any(
                    x["dataset"] == r["dataset"] and x["resource"] == r["metadata"]["source_image"]
                    for x in p["sources"]
                ),
                "Row absent from frozen aliases",
            )
        for p in self.photos.values():
            owner = p["sources"][0]
            require(
                any(
                    r["photo_id"] == p["photo_id"]
                    and r["dataset"] == owner["dataset"]
                    and r["metadata"]["source_image"] == owner["resource"]
                    for r in self.rows.values()
                ),
                "Canonical owner lacks retained emission",
            )
        self.requests = {}
        for r in self.rows.values():
            for archive, path, role in dependencies(r):
                self.requests.setdefault(archive, {}).setdefault(path, set()).add(role)
        self.signature = sha(
            json.dumps(
                {
                    "rows": sorted(self.rows.values(), key=lambda r: r["id"]),
                    "photos": sorted(self.photos.values(), key=lambda p: p["photo_id"]),
                },
                sort_keys=True,
            ).encode()
        )

    def summary(self):
        return {
            "kind": "frozen_RAW35_metadata_plan",
            "execution_authorized": False,
            "membership_sha256": self.signature,
            "records": dict(Counter(r["split"] for r in self.rows.values())),
            "photos": dict(Counter(p["split"] for p in self.photos.values())),
            "row_sources": {
                s: sorted({r["dataset"] for r in self.rows.values() if r["split"] == s})
                for s in ("train", "val")
            },
            "VAL_groups": len(
                {r["split_group_id"] for r in self.rows.values() if r["split"] == "val"}
            ),
            "TEST_inventory_records_unmaterialized": len(self.test_inventory),
            "resource_requests": {
                a: {p: sorted(roles) for p, roles in sorted(v.items())}
                for a, v in sorted(self.requests.items())
            },
        }

    def decode_order(self):
        # Topological canonical owners first; reject an alias cycle rather than selecting a new owner.
        pending = {r["dataset"] for r in self.rows.values()}
        result = []
        while pending:
            ready = sorted(
                d
                for d in pending
                if all(
                    self.photos[r["photo_id"]]["canonical_dataset"] in set(result) | {d}
                    for r in self.rows.values()
                    if r["dataset"] == d
                )
            )
            require(ready, "Canonical owner ordering cycle requires explicit review")
            result.extend(ready)
            pending.difference_update(ready)
        return result


def resolve_requests(names, requests, identities=None):
    """Resolve metadata only. Ambiguous transport aliases require reviewed exact closure."""
    resolved = {}
    for requested in requests:
        suffix = canonical_path(requested).lower()
        exact = [n for n in names if n.lower() == suffix]
        found = exact or [n for n in names if n.lower().endswith("/" + suffix)]
        if len(found) > 1 and identities is not None:
            aliases = {identities[n] for n in found}
            if len(aliases) == 1 and None not in aliases:
                found = [min(found, key=lambda n: (len(n), n))]
        require(len(found) == 1, "Missing/ambiguous closure resource: " + requested)
        resolved[requested] = found[0]
    return resolved


class Budget:
    def __init__(self, profile, output):
        self.profile, self.output, self.started = profile, Path(output), time.monotonic()
        self.read_bytes = 0
        self.last_storage_check = 0

    def check(self, additional_read=0):
        self.read_bytes += additional_read
        require(
            time.monotonic() - self.started < self.profile["max_seconds"], "Time budget exceeded"
        )
        require(self.read_bytes <= self.profile["max_read_bytes"], "Read budget exceeded")
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            <= self.profile["max_rss_bytes"],
            "RSS budget exceeded",
        )
        require(
            shutil.disk_usage(self.output).free >= self.profile["min_free_bytes"],
            "Storage reserve exhausted",
        )
        if time.monotonic() - self.last_storage_check > 1:
            self.check_storage()

    def check_storage(self):
        self.last_storage_check = time.monotonic()
        total = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        require(total <= self.profile["max_output_bytes"], "Output storage budget exceeded")


class FilteredSource:
    """Strict resource closure; membership checked before any underlying bytes/read."""

    def __init__(self, source, names, dataset, rows, budget=None, expected=None):
        self.source, self.names, self.dataset = source, sorted(set(names)), dataset
        require(set(self.names) <= set(source.names), "Closure member missing")
        self.rows, self.budget, self.expected, self.resources = rows, budget, expected or {}, {}

    def retains(self, stem, task):
        return row_id(self.dataset, stem, task) in self.rows

    def find(self, path, *, optional=False):
        try:
            return resolve_requests(self.names, [path])[path]
        except ValueError:
            if optional and not any(
                n.lower() == path.lower() or n.lower().endswith("/" + path.lower())
                for n in self.names
            ):
                return None
            raise

    def images(self, folder, extension):
        return sorted(
            {
                Path(n).name
                for n in self.names
                if "/" + folder.lower().strip("/") + "/" in "/" + n.lower()
                and n.lower().endswith(extension.lower())
            }
        )

    def bytes(self, path):
        name = self.find(path)
        if self.budget:
            self.budget.check(self.expected[name]["size"])
        data = self.source.bytes(name)
        identity = {"size": len(data), "sha256": sha(data)}
        if name in self.expected:
            require(identity == self.expected[name], "Native resource byte identity drift: " + name)
        require(
            name not in self.resources or self.resources[name] == identity,
            "Resource changed during decode",
        )
        self.resources[name] = identity
        return data

    def text(self, path):
        data = self.bytes(path)
        if path.lower().endswith(".gz"):
            data = gzip.decompress(data)
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp1252")

    def read(self, path, mode="L"):
        data = self.bytes(path)
        if path.lower().endswith(".gz"):
            data = gzip.decompress(data)
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert(mode)).copy()


def project(prepared, target, valid, annotation_fov):
    t = np.stack([prepared.geometry.prepare_mask(a) for a in target]).astype(np.float32)
    v = np.stack([prepared.geometry.prepare_mask(a) for a in valid]).astype(bool)
    v &= prepared.fov[None]
    if annotation_fov is not None:
        v &= prepared.geometry.prepare_mask(np.asarray(annotation_fov) > 0)[None]
    return t, v


def support(target, valid, task):
    result = {
        "valid": valid.sum(axis=(1, 2)).tolist(),
        "positive": ((target > 0) & valid).sum(axis=(1, 2)).tolist(),
    }
    if task == "av":
        result["AV_scoring"] = int(
            (valid[1] & valid[2] & ((target[1] > 0) ^ (target[2] > 0))).sum()
        )
    return result


def preserve_support(old, new):
    for key in old:
        a, b = np.asarray(old[key]), np.asarray(new[key])
        require(not ((a > 0) & (b == 0)).any(), "Supervision support disappeared: " + key)


class Replay:
    """Per-emission replay gate, bounded alias-owner cache, and independent35 writer."""

    def __init__(self, canonical, plan, output, policy=None, budget=None):
        self.canonical, self.plan, self.output = Path(canonical), plan, Path(output)
        self.policy, self.budget = policy, budget
        self.seen, self.photo_done, self.cache = set(), {}, OrderedDict()
        self.rows, self.report = [], []
        self.remaining = Counter(r["photo_id"] for r in plan.rows.values())

    def emit(
        self,
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
        ident = row_id(dataset, stem, task)
        require(
            ident in self.plan.rows and ident not in self.seen,
            "Unexpected/duplicate emission: " + ident,
        )
        row, m = self.plan.rows[ident], dict(metadata or {})
        photo = self.plan.photos[row["photo_id"]]
        # Hash EVERY observer and alias emission, including repeated native resources.
        require(decoded_photo_id(image) == row["photo_id"], "Decoded emission RGB drift")
        source_id = native_source_id(dataset, stem, source_id)
        require(source_id == row["source_id"], "Source/patient identity drift")
        m.setdefault("source_image", m.get("original_resource"))
        m.setdefault("source_split", split)
        require(m["source_image"] == row["metadata"]["source_image"], "Native resource drift")
        require(m["source_split"] == row["metadata"]["source_split"], "Official/source split drift")
        for key in (
            "annotator",
            "official_split",
            "patient_id",
            "patient_group_id",
            "patient_crosswalk",
            "annotation_resources",
            "source_annotation",
            "cup_annotation",
            "annotation72_swap",
            "optic_disc_center_marker_centroid_xy",
            "annotation_policy",
            "source_export_roles",
        ):
            if key in m or key in row["metadata"]:
                require(
                    normalized(m.get(key)) == row["metadata"].get(key),
                    "Annotation provenance drift: " + key,
                )
        pid, owner = row["photo_id"], photo["sources"][0]
        if pid not in self.photo_done:
            require(
                dataset == owner["dataset"] and m["source_image"] == owner["resource"],
                "Frozen owner must emit first",
            )
        # Recompute owner geometry on every owner emission, including FOV variation.
        # Aliases must use that verified canonical owner's geometry, never alias FOV.
        if dataset == owner["dataset"] and m["source_image"] == owner["resource"]:
            native_fov = fov if dataset in OFFICIAL_FOV_DATASETS | {"rimone", "erlangen"} else None
            grids = {
                ppd: prepare_single_net(
                    image,
                    dataset=dataset,
                    fov=native_fov,
                    field_of_view_degrees=photo["field_of_view_degrees"],
                    pixels_per_degree=ppd,
                    normalized=False,
                )
                for ppd in (25, 35)
            }
            require(
                normalized(grids[25].geometry.as_dict()) == photo["geometry"],
                "Frozen25 geometry drift",
            )
            require(grids[25].fov_source == photo["fov_source"], "Frozen25 FOV backend drift")
            if pid in self.photo_done:
                require(
                    sha(grids[35].fov.tobytes()) == self.photo_done[pid]["fov35_sha256"],
                    "Owner FOV iteration drift",
                )
            self.cache[pid] = grids
        else:
            require(
                pid in self.cache, "Owner geometry evicted: decode owner/alias cluster together"
            )
            grids = self.cache[pid]
        self.cache.move_to_end(pid)
        if self.budget:
            require(
                len(self.cache) <= self.budget.profile["max_alias_cached_photos"],
                "Alias owner cache bound exceeded",
            )
        # The cache retains only cross-dataset alias owners. Other photo grids die after emit.
        p25, p35 = grids[25], grids[35]
        require(list(image.shape) == photo["raw_shape"], "Native shape drift")
        for relative in {photo["image"], row["image"]}:
            with Image.open(safe(self.canonical, relative)) as ref:
                require(
                    np.array_equal(p25.image, np.asarray(ref.convert("RGB"))),
                    "RAW25 RGB replay drift",
                )
        with Image.open(safe(self.canonical, photo["fov"])) as ref:
            require(
                np.array_equal(p25.fov, np.asarray(ref.convert("L")) > 0), "RAW25 FOV replay drift"
            )
        target = np.asarray(target, dtype=np.float32)
        valid = np.ones_like(target, bool) if valid is None else np.asarray(valid, bool)
        require(
            target.shape == valid.shape
            and target.ndim == 3
            and target.shape[1:] == image.shape[:2],
            "Native target geometry drift",
        )
        require(
            np.isfinite(target).all() and (target >= 0).all() and (target <= 1).all(),
            "Invalid target values",
        )
        t25, v25 = project(p25, target, valid, fov)
        with np.load(safe(self.canonical, row["mask"]), allow_pickle=False) as ref:
            for key, actual in (("target", t25), ("valid", v25), ("fov", p25.fov)):
                require(np.array_equal(actual, ref[key]), "RAW25 " + key + " replay drift")
        t35, v35 = project(p35, target, valid, fov)
        a, b = support(t25, v25, task), support(t35, v35, task)
        preserve_support(a, b)
        if self.budget:
            self.budget.check()
        # No35 pixels are committed for this emission before the complete25 replay gate.
        for folder in ("images", "fovs", "masks"):
            (self.output / folder).mkdir(exist_ok=True)
        if pid not in self.photo_done:
            new_photo = copy.deepcopy(photo)
            new_photo.update(
                geometry=normalized(p35.geometry.as_dict()),
                pixels_per_degree=35.0,
                preprocessing_profile=PROFILE,
                prepared_shape=list(p35.image.shape),
                prepared_rgb_sha256=decoded_photo_id(p35.image).removeprefix("sha256:"),
            )
            Image.fromarray(p35.image).save(safe(self.output, photo["image"]))
            Image.fromarray(p35.fov.astype(np.uint8) * 255).save(safe(self.output, photo["fov"]))
            self.photo_done[pid] = {"photo": new_photo, "fov35_sha256": sha(p35.fov.tobytes())}
        if row["image"] != photo["image"]:
            require(
                safe(self.output, row["image"]).parent == self.output.resolve() / "images",
                "Unexpected alias destination",
            )
            if not safe(self.output, row["image"]).exists():
                shutil.copyfile(safe(self.output, photo["image"]), safe(self.output, row["image"]))
        np.savez_compressed(safe(self.output, row["mask"]), target=t35, valid=v35, fov=p35.fov)
        result = copy.deepcopy(row)
        result.update(pixels_per_degree=35.0, preprocessing_profile=PROFILE)
        result["metadata"].update(
            pixels_per_degree=35.0,
            preprocessing_profile=PROFILE,
            original_geometry=normalized(p35.geometry.as_dict()),
            frozen25_record_sha256=sha(json.dumps(row, sort_keys=True).encode()),
            raw25_replay_exact=True,
        )
        if "FOV_diameter_pixels_and_estimator_from_actual_geometry" in result["metadata"]:
            result["metadata"]["FOV_diameter_pixels_and_estimator_from_actual_geometry"][
                "rescale_factor"
            ] = p35.geometry.rescale_factor
        self.rows.append(result)
        self.report.append(
            {"id": ident, "decoded_emission_RGB_sha256": pid, "support25": a, "support35": b}
        )
        self.seen.add(ident)
        self.remaining[pid] -= 1
        if (
            self.remaining[pid] == 0
            or len({(s["dataset"], s["resource"]) for s in photo["sources"]}) == 1
        ):
            self.cache.pop(pid, None)
        if self.budget:
            self.budget.check()

    def finish(self, preparation):
        require(self.seen == set(self.plan.rows), "Frozen membership incomplete")
        require(set(self.photo_done) == set(self.plan.photos), "Frozen photo membership incomplete")
        write_json(self.output / "manifest.json", sorted(self.rows, key=lambda r: r["id"]))
        photos = []
        for pid, entry in sorted(self.photo_done.items()):
            p = entry["photo"]
            p["annotations"] = [
                a for a in p["annotations"] if (a if isinstance(a, str) else a["id"]) in self.seen
            ]
            photos.append(p)
        write_json(self.output / "photos.json", photos)
        write_json(self.output / "test-inventory-unmaterialized.json", self.plan.test_inventory)
        write_json(self.output / "preparation.json", preparation)
        write_json(self.output / "replay-report.json", self.report)
        fingerprint = prepared_content_fingerprint(self.output / "manifest.json")
        manifest_sha = sha((self.output / "manifest.json").read_bytes())
        write_json(
            self.output / "integrity.json",
            {
                "fingerprint_schema": FINGERPRINT_SCHEMA,
                "manifest_sha256": manifest_sha,
                "prepared_content_sha256": fingerprint,
                "membership_sha256": self.plan.signature,
            },
        )
        if self.budget:
            self.budget.check()
            self.budget.check_storage()
        (self.output / "_SUCCESS").write_text(manifest_sha + "\n")
        return fingerprint


def validate_frozen_metadata(canonical):
    canonical = Path(canonical)
    require(
        sha((canonical / "manifest.json").read_bytes()) == MANIFEST_SHA, "Frozen manifest pin drift"
    )
    require(sha((canonical / "photos.json").read_bytes()) == PHOTOS_SHA, "Frozen photos pin drift")
    require(
        sha((canonical / "preparation.json").read_bytes()) == PREPARATION_SHA,
        "Frozen preparation pin drift",
    )
    require(
        json.loads((canonical / "integrity.json").read_text())["prepared_content_sha256"]
        == CONTENT_SHA,
        "Frozen content receipt drift",
    )


def load_plan(canonical, policy_path):
    canonical = Path(canonical)
    validate_frozen_metadata(canonical)
    policy_text = Path(policy_path).read_text()
    require(sha(policy_text.encode()) == POLICY_SHA, "Source geometry policy pin drift")
    embedded = {"format_version": 1, "sha256": POLICY_SHA, "document_json": policy_text}
    validate_geometry_policy(embedded)
    rows = json.loads((canonical / "manifest.json").read_text())
    photos = json.loads((canonical / "photos.json").read_text())
    plan = FrozenPlan(rows, photos)
    summary = plan.summary()
    require(
        summary["records"] == {"train": 5700, "val": 993}
        and summary["photos"] == {"train": 2529, "val": 770}
        and len(summary["row_sources"]["train"]) == 21
        and len(summary["row_sources"]["val"]) == 15
        and summary["VAL_groups"] == 612,
        "Frozen scope count drift",
    )
    for p in plan.photos.values():
        if p["canonical_dataset"] in {"drions", "gamma", "idrid_disc", "biomisa"}:
            selected = geometry_rule(embedded, p["canonical_dataset"], p["raw_shape"][:2])
            require(
                selected["field_of_view_degrees"] == p["field_of_view_degrees"],
                "Source angle rule drift",
            )
    return plan, embedded


def validate_profile(profile, plan, canonical, data_root, output):
    require(
        profile.get("execution_authorized") is True,
        "Execution is disabled; reviewed bounded profile required",
    )
    require(profile["membership_sha256"] == plan.signature, "Profile membership mismatch")
    validate_frozen_metadata(canonical)
    frozen = FrozenPlan(
        json.loads((Path(canonical) / "manifest.json").read_text()),
        json.loads((Path(canonical) / "photos.json").read_text()),
    )
    require(
        plan.signature == frozen.signature
        and plan.rows == frozen.rows
        and plan.photos == frozen.photos
        and plan.requests == frozen.requests
        and plan.test_inventory == frozen.test_inventory,
        "Execution plan differs from frozen metadata",
    )
    for field in (
        "max_seconds",
        "max_read_bytes",
        "max_output_bytes",
        "min_free_bytes",
        "max_alias_cached_photos",
        "max_rss_bytes",
    ):
        require(
            type(profile[field]) in (int, float)
            and np.isfinite(profile[field])
            and profile[field] > 0,
            "Explicit positive execution bound required: " + field,
        )
    require(
        profile["Python"] == platform.python_version() == REPLAY_PYTHON,
        "Replay Python backend drift",
    )
    require(
        {p: version(p) for p in profile["runtime_packages"]} == profile["runtime_packages"],
        "Replay backend drift",
    )
    historical = json.loads((Path(canonical) / "preparation.json").read_text())["runtime_packages"]
    require(profile["runtime_packages"] == historical, "Profile must pin original replay backend")
    require(set(profile["resources"]) == set(plan.requests), "Profile archive closure mismatch")
    for archive, requests in plan.requests.items():
        entry = profile["resources"][archive]
        require(
            set(entry["request_to_member"]) == set(requests),
            "Incomplete explicit resource resolution",
        )
        require(
            set(entry["request_to_member"].values()) == set(entry["members"]),
            "Unused/extra resource closure",
        )
        for identity in entry["members"].values():
            require(
                set(identity) == {"size", "sha256"}
                and type(identity["size"]) is int
                and identity["size"] > 0
                and re.fullmatch("[a-f0-9]{64}", identity["sha256"]),
                "Missing resource byte pin",
            )
    canonical, data_root, output = (
        Path(canonical).resolve(),
        Path(data_root).resolve(),
        Path(output).absolute(),
    )
    parent = Path(profile["exclusive_output_parent"]).resolve()
    require(
        parent.is_dir()
        and not output.exists()
        and output.parent.resolve() == parent
        and not output.is_symlink()
        and "owncloud" not in str(output).lower()
        and not output.resolve().is_relative_to(canonical)
        and not canonical.is_relative_to(output.resolve())
        and not output.resolve().is_relative_to(data_root)
        and not data_root.is_relative_to(output.resolve()),
        "Choose new independent exclusive local output",
    )
    require(
        shutil.disk_usage(parent).free >= profile["min_free_bytes"],
        "Insufficient initial disk reserve",
    )


def run(canonical, data_root, output, profile, plan, policy):
    """Explicit future execution API. Failure leaves incomplete output without _SUCCESS."""
    from contextlib import ExitStack

    validate_profile(profile, plan, canonical, data_root, output)
    require(policy["sha256"] == POLICY_SHA, "Source policy parent pin drift")
    validate_geometry_policy(policy)
    output = Path(output).absolute()
    output.mkdir()  # Exclusive creation. Never overwrite a previous attempt.
    budget = Budget(profile, output)
    replay = Replay(canonical, plan, output, policy, budget)
    resources = {}
    for dataset in plan.decode_order():
        with ExitStack() as stack:
            scopes = {}
            archives = {ARCHIVES.get(dataset, "BIOMISA.zip")}
            if dataset in {"hrf", "drhagis"} and EXTRA in plan.requests:
                archives.add(EXTRA)
            for archive in sorted(archives):
                source_path = safe(data_root, archive)
                require(
                    source_path.is_dir() or zipfile.is_zipfile(source_path),
                    "Producer supports indexed ZIP/directories only; TAR duplicate probing is not predecode-safe",
                )
                source = stack.enter_context(Source(source_path))
                pinned = profile["resources"][archive]
                # Resolve before pixel/GT decoding; no source.find fallback or byte-based alias search.
                actual = resolve_requests(
                    source.names,
                    plan.requests[archive],
                    source._identities if source.zip is not None else None,
                )
                require(actual == pinned["request_to_member"], "Native member resolution drift")
                scopes[archive] = FilteredSource(
                    source, pinned["members"], dataset, plan.rows, budget, pinned["members"]
                )
            scope = scopes[ARCHIVES.get(dataset, "BIOMISA.zip")]

            def emit(*args, _dataset=dataset, **kwargs):
                replay.emit(_dataset, *args, **kwargs)

            if dataset in CORE_DATASETS:
                decode_core(dataset, scope, emit, scopes.get(EXTRA))
            elif dataset == "biomisa":
                biomisa(scope, emit, label_policy=FIXED_POLICY)
            elif dataset == "erlangen":
                erlangen(scope, emit)
            else:
                {**ADAPTERS, **ADDITIONAL}[dataset](scope, emit)
            require(
                len(replay.cache) <= profile["max_alias_cached_photos"],
                "Alias owner cache bound exceeded",
            )
            for archive, scoped in scopes.items():
                resources.setdefault(archive, {}).update(scoped.resources)
    for archive, pinned in profile["resources"].items():
        require(
            set(resources.get(archive, {})) == set(pinned["members"]),
            "Closure resource not consumed",
        )
    # New35 contract has a new SHA. Parent2591 retains its original25 text untouched.
    derived_policy = {
        "kind": "native35_frozen_source_choices",
        "policy_id": "raw35_frozen2591_20261112_v1",
        "parent_geometry_policy_sha256": policy["sha256"],
        "parent_pixels_per_degree": 25,
        "target_pixels_per_degree": 35,
        "parent25_geometry_policy_sha256": policy["sha256"],
        "source_rule_document": json.loads(policy["document_json"]),
        "scale_anchor": "native max(H,W) before inclusive crop",
        "label_FOV_interpolation": "original nearest/order0",
    }
    policy_bytes = (json.dumps(derived_policy, sort_keys=True) + "\n").encode()
    (output / "geometry-policy35.json").write_bytes(policy_bytes)
    new_policy_sha = sha(policy_bytes)
    for row in replay.rows:
        if "geometry_policy_sha256" in row["metadata"]:
            row["metadata"].update(
                parent25_geometry_policy_sha256=POLICY_SHA,
                parent_geometry_policy_sha256=POLICY_SHA,
                geometry_policy_sha256=new_policy_sha,
                policy_sha256=new_policy_sha,
            )
    for entry in replay.photo_done.values():
        if "geometry_policy" in entry["photo"]:
            entry["photo"]["parent25_geometry_policy"] = entry["photo"].pop("geometry_policy")
        entry["photo"]["geometry35_policy_sha256"] = new_policy_sha
        entry["photo"]["geometry_policy"] = {
            "sha256": new_policy_sha,
            "pixels_per_degree": 35,
            "parent_geometry_policy_sha256": POLICY_SHA,
        }
    preparation = {
        "kind": PROFILE,
        "pixels_per_degree": 35,
        "frozen25_manifest_sha256": MANIFEST_SHA,
        "frozen25_photos_sha256": PHOTOS_SHA,
        "frozen25_content_sha256": CONTENT_SHA,
        "parent25_geometry_policy_sha256": POLICY_SHA,
        "parent_geometry_policy_sha256": POLICY_SHA,
        "geometry_policy_sha256": new_policy_sha,
        "membership_sha256": plan.signature,
        "runtime_packages": profile["runtime_packages"],
        "Python": profile["Python"],
        "execution_profile": profile,
        "resources": resources,
        "code_sha256": {
            str(p.relative_to(Path(__file__).parents[2])): sha(p.read_bytes())
            for p in Path(__file__).parent.rglob("*.py")
        },
        "TEST_materialized": False,
        "trainer35_admission": False,
    }
    budget.check()
    result = replay.finish(preparation)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "produce"))
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--geometry-policy", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, policy = load_plan(args.canonical, args.geometry_policy)
    if args.action == "plan":
        print(json.dumps(plan.summary(), indent=2))
    else:
        require(
            args.profile and args.profile_sha256 and args.data_root and args.output,
            "produce requires explicit profile, profile hash, data-root and output",
        )
        raw = args.profile.read_bytes()
        require(sha(raw) == args.profile_sha256, "Profile byte pin mismatch")
        print(run(args.canonical, args.data_root, args.output, json.loads(raw), plan, policy))


if __name__ == "__main__":
    main()
