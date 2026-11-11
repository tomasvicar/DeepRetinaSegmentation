"""Replay exactly three pinned TRAIN photos at25, then prepare them directly from RAW at35.

This is not a full-corpus producer, trainer, new split policy or unified35 API.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import resource
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from deepretina.adapters.core import decode_core
from deepretina.adapters.segmentation import ADAPTERS
from deepretina.legacy.preprocessing import OFFICIAL_FOV_DATASETS, ImageGeometry, prepare_single_net
from deepretina.sources import Source
from deepretina.unified import restore_probabilities

MANIFEST_SHA = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
PHOTO_IDS = {
    "sha256:04ddfc23f162d135569b49322c9cceafc0c20a0677280d6e18256b451d0a431c",
    "sha256:00ae0718507542427abed32506bca5de853dfc9a89a0d8c85272d28bfdeec400",
    "sha256:032b428eac5d8fd1cb5fee4897749914d6ee19fb8851e6e0eb54ef20143d0493",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rgb_hash(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return digest(np.asarray(image.shape, dtype="<i8").tobytes() + image.tobytes())


def normalize(value):
    return json.loads(json.dumps(value))


class FilteredSource:
    """Filter before enumeration/find/read; no underlying pixel decode outside allowlist."""

    def __init__(self, source, names):
        self.source = source
        self.names = sorted(names)
        assert set(self.names) <= set(source.names)
        self.resources = {}
        self.reads = []

    def find(self, suffix, *, optional=False):
        suffix = suffix.lower()
        exact = [n for n in self.names if n.lower() == suffix]
        found = exact or [n for n in self.names if n.lower().endswith("/" + suffix)]
        if not found:
            if optional:
                return None
            raise ValueError("Resource outside exact TRAIN allowlist: " + suffix)
        if len(found) != 1:
            raise ValueError("Ambiguous TRAIN resource: " + suffix)
        return found[0]

    def images(self, folder, extension):
        return sorted(
            {
                Path(n).name
                for n in self.names
                if "/" + folder.lower().strip("/") + "/" in "/" + n.lower()
                and n.lower().endswith(extension)
            }
        )

    def bytes(self, path):
        name = self.find(path)
        data = self.source.bytes(name)
        record = {"bytes": len(data), "sha256": digest(data)}
        if name in self.resources and self.resources[name] != record:
            raise ValueError("Selected encoded resource changed during replay")
        self.resources[name] = record
        return data

    def read(self, path, mode="L"):
        name = self.find(path)  # Crucially before opening any image.
        data = self.bytes(name)
        self.reads.append({"resource": name, "mode": mode})
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert(mode)).copy()


def safe(root, relative):
    path = Path(root) / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError("Reference path escapes canonical root")
    return path


def channels(target, valid):
    return [
        {
            "valid_pixels": int(v.sum()),
            "valid_positive_pixels": int((v & (t > 0)).sum()),
            "positive_value_mass": float(t[v].sum(dtype=np.float64)),
            "target_values": np.unique(t).astype(float).tolist(),
        }
        for t, v in zip(target, valid)
    ]


def av_supports(target, valid):
    known = valid[1] & valid[2]
    scoring = known & ((target[1] > 0) ^ (target[2] > 0))
    return {
        "AV_class_validity_pixels_including_background": int(known.sum()),
        "AV_conditional_scoring_pixels": int(scoring.sum()),
        "AV_unknown_vessel_pixels": int(((target[0] > 0) & valid[0] & ~known).sum()),
    }


def project(prepared, target, valid, annotation_fov):
    geometry = prepared.geometry
    target = np.stack([geometry.prepare_mask(a) for a in target]).astype(np.float32)
    valid = np.stack([geometry.prepare_mask(a) for a in valid]).astype(bool)
    valid &= prepared.fov[None]
    if annotation_fov is not None:
        valid &= geometry.prepare_mask(np.asarray(annotation_fov) > 0)[None]
    return target, valid


def regrid35_to25(probability, prepared35, prepared25):
    """One continuous joint-A/V resize into a declared fixed25 grid; never resize GT."""
    a, b = prepared35.geometry, prepared25.geometry
    if (
        a.original_shape != b.original_shape
        or a.crop_position != b.crop_position
        or prepared35.field_of_view_degrees != prepared25.field_of_view_degrees
        or prepared35.pixels_per_degree != 35
        or prepared25.pixels_per_degree != 25
    ):
        raise ValueError("Cross-grid geometry does not share frozen original frame/crop/angle")
    probability = np.asarray(probability, dtype=np.float32).copy()
    assert probability.shape == (5, *a.prepared_shape)
    probability[[0, 3, 4]] *= prepared35.fov[None]
    h, w = b.prepared_shape
    mapping = ImageGeometry((h, w), (0, h - 1, 0, w - 1), 1.0, a.prepared_shape)
    result = restore_probabilities(probability, mapping)
    result[[0, 3, 4]] *= prepared25.fov[None]
    result[1:3, ~prepared25.fov] = 0.5
    return result


class Replay:
    def __init__(self, canonical, rows, photos, output):
        self.canonical, self.output = Path(canonical), Path(output)
        self.rows = {r["id"]: r for r in rows}
        self.photos = {p["photo_id"]: p for p in photos}
        assert set(self.photos) == PHOTO_IDS and len(self.rows) == 7
        assert all(r["split"] == "train" for r in rows)
        self.seen = set()
        self.prepared = {}
        self.report = []
        self.resource_rgb = {}

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
        metadata = dict(metadata or {})
        ident = dataset + "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", stem) + "_" + task
        assert ident in self.rows and ident not in self.seen, "Unknown/duplicate annotation"
        frozen = self.rows[ident]
        photo = self.photos[frozen["photo_id"]]
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3
        # Never memoize this by (dataset,resource): every observer emission is hashed.
        actual_rgb = rgb_hash(image)
        assert "sha256:" + actual_rgb == frozen["photo_id"], "Decoded emission RGB differs"
        key = (dataset, metadata["source_image"])
        if key in self.resource_rgb and self.resource_rgb[key] != actual_rgb:
            raise ValueError("Same resource emitted different RGB")
        self.resource_rgb[key] = actual_rgb
        source_id = source_id or f"{dataset}:{stem}"
        if not source_id.startswith(dataset + ":") and not (
            dataset == "rite" and source_id.startswith("drive:")
        ):
            source_id = dataset + ":" + source_id
        assert (
            source_id == frozen["source_id"]
            and metadata["source_image"] == frozen["metadata"]["source_image"]
        )
        assert split == frozen["metadata"]["source_split"]
        assert metadata.get("annotator") == frozen["metadata"].get("annotator")
        for field in [
            "annotation_resources",
            "source_annotation",
            "cup_annotation",
            "annotation72_swap",
            "optic_disc_center_marker_centroid_xy",
            "split_policy",
            "official_split",
        ]:
            if field in frozen["metadata"]:
                assert normalize(metadata.get(field)) == frozen["metadata"][field], (
                    "Annotation provenance changed: " + field
                )
        pid = frozen["photo_id"]
        owner = photo["sources"][0]
        if pid not in self.prepared:
            assert dataset == photo["canonical_dataset"] == owner["dataset"]
            assert metadata["source_image"] == owner["resource"], (
                "First alias is not frozen canonical owner"
            )
            native_fov = fov if dataset in OFFICIAL_FOV_DATASETS | {"rimone", "erlangen"} else None
            self.prepared[pid] = {
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
            replay25 = self.prepared[pid][25]
            assert normalize(replay25.geometry.as_dict()) == photo["geometry"]
            assert replay25.fov_source == photo["fov_source"]
            assert list(image.shape) == photo["raw_shape"]
            for ppd, prepared in self.prepared[pid].items():
                directory = self.output / str(ppd)
                directory.mkdir(exist_ok=True)
                Image.fromarray(prepared.image).save(directory / (actual_rgb + ".png"))
                Image.fromarray(prepared.fov.astype(np.uint8) * 255).save(
                    directory / (actual_rgb + "-fov.png")
                )
        prepared25 = self.prepared[pid][25]
        for relative in [photo["image"], frozen["image"]]:
            with Image.open(safe(self.canonical, relative)) as ref:
                assert np.array_equal(prepared25.image, np.asarray(ref.convert("RGB"))), (
                    "RAW25 RGB replay drift"
                )
        with Image.open(safe(self.canonical, photo["fov"])) as ref:
            assert np.array_equal(prepared25.fov, np.asarray(ref.convert("L")) > 0), (
                "RAW25 FOV replay drift"
            )
        target = np.asarray(target, dtype=np.float32)
        valid = np.ones_like(target, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        assert np.isfinite(target).all() and (target >= 0).all() and (target <= 1).all()
        row_report = {
            "id": ident,
            "photo_id": pid,
            "dataset": dataset,
            "source_id": source_id,
            "split": frozen["split"],
            "split_group_id": frozen["split_group_id"],
            "annotator": metadata.get("annotator"),
            "source_image": metadata["source_image"],
            "decoded_emission_RGB_sha256": actual_rgb,
            "hash_stage": "adapter RGB before UoA historical watermark suppression or any geometry",
            "canonical_owner": owner,
            "native_channels": channels(target, valid),
            "grids": {},
        }
        for ppd, prepared in self.prepared[pid].items():
            projected, allowed = project(prepared, target, valid, fov)
            if ppd == 25:
                with np.load(safe(self.canonical, frozen["mask"]), allow_pickle=False) as ref:
                    for field, actual in [
                        ("target", projected),
                        ("valid", allowed),
                        ("fov", prepared.fov),
                    ]:
                        assert np.array_equal(actual, ref[field]), (
                            "RAW25 " + field + " replay drift"
                        )
            np.savez_compressed(
                self.output / str(ppd) / (ident + ".npz"),
                target=projected,
                valid=allowed,
                fov=prepared.fov,
            )
            data = {
                "geometry": normalize(prepared.geometry.as_dict()),
                "channels": channels(projected, allowed),
                "RGB_shape": list(prepared.image.shape),
                "FOV_pixels": int(prepared.fov.sum()),
                "actual25_reference_exact": ppd == 25,
            }
            if task == "av":
                data.update(av_supports(projected, allowed))
            row_report["grids"][str(ppd)] = data
        self.seen.add(ident)
        self.report.append(row_report)


def run(data_root, canonical, output, packet):
    started = time.monotonic()
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
    actual_backend = {k: version(k) for k in packet["replay_backend_versions"]}
    assert actual_backend == packet["replay_backend_versions"]
    assert platform.python_version() == packet["Python"]
    canonical = Path(canonical)
    output = Path(output)
    assert (
        digest((canonical / "manifest.json").read_bytes())
        == MANIFEST_SHA
        == packet["manifest_sha256"]
    )
    assert digest((canonical / "photos.json").read_bytes()) == packet["photos_sha256"]
    assert (
        json.loads((canonical / "preparation.json").read_text())["runtime_packages"]
        == actual_backend
    )
    rows = [
        r
        for r in json.loads((canonical / "manifest.json").read_text())
        if r["id"] in packet["annotation_ids"]
    ]
    photos = [
        p for p in json.loads((canonical / "photos.json").read_text()) if p["photo_id"] in PHOTO_IDS
    ]
    assert {r["id"] for r in rows} == set(packet["annotation_ids"]) and all(
        r["split"] == "train" for r in rows
    )
    output = output.absolute()
    assert (
        not output.exists()
        and "owncloud" not in str(output).lower()
        and not output.is_relative_to(canonical)
    )
    output.mkdir()
    replay = Replay(canonical, rows, photos, output)
    resources = {}
    reads = {}
    archives = {
        "drive": "DRIVE.zip",
        "rite": "RITE.zip",
        "rimone": "RIM-ONE.zip",
        "uoadr": "UoA_DR.zip",
    }
    historical = json.loads((canonical / "parent-v1-metadata/preparation.json").read_text())
    historical = {Path(x["path"]).name: x for x in historical["archives"]}
    for dataset in packet["decode_order"]:
        archive = archives[dataset]
        path = Path(data_root) / archive
        before = path.stat()
        assert before.st_size == historical[archive]["size"]
        with Source(path) as source:
            allowed = packet["resources_exact_allowlist"][archive]
            for name in allowed:
                info = source._entries[name]
                prior = [
                    v
                    for v in historical[archive]["members"]
                    if v["name"] == name or v["name"].endswith("/" + name)
                ]
                assert prior and {(v["size"], v["crc32"]) for v in prior} == {
                    (info.file_size, info.CRC)
                }, "Historical resource CRC/size drift"
            scoped = FilteredSource(source, allowed)

            def emit(*args, _dataset=dataset, **kwargs):
                replay.emit(_dataset, *args, **kwargs)

            if dataset in {"drive", "rite"}:
                decode_core(dataset, scoped, emit)
            else:
                ADAPTERS[dataset](scoped, emit)
            assert set(scoped.resources) == set(allowed), "Selected resource not consumed"
            resources[archive] = scoped.resources
            reads[archive] = scoped.reads
        after = path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    assert replay.seen == set(packet["annotation_ids"])
    assert digest((canonical / "manifest.json").read_bytes()) == MANIFEST_SHA
    report = {
        "scope": "Only3 TRAIN photos/7 annotations; RAW25 exact replay and direct RAW35; not full corpus/API or model quality",
        "manifest_sha256": MANIFEST_SHA,
        "packet_sha256": digest((json.dumps(packet, indent=2) + "\n").encode()),
        "rows": replay.report,
        "backend_versions": actual_backend,
        "Python": platform.python_version(),
        "actual_CUDA_initialized": torch.cuda.is_initialized(),
        "soft_target_policy": "Keep float32 target values exactly through nearest geometry; unchanged source RIM binary>127 decoding, no new soft GT interpretation",
        "resources": resources,
        "actual_decodes": reads,
        "seconds": time.monotonic() - started,
        "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "VAL_TEST_decodes": 0,
        "geometry35_source": "native adapter RGB and native categorical targets, never25 prepared/cache upsampling",
        "split_assignment": "exact frozen TRAIN rows/source/groups/observers, no split calculation",
        "new_trainer_CLI_or_cache35_support": False,
        "GPU_or_NN": False,
    }
    assert report["peak_RSS_bytes"] < 3 * 1024**3
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "_SUCCESS").write_text(digest((output / "report.json").read_bytes()) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data-root", "canonical", "output", "packet"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    a = parser.parse_args()
    raw = a.packet.read_bytes()
    assert digest(raw) == a.packet_sha256
    print(json.dumps(run(a.data_root, a.canonical, a.output, json.loads(raw))))


if __name__ == "__main__":
    main()
