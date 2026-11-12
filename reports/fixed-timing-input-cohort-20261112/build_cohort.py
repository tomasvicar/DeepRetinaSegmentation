"""Freeze two preselected TRAIN RGB/official-FOV pairs; no model/timing/GT readers."""

import argparse
import hashlib
import importlib.util
import io
import json
import struct
import sys
import unicodedata
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def photo_identity(rgb):
    return "sha256:" + sha(struct.pack("<qqq", rgb.height, rgb.width, 3) + rgb.tobytes())


def archive_state(path):
    s = path.stat()
    return {
        "bytes": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
        "device": s.st_dev,
        "inode": s.st_ino,
    }


def selected_bytes(archive, suffix):
    """Read exactly one selected entry. Alias agreement uses directory CRC/size only."""

    def canonical(name):
        name = unicodedata.normalize("NFC", name.replace("\\", "/")).lstrip("/")
        marker = "/data/databases/"
        where = name.lower().find(marker)
        return name[where + len(marker) :] if where >= 0 else name

    matches = [
        i
        for i in archive.infolist()
        if not i.is_dir()
        and (
            canonical(i.filename).lower() == suffix.lower()
            or canonical(i.filename).lower().endswith("/" + suffix.lower())
        )
    ]
    assert matches, "Missing selected official resource; never substitute a GT mask"
    assert len({(i.file_size, i.CRC) for i in matches}) == 1, "Conflicting transport aliases"
    exact = [i for i in matches if canonical(i.filename).lower() == suffix.lower()]
    info = min(exact or matches, key=lambda i: (len(i.filename), i.filename))
    assert 0 < info.file_size <= 20 * 1024**2
    raw = archive.read(info)
    assert len(raw) == info.file_size
    return raw, {
        "member": suffix,
        "encoded_sha256": sha(raw),
        "encoded_bytes": len(raw),
        "zip_CRC32": info.CRC,
        "transport_alias_count": len(matches),
        "transport_alias_CRC_size_equal": True,
    }


def main(args):
    assert sha(args.manifest.read_bytes()) == MANIFEST
    rows = json.loads(args.manifest.read_bytes())
    photos = {
        p["photo_id"]: p for p in json.loads((args.manifest.parent / "photos.json").read_bytes())
    }
    eligible = [
        r
        for r in rows
        if r["split"] == "train"
        and r["task"] in {"vessels", "av", "disc_cup"}
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    ]
    hrf_paths = sorted({r["metadata"]["source_image"] for r in eligible if r["dataset"] == "hrf"})
    assert hrf_paths
    selections = [
        (
            "drive21",
            "drive",
            "training/images/21_training.tif",
            "training/mask/21_training_mask.gif",
            args.drive_archive,
            45.0,
            [584, 565],
        ),
        (
            "hrf_lexfirst_train",
            "hrf",
            hrf_paths[0],
            "mask/" + Path(hrf_paths[0]).stem + "_mask.tif",
            args.hrf_archive,
            60.0,
            [2336, 3504],
        ),
    ]
    fov_module_path = ROOT / "src/deepretina/legacy/matlab_fov.py"
    spec = importlib.util.spec_from_file_location("timing_cohort_pure_fov", fov_module_path)
    fov_math = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fov_math
    spec.loader.exec_module(fov_math)
    assert "torch" not in sys.modules, "No Torch import permitted in this preparation"
    assert not args.output.exists() and "owncloud" not in str(args.output).lower()
    args.output.mkdir(parents=True)
    public, private = [], []
    for label, dataset, image_member, fov_member, archive_path, angle, shape in selections:
        selected = [
            r
            for r in eligible
            if r["dataset"] == dataset and r["metadata"]["source_image"] == image_member
        ]
        assert selected and len({r["photo_id"] for r in selected}) == 1
        pid = selected[0]["photo_id"]
        photo = photos[pid]
        assert photo["split"] == "train" and photo["field_of_view_degrees"] == angle
        assert photo["fov_source"] == "provided original-frame mask"
        assert all(
            r["field_of_view_degrees"] == angle and r["original_size"] == shape for r in selected
        )
        before = archive_state(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            rgb_raw, rgb_pin = selected_bytes(archive, image_member)
            fov_raw, fov_pin = selected_bytes(archive, fov_member)
        assert archive_state(archive_path) == before
        with Image.open(io.BytesIO(rgb_raw)) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
        with Image.open(io.BytesIO(fov_raw)) as handle:
            fov = handle.convert("L")
            fov.load()
        assert [rgb.height, rgb.width] == [fov.height, fov.width] == shape
        assert photo_identity(rgb) == pid == "sha256:" + photo["raw_rgb_sha256"]
        assert all(r["metadata"]["raw_rgb_sha256"] == photo["raw_rgb_sha256"] for r in selected)
        binary = np.asarray(fov) > 127
        assert binary.any() and not binary.all()
        calibration = fov_math.calibrate_provided_fov(binary, dataset=dataset)
        relative_rgb, relative_fov = (
            label + Path(image_member).suffix,
            label + "-official-FOV" + Path(fov_member).suffix,
        )
        for filename, raw in [(relative_rgb, rgb_raw), (relative_fov, fov_raw)]:
            with (args.output / filename).open("xb") as stream:
                stream.write(raw)
            (args.output / filename).chmod(0o444)
            assert sha((args.output / filename).read_bytes()) == sha(raw)
        public.append(
            {
                "id": label,
                "dataset": dataset,
                "split": "train",
                "photo_id": pid,
                "source_id": selected[0]["source_id"],
                "split_group_id": selected[0]["split_group_id"],
                "TRAIN_record_ids": sorted(r["id"] for r in selected),
                "archive_basename": archive_path.name,
                "RGB": {
                    **rgb_pin,
                    "artifact_relative_path": relative_rgb,
                    "raw_shape_HWC": [*shape, 3],
                    "decoded_photo_id": pid,
                    "decoded_identity_convention": "SHA256(shape little-endian int64 H,W,3 + contiguous RGB uint8 bytes)",
                },
                "official_FOV": {
                    **fov_pin,
                    "artifact_relative_path": relative_fov,
                    "raw_shape_HW": shape,
                    "decoded_display_intensity_threshold": "PIL convert L >127, exactly dataset adapter",
                    "foreground_pixels": int(binary.sum()),
                    "binary_pixels_sha256": sha(np.ascontiguousarray(binary).tobytes()),
                    "source": "official dataset FOV under mask/, never manual1 or disc/cup/vessel annotation",
                },
                "geometry": {
                    "field_of_view_degrees": angle,
                    "pixels_per_degree": 25.0,
                    "original_provided_FOV_calibration": {
                        **asdict(calibration),
                        "diameter_pixels": calibration.diameter_pixels,
                    },
                    "canonical_preparation_geometry": selected[0]["metadata"]["original_geometry"],
                },
                "selection_policy": "fixed DRIVE TRAIN21"
                if dataset == "drive"
                else "lexicographically first distinct source_image among eligible canonical TRAIN HRF, before any speed/model measurement",
            }
        )
        private.append(
            {
                "id": label,
                "archive_path": str(archive_path),
                "archive_stat_before_after": before,
                "selected_rows": selected,
                "photo_metadata": photo,
            }
        )
    assert "torch" not in sys.modules
    cohort = {
        "schema_version": 1,
        "kind": "fixed_two_geometry_TRAIN_timing_input_cohort_NOT_benchmark_result",
        "manifest_sha256": MANIFEST,
        "prepared_content_sha256": "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52",
        "geometry_policy_sha256": "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d",
        "cohort_selected_before_final_model_and_speed": True,
        "inputs": public,
        "hrf_eligible_distinct_source_images": len(hrf_paths),
        "hrf_sorted_first10_source_images": hrf_paths[:10],
        "FOV_calibration_source_sha256": sha(fov_module_path.read_bytes()),
        "adapter_source_sha256": sha((ROOT / "src/deepretina/adapters/core.py").read_bytes()),
        "no_GT_TEST_models_Torch_GPU_or_timing": True,
        "limitations": [
            "Two deterministic TRAIN geometries only; not a population/clinical performance estimate.",
            "Do not replace an image after seeing latency; same RGB/FOV bytes must feed both final pipelines.",
            "No quality, memory-fit, speedup or final model selection claim.",
        ],
    }
    (args.output / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")
    (args.output / "private-source-snapshot.json").write_text(
        json.dumps(
            {
                "manifest_path": str(args.manifest),
                "artifact_root": str(args.output),
                "sources": private,
                "only_selected_RGB_and_official_FOV_entries_read": True,
                "whole_archives_not_hashed_or_written": True,
            },
            indent=2,
        )
        + "\n"
    )
    for p in args.output.iterdir():
        p.chmod(0o444)
    print(
        json.dumps(
            {
                "cohort_path": str(args.output / "cohort.json"),
                "cohort_sha256": sha((args.output / "cohort.json").read_bytes()),
                "inputs": [
                    {k: x[k] for k in ["id", "dataset", "photo_id", "geometry"]} for x in public
                ],
                "Torch_imported": "torch" in sys.modules,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "drive-archive", "hrf-archive", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
