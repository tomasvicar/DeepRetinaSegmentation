"""CPU-only FOV metadata audit and one TRAIN-photo photometric sanity; no model/GT."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from deepretina import photometry as p
from deepretina.legacy.preprocessing import prepare_single_net
from deepretina.unified import UnifiedConfig, UnifiedDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha256(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hashlib.sha256(np.asarray(rgb.shape, dtype="<i8").tobytes() + rgb.tobytes()).hexdigest()


def measure_train_sizes(manifest, index, eligible, canary):
    train = [pid for pid, split in eligible.items() if split == "train"]
    area = lambda pid: int(np.prod(index[pid]["prepared_shape"][:2]))
    medium = min(
        (pid for pid in train if pid != canary), key=lambda pid: abs(area(pid) - 1_000_000)
    )
    largest = max(train, key=area)
    selected = list(dict.fromkeys((canary, medium, largest)))
    results = []
    for profile in p.PROFILES:
        cfg = UnifiedConfig(photometric_profile=profile)
        dataset = UnifiedDataset(manifest, config=cfg)
        misses = hits = 0
        for pid in selected:
            record = index[pid]
            with Image.open(manifest.parent / record["image"]) as im:
                rgb = np.asarray(im.convert("RGB")).copy()
            fov = p.image_fov(manifest.parent, pid, rgb.shape, index)
            repetitions = []
            for _ in range(3):
                start = time.perf_counter()
                expected = p.apply(rgb, profile, fov=fov)
                repetitions.append(time.perf_counter() - start)
            position = dataset.photo_ids.index(pid)
            key = pid if profile == "raw" else (pid, profile, cfg.pixels_per_degree)
            before = key in dataset._image_cache
            start = time.perf_counter()
            first = dataset._image(position)
            cold = time.perf_counter() - start
            misses += int(not before)
            hits += int(before)
            before = key in dataset._image_cache
            start = time.perf_counter()
            cached = dataset._image(position)
            warm = time.perf_counter() - start
            misses += int(not before)
            hits += int(before)
            if first is not cached or not np.array_equal(
                first.numpy().transpose(1, 2, 0), expected.astype(np.float32) / 255
            ):
                raise ValueError("Cached/uncached full-image photometry differs")
            mpix = area(pid) / 1e6
            results.append(
                {
                    "profile": profile,
                    "photo_id": pid,
                    "dataset": record["canonical_dataset"],
                    "prepared_shape": list(rgb.shape),
                    "MPix": mpix,
                    "operator_seconds_three_calls": repetitions,
                    "operator_median_seconds": float(np.median(repetitions)),
                    "operator_MPix_per_second": mpix / float(np.median(repetitions))
                    if profile != "raw"
                    else None,
                    "operator_raw_identity_rate_not_bandwidth": profile == "raw",
                    "Dataset_cold_image_load_seconds": cold,
                    "Dataset_cold_MPix_per_second": mpix / cold,
                    "Dataset_LRU_hit_seconds": warm,
                    "cached_uncached_exact": True,
                    "outside_FOV_exact_raw": bool(np.array_equal(expected[~fov], rgb[~fov])),
                    "processed_pixel_sha256": pixel_sha256(expected),
                }
            )
        results.append(
            {
                "profile": profile,
                "bounded_sequence_requests": 2 * len(selected),
                "actual_full_image_LRU_misses": misses,
                "actual_LRU_hits": hits,
                "image_cache_limit_bytes": dataset._cache_limit,
                "cache_bytes_after_sequence": dataset._cache_bytes,
                "not_full_epoch_cache_miss_estimate": True,
            }
        )
    return results


def inspect(manifest, raw_image, photo_id, dataset, angle, output):
    import cv2

    cv2.setNumThreads(1)
    manifest, output = Path(manifest), Path(output)
    if (
        output.exists()
        or output.resolve().is_relative_to(manifest.parent.resolve())
        or "owncloud" in str(output).lower()
    ):
        raise ValueError("Fresh output outside canonical data/ownCloud required")
    rows = json.loads(manifest.read_text())
    index = p.fov_index(manifest.parent)
    eligible = {
        r["photo_id"]: r["split"]
        for r in rows
        if r["split"] in {"train", "val"}
        and r["task"] in {"vessels", "av", "disc_cup"}
        and r.get("usable_for_training", True)
        and not r.get("exclude_from_training", False)
    }
    headers = []
    for pid, split in sorted(eligible.items()):
        record = index[pid]
        relative = Path(record["fov"])
        path = manifest.parent / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(manifest.parent.resolve())
            or not path.is_file()
        ):
            raise ValueError("Canonical FOV unavailable/unsafe")
        if (
            record["preprocessing_profile"] != "legacy25_raw"
            or record["normalized"] is not False
            or record["pixels_per_degree"] != 25
        ):
            raise ValueError("Nonraw canonical photo geometry")
        with Image.open(path) as im:
            if im.size != (record["prepared_shape"][1], record["prepared_shape"][0]):
                raise ValueError("FOV/RGB size mismatch")
        headers.append(
            {
                "photo_id": pid,
                "split": split,
                "fov": relative.as_posix(),
                "encoded_bytes": path.stat().st_size,
                "fov_source": record.get("fov_source"),
                "prepared_shape": record["prepared_shape"],
            }
        )
    if eligible[photo_id] != "train":
        raise ValueError("Canary must be actual eligible TRAIN photo")
    with Image.open(raw_image) as im:
        raw = np.asarray(im.convert("RGB")).copy()
    record = index[photo_id]
    if pixel_sha256(raw) != record["raw_rgb_sha256"]:
        raise ValueError("Raw TRAIN photograph decoded identity mismatch")
    prepared = prepare_single_net(
        raw, dataset=dataset, field_of_view_degrees=angle, normalized=False
    )
    with Image.open(manifest.parent / record["image"]) as im:
        canonical = np.asarray(im.convert("RGB")).copy()
    domain = p.image_fov(manifest.parent, photo_id, canonical.shape, index)
    if not np.array_equal(canonical, prepared.image) or not np.array_equal(domain, prepared.fov):
        raise ValueError("Raw inference geometry/RGB/FOV differs from canonical TRAIN")
    cfg = UnifiedConfig()
    ds = UnifiedDataset(manifest, config=cfg)
    position = ds.photo_ids.index(photo_id)
    views = []
    results = {}
    for profile in p.PROFILES:
        timings = []
        value = None
        for _ in range(3):
            started = time.perf_counter()
            value = p.apply(prepared.image, profile, fov=prepared.fov)
            timings.append(time.perf_counter() - started)
        cfg.photometric_profile = profile
        tensor = ds._image(position)
        reconstructed = np.transpose(tensor.numpy(), (1, 2, 0))
        if not np.array_equal(reconstructed, value.astype(np.float32) / 255):
            raise ValueError(
                "Full prepared photometry differs between Dataset and raw-inference path"
            )
        if not np.array_equal(value[~domain], canonical[~domain]):
            raise ValueError("Outside-FOV background changed")
        views.append((profile, value))
        results[profile] = {
            "contract": p.contract(profile),
            "seconds_full_image_three_calls": timings,
            "median_seconds": float(np.median(timings)),
            "prepared_RGB_sha256": pixel_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "range": [int(value.min()), int(value.max())],
            "outside_FOV_exact_raw": True,
            "Dataset_vs_raw_inference_full_prepared_image_exact": True,
            "changed_inside_FOV_elements": int(
                np.count_nonzero(value[domain] != canonical[domain])
            ),
        }
    size_timings = measure_train_sizes(manifest, index, eligible, photo_id)
    output.mkdir(parents=True)
    thumbs = []
    for label, value in views:
        im = Image.fromarray(value)
        im.thumbnail((600, 500))
        canvas = Image.new("RGB", (600, 530), "white")
        canvas.paste(im, (0, 30))
        ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
        thumbs.append(canvas)
    contact = Image.new("RGB", (1800, 530), "white")
    for i, view in enumerate(thumbs):
        contact.paste(view, (600 * i, 0))
    contact.save(output / "TRAIN-private-contact-sheet.png")
    report = {
        "kind": "TRAIN_only_photometric_sanity_no_NN",
        "TRAIN_size_timings": size_timings,
        "OpenCV_threads_for_this_measurement_only": cv2.getNumThreads(),
        "manifest_sha256": sha(manifest),
        "photos_json_sha256": sha(manifest.parent / "photos.json"),
        "eligible_TRAIN_photos": sum(s == "train" for s in eligible.values()),
        "eligible_VAL_photos": sum(s == "val" for s in eligible.values()),
        "all_referenced_FOV_files_exist_with_matching_headers": True,
        "header_inventory_sha256": hashlib.sha256(
            json.dumps(headers, sort_keys=True).encode()
        ).hexdigest(),
        "canonical_fingerprint_already_covers_FOVs": "schema2 hashes photos.json and every photo image/fov; this audit does not rehash entire corpus",
        "photo_id": photo_id,
        "raw_image_encoded_sha256": sha(raw_image),
        "raw_shape": list(raw.shape),
        "decoded_pixel_sha256_scheme": "SHA256(little-endian int64 shape + contiguous uint8 RGB bytes)",
        "prepared_shape": list(canonical.shape),
        "declared_dataset": dataset,
        "declared_field_angle": angle,
        "photometry": results,
        "contact_sheet_sha256": sha(output / "TRAIN-private-contact-sheet.png"),
        "GT_arrays_weights_NN_GPU_or_TEST_decoding": False,
        "quality_improvement_claim": False,
        "performance_qualification": "Single TRAIN full-image transform calls, CPU only; not DataLoader/GPU epoch speed. Repeated image-cache misses can repeat this cost.",
    }
    (output / "sanity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "raw-image", "photo-id", "dataset", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--field-angle", type=float, required=True)
    args = parser.parse_args()
    inspect(
        args.manifest, args.raw_image, args.photo_id, args.dataset, args.field_angle, args.output
    )
