"""Immutable, profile-specific full-image RGB derivatives; labels stay canonical."""

import argparse
import io
import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from . import photometry
from .integrity import verify_prepared_content

SCHEMA = 1
MAX_INDEX_BYTES = 16 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        import hashlib

        return hashlib.file_digest(stream, "sha256").hexdigest()


def operator_fingerprint():
    root = Path(__file__).parent
    h = sha256()
    for name in ("photometry.py", "legacy/preprocessing.py"):
        h.update(name.encode() + b"\0" + (root / name).read_bytes())
    return h.hexdigest()


def fingerprint(index_bytes):
    return sha256(b"photometry-RGB-cache:1\n" + index_bytes).hexdigest()


def safe(root, relative):
    relative = Path(relative)
    result = Path(root) / relative
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and result.resolve().is_relative_to(Path(root).resolve()),
        "Cache/source path escapes root",
    )
    require(
        result.is_file() and not result.is_symlink(),
        "Regular nonsymlink cache/source file required",
    )
    return result


def eligible_photos(manifest):
    partitions = {}
    for row in json.loads(Path(manifest).read_text()):
        if (
            row["split"] not in {"train", "val"}
            or row["task"] not in {"vessels", "av", "disc_cup"}
            or not row.get("usable_for_training", True)
            or row.get("exclude_from_training", False)
        ):
            continue
        pid = row["photo_id"]
        require(
            partitions.setdefault(pid, row["split"]) == row["split"], "Photo crosses partitions"
        )
    require(bool(partitions), "No eligible TRAIN/VAL photos")
    return partitions


def build(
    manifest,
    output,
    profile,
    *,
    expected_content_sha256,
    expected_descriptor_sha256,
    photo_ids=None,
    subset_proof=False,
):
    """Full build verifies canonical encoded bytes; explicit subset proofs cannot train full_data."""
    manifest, output = Path(manifest).resolve(), Path(output).absolute()
    require(profile in photometry.PROFILES[1:], "Photometry cache requires a nonraw profile")
    descriptor = photometry.contract(profile)
    require(
        sha256(canonical(descriptor)).hexdigest() == expected_descriptor_sha256,
        "Externally reviewed descriptor/backend differs",
    )
    require(
        not output.exists()
        and not output.is_symlink()
        and "owncloud" not in str(output.resolve()).lower()
        and not output.resolve().is_relative_to(manifest.parent),
        "Fresh output outside canonical/ownCloud required",
    )
    audit = json.loads((manifest.parent / "integrity.json").read_text())
    require(
        audit["fingerprint_schema"] == 2
        and audit["prepared_content_sha256"] == expected_content_sha256
        and audit["manifest_sha256"] == sha(manifest),
        "Canonical metadata pin differs",
    )
    require(
        (manifest.parent / "_SUCCESS").read_text().strip() == sha(manifest),
        "Canonical completion marker differs",
    )
    required = eligible_photos(manifest)
    chosen = set(required) if photo_ids is None else set(photo_ids)
    require(bool(chosen) and chosen <= set(required), "Unknown or empty photo selection")
    require(subset_proof or chosen == set(required), "Partial build requires explicit subset_proof")
    require(
        not subset_proof or all(required[pid] == "train" for pid in chosen),
        "Proof subset is TRAIN-only",
    )
    before = {
        "manifest": sha(manifest),
        "photos": sha(manifest.parent / "photos.json"),
        "operator": operator_fingerprint(),
        "builder": sha(__file__),
    }
    if not subset_proof:
        verify_prepared_content(manifest, expected_content_sha256)
    from .unified import _training_source_sha256

    construction_training_source = _training_source_sha256()
    photos = photometry.fov_index(manifest.parent)
    aliases = {}
    for row in json.loads(manifest.read_text()):
        if (
            row.get("photo_id") in chosen
            and row["split"] in {"train", "val"}
            and row["task"] in {"vessels", "av", "disc_cup"}
            and row.get("usable_for_training", True)
            and not row.get("exclude_from_training", False)
        ):
            aliases.setdefault(row["photo_id"], set()).add(row["image"])
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / (output.name + ".build-lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
        records = {}
        for pid in sorted(chosen):
            photo = photos[pid]
            source = safe(manifest.parent, photo["image"])
            fov_source = safe(manifest.parent, photo["fov"])
            input_sha, fov_sha = sha(source), sha(fov_source)
            with Image.open(source) as im:
                raw = np.asarray(im.convert("RGB")).copy()
            alias_sha = {}
            for relative_alias in sorted(aliases[pid]):
                alias = safe(manifest.parent, relative_alias)
                alias_sha[relative_alias] = sha(alias)
                with Image.open(alias) as im:
                    require(
                        np.array_equal(raw, np.asarray(im.convert("RGB"))),
                        "Photo RGB aliases differ",
                    )
            fov = photometry.image_fov(manifest.parent, pid, raw.shape, photos)
            try:
                processed = photometry.apply(raw, profile, fov=fov)
            except ValueError as exc:
                raise ValueError(f"Photometry failed for {pid}: {exc}") from exc
            require(
                processed.dtype == np.uint8
                and processed.shape == raw.shape
                and np.array_equal(processed[~fov], raw[~fov]),
                "Photometry output contract differs",
            )
            relative = "rgb/" + sha256(pid.encode()).hexdigest() + ".npy"
            target = staging / relative
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                np.save(stream, processed, allow_pickle=False)
            loaded = np.load(target, allow_pickle=False)
            require(
                np.array_equal(loaded, processed), "Serialized RGB construction equality failed"
            )
            require(
                sha(source) == input_sha and sha(fov_source) == fov_sha,
                "Source RGB/FOV changed during build",
            )
            records[pid] = {
                "path": relative,
                "sha256": sha(target),
                "bytes": target.stat().st_size,
                "shape": list(processed.shape),
                "dtype": "uint8",
                "split": required[pid],
                "source_alias_sha256": alias_sha,
                "aliases_decoded_RGB_exact": True,
                "source_image": photo["image"],
                "source_image_sha256": input_sha,
                "source_fov": photo["fov"],
                "source_fov_sha256": fov_sha,
                "outside_FOV_original_RGB_exact": True,
                "construction_exact": True,
            }
            require(
                all(sha(safe(manifest.parent, rel)) == value for rel, value in alias_sha.items()),
                "RGB alias changed during construction",
            )
            target.chmod(0o444)
        require(
            before
            == {
                "manifest": sha(manifest),
                "photos": sha(manifest.parent / "photos.json"),
                "operator": operator_fingerprint(),
                "builder": sha(__file__),
            },
            "Source/code/metadata changed during construction",
        )
        if not subset_proof:
            verify_prepared_content(manifest, expected_content_sha256)
        require(
            _training_source_sha256() == construction_training_source,
            "Training source changed during build",
        )
        index = {
            "schema": SCHEMA,
            "kind": "full_prepared_photometry_RGB",
            "canonical_fingerprint": expected_content_sha256,
            "manifest_sha256": before["manifest"],
            "photos_json_sha256": before["photos"],
            "operator_source_sha256": before["operator"],
            "builder_source_sha256": before["builder"],
            "construction_training_source_sha256": construction_training_source,
            "descriptor": descriptor,
            "descriptor_sha256": expected_descriptor_sha256,
            "coverage": "subset_proof" if subset_proof else "train_val",
            "whole_canonical_encoded_bytes_verified_before_after": not subset_proof,
            "eligible_photo_count": len(required),
            "photos": records,
        }
        index_bytes = canonical(index)
        require(len(index_bytes) <= MAX_INDEX_BYTES, "Cache index exceeds schema safety budget")
        (staging / "index.json").write_bytes(index_bytes)
        cache_fp = fingerprint(index_bytes)
        (staging / "_SUCCESS").write_text(cache_fp + "\n")
        for path in staging.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        staging.chmod(0o555)
        require(
            not output.exists() and not output.is_symlink(),
            "Destination appeared during build; preserve staging",
        )
        staging.rename(output)
        return {
            "root": str(output),
            "fingerprint": cache_fp,
            "descriptor_sha256": expected_descriptor_sha256,
            "coverage": index["coverage"],
            "photos": len(records),
            "RGB_file_bytes": sum(r["bytes"] for r in records.values()),
        }
    finally:
        os.close(fd)
        lock.unlink()


class VerifiedPhotometryCache:
    """Index identity at admission; immutable encoded bytes verified on every read."""

    def __init__(
        self,
        root,
        manifest,
        profile,
        *,
        expected_fingerprint,
        expected_canonical_fingerprint,
        allow_subset=False,
    ):
        self.root, self.manifest = Path(root).resolve(), Path(manifest).resolve()
        require(profile != "raw", "Never use processed RGB for a raw model")
        index_path = safe(self.root, "index.json")
        require(
            index_path.stat().st_size <= MAX_INDEX_BYTES, "Cache index exceeds schema safety budget"
        )
        require(safe(self.root, "_SUCCESS").stat().st_size < 128, "Invalid completion marker size")
        raw = index_path.read_bytes()
        require(
            fingerprint(raw) == expected_fingerprint
            and safe(self.root, "_SUCCESS").read_text().strip() == expected_fingerprint,
            "External cache fingerprint/completion differs",
        )
        self.index = json.loads(raw)
        index = self.index
        require(
            type(index["schema"]) is int
            and index["schema"] == SCHEMA
            and index["kind"] == "full_prepared_photometry_RGB",
            "Wrong cache type",
        )
        require(
            canonical(index["descriptor"]) == canonical(photometry.contract(profile))
            and index["descriptor_sha256"] == sha256(canonical(index["descriptor"])).hexdigest()
            and index["operator_source_sha256"] == operator_fingerprint(),
            "Photometry descriptor/backend/operator differs",
        )
        require(
            index["canonical_fingerprint"] == expected_canonical_fingerprint
            and index["manifest_sha256"] == sha(self.manifest)
            and index["photos_json_sha256"] == sha(self.manifest.parent / "photos.json"),
            "Canonical/manifest/FOV metadata identity differs",
        )
        required = eligible_photos(self.manifest)
        require(
            index["eligible_photo_count"] == len(required), "Eligible photo inventory count differs"
        )
        full = index["coverage"] == "train_val" and set(index["photos"]) == set(required)
        require(
            full
            or (
                allow_subset
                and index["coverage"] == "subset_proof"
                and set(index["photos"]) <= set(required)
            ),
            "Full TRAIN/VAL cache coverage required",
        )
        require(
            not full or index["whole_canonical_encoded_bytes_verified_before_after"] is True,
            "Full cache lacks canonical construction identity check",
        )
        photos = photometry.fov_index(self.manifest.parent)
        aliases = {}
        for row in json.loads(self.manifest.read_text()):
            if (
                row.get("photo_id") in index["photos"]
                and row["split"] in {"train", "val"}
                and row["task"] in {"vessels", "av", "disc_cup"}
                and row.get("usable_for_training", True)
                and not row.get("exclude_from_training", False)
            ):
                aliases.setdefault(row["photo_id"], set()).add(row["image"])
        for pid, record in index["photos"].items():
            photo = photos[pid]
            require(
                set(record["source_alias_sha256"]) == aliases[pid]
                and record["aliases_decoded_RGB_exact"] is True,
                "Source RGB alias construction binding differs",
            )
            require(
                record["shape"] == photo["prepared_shape"]
                and record["dtype"] == "uint8"
                and record["source_image"] == photo["image"]
                and record["source_fov"] == photo["fov"]
                and record["split"] == required[pid]
                and record["construction_exact"] is True
                and record["outside_FOV_original_RGB_exact"] is True,
                "Source/processed RGB mapping differs",
            )
        require(
            full or all(required[pid] == "train" for pid in index["photos"]),
            "Subset proofs are TRAIN-only",
        )
        self.profile = profile
        self.identity = {
            "schema": SCHEMA,
            "kind": "full_prepared_photometry_RGB",
            "fingerprint": expected_fingerprint,
            "canonical_fingerprint": expected_canonical_fingerprint,
            "descriptor_sha256": index["descriptor_sha256"],
            "operator_source_sha256": index["operator_source_sha256"],
            "coverage": index["coverage"],
            "fallback": False,
        }

    def require_coverage(self, rows):
        ids = {
            row["photo_id"]
            for row in rows
            if row["task"] in {"vessels", "av", "disc_cup"}
            and row.get("usable_for_training", True)
            and not row.get("exclude_from_training", False)
        }
        require(ids <= set(self.index["photos"]), "Missing photo coverage; never fallback")

    def verify_all_bytes(self):
        """Stream encoded bytes and inspect NPY headers only, not pixel arrays."""
        total = 0
        for record in self.index["photos"].values():
            path = safe(self.root, record["path"])
            require(
                path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                "Processed RGB encoded byte hash differs",
            )
            with path.open("rb") as stream:
                fmt = np.lib.format.read_magic(stream)
                require(fmt == (1, 0), "Expected ordinary NPY1.0 RGB export")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                require(
                    list(shape) == record["shape"]
                    and dtype == np.uint8
                    and not fortran
                    and stream.tell() + int(np.prod(shape)) == record["bytes"],
                    "RGB NPY header differs",
                )
            total += record["bytes"]
        return {
            "photos": len(self.index["photos"]),
            "RGB_file_bytes": total,
            "fingerprint": self.identity["fingerprint"],
            "encoded_byte_hashes_verified": True,
            "array_data_decoded": False,
        }

    def rgb(self, photo_id):
        require(photo_id in self.index["photos"], "Missing processed photo; no implicit fallback")
        record = self.index["photos"][photo_id]
        raw = safe(self.root, record["path"]).read_bytes()
        require(
            len(raw) == record["bytes"] and sha256(raw).hexdigest() == record["sha256"],
            "Processed RGB byte hash differs",
        )
        image = np.load(io.BytesIO(raw), allow_pickle=False)
        require(
            image.dtype == np.uint8
            and image.ndim == 3
            and image.shape[2] == 3
            and list(image.shape) == record["shape"]
            and record["dtype"] == "uint8",
            "Processed RGB dtype/shape differs",
        )
        return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=photometry.PROFILES[1:], required=True)
    parser.add_argument("--expected-content-sha256", required=True)
    parser.add_argument("--expected-descriptor-sha256", required=True)
    parser.add_argument(
        "--proof-photo-id",
        action="append",
        help="Explicit TRAIN-only subset proof; cannot train full_data",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.manifest,
                args.output,
                args.profile,
                expected_content_sha256=args.expected_content_sha256,
                expected_descriptor_sha256=args.expected_descriptor_sha256,
                photo_ids=args.proof_photo_id,
                subset_proof=args.proof_photo_id is not None,
            )
        )
    )


if __name__ == "__main__":
    main()
