"""Explicit, byte-verified, lossless derivatives of an audited canonical corpus."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .integrity import verify_prepared_content

CACHE_SCHEMA = 1


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe(root, relative):
    relative = Path(relative)
    path = Path(root) / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError(f"Cache reference escapes its portable directory: {relative}")
    return path


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cache_fingerprint(root, *, index_bytes=None):
    """Read actual index and file bytes; file times and absolute paths are excluded."""
    root = Path(root)
    if index_bytes is None:
        index_bytes = (root / "index.json").read_bytes()
    index = json.loads(index_bytes)
    digest = hashlib.sha256(f"derived-cache:{CACHE_SCHEMA}\n".encode())
    digest.update(index_bytes)
    for relative, entry in sorted(index["files"].items()):
        path = _safe(root, relative)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Derived cache byte identity differs: {relative}")
        digest.update(json.dumps([relative, actual]).encode())
    return digest.hexdigest()


def _source_references(rows):
    result = {}
    for row in rows:
        result[row["image"]] = "rgb"
        result[row["mask"]] = "labels"
    return result


def select_photos(rows, count, seed=42):
    """Deterministic round-robin source coverage; exact photograph aliases merge."""
    by_source = defaultdict(set)
    for row in rows:
        by_source[row["dataset"]].add(row["photo_id"])
    if count < len(by_source):
        raise ValueError("Photo limit must allow all selected sources")
    choices = {
        source: sorted(ids, key=lambda value: _sha(f"{seed}:{value}".encode()))
        for source, ids in sorted(by_source.items())
    }
    selected = set()
    while len(selected) < count:
        previous = len(selected)
        for values in choices.values():
            remaining = next((value for value in values if value not in selected), None)
            if remaining is not None:
                selected.add(remaining)
                if len(selected) == count:
                    break
        if len(selected) == previous:
            break
    return selected


def _float_bits(array):
    return np.ascontiguousarray(array, dtype=np.float32).tobytes()


def build_cache(manifest, cache_base, *, splits=("train", "val"), max_photos=None, seed=42):
    """Create a new fingerprint-keyed cache; never write into the canonical tree."""
    manifest, cache_base = Path(manifest), Path(cache_base)
    splits = tuple(sorted(set(splits)))
    if not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("Select nonempty train/val/test splits explicitly")
    if max_photos is not None and max_photos < 1:
        raise ValueError("Photo limit must be positive")
    canonical = manifest.parent.resolve()
    if cache_base.resolve().is_relative_to(canonical):
        raise ValueError("Derived cache must be outside the canonical directory")
    canonical_fp = verify_prepared_content(manifest)
    all_rows = json.loads(manifest.read_text())
    rows = [row for row in all_rows if row["split"] in splits]
    if not rows:
        raise ValueError("Selected splits contain no records")
    all_ids = {row["photo_id"] for row in rows}
    selected = all_ids if max_photos is None else select_photos(rows, max_photos, seed)
    rows = [row for row in rows if row["photo_id"] in selected]
    partial = selected != all_ids
    scope = (
        "subset-" + _sha(json.dumps(sorted(selected)).encode())[:16]
        if partial
        else "-".join(sorted(splits))
    )
    output = cache_base / canonical_fp / f"v{CACHE_SCHEMA}-{scope}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new cache base; existing cache preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "rgb").mkdir()
    (output / "labels").mkdir()
    files, mapping = {}, {}
    checked = Counter()
    for source, kind in sorted(_source_references(rows).items()):
        payload = _safe(canonical, source).read_bytes()
        source_hash = _sha(payload)
        relative = f"{kind}/{source_hash}." + ("npy" if kind == "rgb" else "npz")
        encoded = io.BytesIO()
        if kind == "rgb":
            with Image.open(io.BytesIO(payload)) as image:
                original = np.asarray(image.convert("RGB")).copy()
            np.save(encoded, original, allow_pickle=False)
            restored = np.load(io.BytesIO(encoded.getvalue()), allow_pickle=False)
            if original.dtype != restored.dtype or original.tobytes() != restored.tobytes():
                raise ValueError(f"RGB construction equality failed: {source}")
            encoding = "uint8-hwc-npy"
            checked["rgb_exact"] += 1
        else:
            with np.load(io.BytesIO(payload), allow_pickle=False) as labels:
                original = {key: labels[key] for key in labels.files}
            target = original["target"]
            if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
                raise ValueError(f"Invalid canonical target: {source}")
            # Signed zero must retain its float representation too.
            binary = bool(((target == 0) | (target == 1)).all()) and not (
                np.issubdtype(target.dtype, np.floating) and np.signbit(target).any()
            )
            derived = dict(original)
            derived["target"] = target.astype(bool) if binary else target
            derived["valid"] = original["valid"].astype(bool)
            if "fov" in original:
                derived["fov"] = original["fov"].astype(bool)
            np.savez_compressed(encoded, **derived)
            with np.load(io.BytesIO(encoded.getvalue()), allow_pickle=False) as restored:
                equal = _float_bits(original["target"]) == _float_bits(restored["target"])
                equal &= np.array_equal(original["valid"].astype(bool), restored["valid"])
                if "fov" in original:
                    equal &= np.array_equal(original["fov"].astype(bool), restored["fov"])
                for key in original.keys() - {"target", "valid", "fov"}:
                    equal &= np.array_equal(original[key], restored[key])
                if not equal:
                    raise ValueError(f"Label construction equality failed: {source}")
            encoding = "binary-bool-npz" if binary else "preserved-soft-npz"
            checked[encoding] += 1
            checked["labels_exact"] += 1
        blob = encoded.getvalue()
        if relative not in files:
            _safe(output, relative).write_bytes(blob)
            files[relative] = {"sha256": _sha(blob), "bytes": len(blob), "kind": kind}
        mapping[source] = {
            "derived": relative,
            "kind": kind,
            "source_sha256": source_hash,
            "encoding": encoding,
        }
    index = {
        "schema": CACHE_SCHEMA,
        "canonical_fingerprint": canonical_fp,
        "canonical_manifest_sha256": _sha(manifest.read_bytes()),
        "coverage": {
            "kind": "subset" if partial else "selected-splits",
            "splits": sorted(splits),
            "photos": len(selected),
            "records": len(rows),
            "photo_ids": sorted(selected),
        },
        "files": files,
        "source_to_derived": mapping,
        "construction": {"exact_equality_verified": True, **dict(checked)},
    }
    _json(output / "index.json", index)
    if verify_prepared_content(manifest, canonical_fp) != canonical_fp:
        raise ValueError("Canonical corpus changed while building cache")
    fingerprint = cache_fingerprint(output)
    (output / "_SUCCESS").write_text(fingerprint + "\n")
    return {
        "root": str(output),
        "cache_fingerprint": fingerprint,
        "canonical_fingerprint": canonical_fp,
        "coverage": index["coverage"],
        "bytes": sum(value["bytes"] for value in files.values()),
        "construction": index["construction"],
    }


class VerifiedCache:
    """Pinned cache; every derived byte stream is checked before decoding."""

    def __init__(
        self,
        root,
        manifest,
        *,
        expected_cache_fingerprint,
        expected_canonical_fingerprint,
        allow_canonical_fallback=False,
    ):
        if not expected_cache_fingerprint or not expected_canonical_fingerprint:
            raise ValueError("Cache access requires both externally pinned fingerprints")
        self.root, self.manifest = Path(root), Path(manifest)
        self.canonical_root = self.manifest.parent.resolve()
        if (
            verify_prepared_content(self.manifest, expected_canonical_fingerprint)
            != expected_canonical_fingerprint
        ):
            raise ValueError("Canonical fingerprint differs")
        index_bytes = (self.root / "index.json").read_bytes()
        self.index = json.loads(index_bytes)
        if (
            self.index["schema"] != CACHE_SCHEMA
            or self.index["canonical_fingerprint"] != expected_canonical_fingerprint
            or self.index["canonical_manifest_sha256"] != _sha(self.manifest.read_bytes())
        ):
            raise ValueError("Cache schema/canonical identity differs")
        actual = cache_fingerprint(self.root, index_bytes=index_bytes)
        if (
            actual != expected_cache_fingerprint
            or (self.root / "_SUCCESS").read_text().strip() != actual
        ):
            raise ValueError("Cache fingerprint or completion marker differs")
        self.fingerprint = actual
        self.allow_canonical_fallback = bool(allow_canonical_fallback)
        self.stats = Counter()

    @property
    def identity(self):
        return {
            "schema": CACHE_SCHEMA,
            "fingerprint": self.fingerprint,
            "canonical_fingerprint": self.index["canonical_fingerprint"],
            "allow_canonical_fallback": self.allow_canonical_fallback,
        }

    def require_coverage(self, rows):
        required = _source_references(rows)
        missing = set(required) - set(self.index["source_to_derived"])
        if missing:
            raise ValueError(f"Derived cache lacks {len(missing)} required source references")
        for source, kind in required.items():
            if self.index["source_to_derived"][source]["kind"] != kind:
                raise ValueError(f"Cache source kind differs: {source}")

    def _blob(self, source, kind):
        _safe(self.canonical_root, source)
        entry = self.index["source_to_derived"].get(source)
        if entry is None:
            if not self.allow_canonical_fallback:
                raise ValueError(f"Source is absent from pinned cache: {source}")
            self.stats[kind + "_canonical_fallback"] += 1
            return None
        if entry["kind"] != kind:
            raise ValueError(f"Wrong cache source kind: {source}")
        relative = entry["derived"]
        payload = _safe(self.root, relative).read_bytes()
        if _sha(payload) != self.index["files"][relative]["sha256"]:
            raise ValueError(f"Derived cache file changed: {relative}")
        self.stats[kind + "_hits"] += 1
        return payload

    def rgb(self, source):
        payload = self._blob(source, "rgb")
        if payload is None:
            with Image.open(_safe(self.canonical_root, source)) as image:
                return np.asarray(image.convert("RGB")).copy()
        image = np.load(io.BytesIO(payload), allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Derived image must remain uint8 RGB")
        return image

    def labels(self, source):
        payload = self._blob(source, "labels")
        return np.load(
            _safe(self.canonical_root, source) if payload is None else io.BytesIO(payload),
            allow_pickle=False,
        )
