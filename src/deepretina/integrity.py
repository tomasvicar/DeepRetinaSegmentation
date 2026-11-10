"""Portable byte-level identity of canonical prepared photographs and annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FINGERPRINT_SCHEMA = 2


def prepared_content_fingerprint(manifest: Path) -> str:
    """Hash every referenced input, including photo geometry and standalone FOVs.

    Hardlinked aliases share one streaming read. Paths and bytes determine the
    result; inode numbers and timestamps do not, so relocated copies agree.
    """
    manifest = Path(manifest)
    root = manifest.parent.resolve()
    rows = json.loads(manifest.read_text())
    photos_path = root / "photos.json"
    photos = json.loads(photos_path.read_text())
    paths = {manifest.name, "photos.json"}
    for row in rows:
        paths.update((row["image"], row["mask"]))
    for photo in photos:
        paths.update((photo["image"], photo["fov"]))
    result = hashlib.sha256(f"prepared-data-schema:{FINGERPRINT_SCHEMA}\n".encode())
    cache = {}
    for relative in sorted(paths):
        candidate = Path(relative)
        path = root / candidate
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(
                f"Prepared inputs must remain inside their portable directory: {relative}"
            )
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if key not in cache:
            with path.open("rb") as source:
                cache[key] = hashlib.file_digest(source, "sha256").hexdigest()
        after = path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != key:
            raise ValueError(f"Prepared input changed while checking its identity: {relative}")
        result.update(json.dumps([candidate.as_posix(), cache[key]]).encode())
        result.update(b"\n")
    return result.hexdigest()


def verify_prepared_content(manifest: Path, expected: str | None = None) -> str:
    """Require a completed audited preparation and re-read its actual input bytes."""
    manifest = Path(manifest)
    audit = json.loads((manifest.parent / "integrity.json").read_text())
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (manifest.parent / "_SUCCESS").read_text().strip() != manifest_sha:
        raise ValueError("Prepared manifest completion marker differs")
    if audit.get("manifest_sha256") != manifest_sha:
        raise ValueError("Prepared manifest differs from its integrity audit")
    if audit.get("fingerprint_schema") != FINGERPRINT_SCHEMA:
        raise ValueError("Re-run the prepared-data audit for fingerprint schema2")
    current = prepared_content_fingerprint(manifest)
    if current != audit.get("prepared_content_sha256") or (
        expected is not None and current != expected
    ):
        raise ValueError("Prepared image/annotation content changed after audit or search planning")
    return current
