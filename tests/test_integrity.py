"""Data identity must survive relocation and detect annotation changes on resume."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from deepretina.integrity import (
    FINGERPRINT_SCHEMA,
    prepared_content_fingerprint,
    verify_prepared_content,
)


def prepared(root):
    root.mkdir()
    Image.fromarray(np.full((8, 9, 3), 100, np.uint8)).save(root / "image.png")
    Image.fromarray(np.full((8, 9), 255, np.uint8)).save(root / "fov.png")
    np.savez_compressed(root / "mask.npz", target=np.zeros((1, 8, 9), np.float32))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps([{"id": "one", "image": "image.png", "mask": "mask.npz"}]))
    (root / "photos.json").write_text(
        json.dumps(
            [
                {
                    "image": "image.png",
                    "fov": "fov.png",
                    "geometry": {"shape": [8, 9]},
                }
            ]
        )
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "_SUCCESS").write_text(manifest_sha)
    value = prepared_content_fingerprint(manifest)
    (root / "integrity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "prepared_content_sha256": value,
                "fingerprint_schema": FINGERPRINT_SCHEMA,
            }
        )
    )
    return manifest, value


def test_prepared_identity_is_portable_across_paths_and_timestamps(tmp_path):
    manifest, expected = prepared(tmp_path / "original")
    destination = tmp_path / "relocated"
    shutil.copytree(manifest.parent, destination)
    os.utime(destination / "mask.npz", (1, 1))
    assert verify_prepared_content(destination / "manifest.json", expected) == expected


@pytest.mark.parametrize("name", ["image.png", "mask.npz", "fov.png", "photos.json"])
def test_changed_referenced_content_is_detected_even_with_same_timestamp(tmp_path, name):
    manifest, expected = prepared(tmp_path / "data")
    changed = manifest.parent / name
    old = changed.stat()
    contents = bytearray(changed.read_bytes())
    if name == "photos.json":
        metadata = json.loads(contents)
        metadata[0]["geometry"]["shape"][0] = 9
        changed.write_text(json.dumps(metadata))
    else:
        contents[-1] ^= 1
        changed.write_bytes(contents)
    os.utime(changed, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="content changed"):
        verify_prepared_content(manifest, expected)


def test_fresh_audit_does_not_silently_replace_frozen_search_data(tmp_path):
    manifest, frozen = prepared(tmp_path / "data")
    np.savez_compressed(manifest.parent / "mask.npz", target=np.ones((1, 8, 9), np.float32))
    audit_path = manifest.parent / "integrity.json"
    audit = json.loads(audit_path.read_text())
    audit["prepared_content_sha256"] = prepared_content_fingerprint(manifest)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="content changed"):
        verify_prepared_content(manifest, frozen)
