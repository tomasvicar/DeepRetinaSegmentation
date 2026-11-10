"""Explicit byte-identity bridge for an unchanged VAL domain across corpus versions.

Pure stdlib. Encoded RGB/NPZ/FOV bytes are hashed, never decoded or scored. This
scoped proof does not re-audit TRAIN/TEST or certify a global canonical identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

KIND = "val_domain_byte_identity_v1"
WRAPPER = "val_domain_recontextualized_original_reference"
CHANNEL_TASKS = {
    "vessel": {"vessels", "av"},
    "artery": {"av"},
    "vein": {"av"},
    "disc": {"disc_cup"},
    "cup": {"disc_cup"},
}


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _state(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _path(root, relative):
    p = Path(relative)
    require(
        not p.is_absolute() and p.parts and ".." not in p.parts,
        "Require portable relative asset paths",
    )
    result = root / p
    require(result.resolve().is_relative_to(root), "Asset escaped corpus root")
    current = result
    while current != root:
        require(not current.is_symlink(), "Symlink assets/directories are not bridgeable")
        current = current.parent
    require(result.is_file(), "Missing scoped VAL asset")
    return result


def _file_sha(path, expected_state=None):
    before = _state(path)
    if expected_state is not None:
        require(before == expected_state, "VAL asset changed during bridge check")
    with path.open("rb") as stream:
        s = os.fstat(stream.fileno())
        opened = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(opened == before, "Asset replaced while opening")
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    require(_state(path) == before, "Asset changed while hashing")
    return result


def _root_snapshot(directory, expected_content, expected_photos):
    root = Path(directory).resolve(strict=True)
    require(_sha(expected_content), "Require independently pinned canonical audit identity")
    raw = {
        name: _path(root, name).read_bytes()
        for name in ("manifest.json", "photos.json", "integrity.json", "_SUCCESS")
    }
    manifest_sha = digest(raw["manifest.json"])
    audit = json.loads(raw["integrity.json"])
    require(
        raw["_SUCCESS"].decode().strip() == manifest_sha
        and audit.get("manifest_sha256") == manifest_sha,
        "Incomplete audited manifest",
    )
    require(
        audit.get("fingerprint_schema") == 2
        and audit.get("prepared_content_sha256") == expected_content,
        "Expected schema2 canonical audit pin differs",
    )
    document = json.loads(raw["manifest.json"])
    rows = document["records"] if isinstance(document, dict) else document
    require(isinstance(rows, list), "Manifest records must be a list")
    ids, partitions = set(), {}
    for row in rows:
        require(row["id"] not in ids, "Duplicate manifest annotation ID")
        ids.add(row["id"])
        require(row["split"] in {"train", "val", "test"}, "Unsupported partition")
        official = row.get("metadata", {}).get("official_split")
        require(official != "test" or row["split"] == "test", "Official TEST membership violated")
        require(official != "val" or row["split"] != "train", "Official VAL used for training")
        train_only = (
            row.get("metadata", {}).get("split_policy") == "train_only_without_patient_crosswalk"
        )
        require(
            not train_only or row["split"] == "train",
            "Train-only unknown-patient source moved to heldout",
        )
        for key in ("photo_id", "source_id", "split_group_id"):
            ident = row.get(key)
            require(
                isinstance(ident, str) and bool(ident),
                "Missing explicit photo/source/group identity",
            )
            pair = key, ident
            require(
                partitions.setdefault(pair, row["split"]) == row["split"],
                "Identity crosses corpus partitions",
            )
    selected = sorted((r for r in rows if r["split"] == "val"), key=lambda r: r["id"])
    photo_ids = {row["photo_id"] for row in selected}
    require(len(photo_ids) == expected_photos, "Require the exact intended VAL photo count")
    photos = json.loads(raw["photos.json"])
    index = {p["photo_id"]: p for p in photos}
    require(len(index) == len(photos), "Duplicate photograph index identity")
    require(
        {pid for pid, photo in index.items() if photo.get("split") == "val"} == photo_ids,
        "Extra or missing VAL photo index membership",
    )
    photo_rows = []
    for pid in sorted(photo_ids):
        require(pid in index and index[pid]["split"] == "val", "VAL photo index mismatch")
        photo_rows.append(index[pid])
        records = {r["id"] for r in selected if r["photo_id"] == pid}
        require(
            len(index[pid]["annotations"]) == len(records)
            and records == {r["id"] for r in index[pid]["annotations"]},
            "Photo observer/record inventory differs",
        )
    paths = {r[key] for r in selected for key in ("image", "mask")}
    paths.update(p[key] for p in photo_rows for key in ("image", "fov"))
    assets = {rel: _path(root, rel) for rel in sorted(paths)}
    return {
        "root": root,
        "raw": raw,
        "manifest_sha256": manifest_sha,
        "canonical_sha256": expected_content,
        "rows": selected,
        "photos": photo_rows,
        "assets": assets,
    }


def _domain(snapshot):
    return {
        "manifest_sha256": snapshot["manifest_sha256"],
        "prepared_content_sha256": snapshot["canonical_sha256"],
        "photos_json_sha256": digest(snapshot["raw"]["photos.json"]),
        "integrity_json_sha256": digest(snapshot["raw"]["integrity.json"]),
    }


def _stable_metadata(snapshot):
    for name, raw in snapshot["raw"].items():
        require(
            (snapshot["root"] / name).read_bytes() == raw,
            "Corpus metadata changed during bridge check",
        )


def _reference(reference, snapshot, expected_photos):
    require(
        reference.get("family") in {"original_reference", "optional_fresh_paired_cup_experiment"},
        "Require a declared original reference family",
    )
    require(
        reference.get("split") == "val"
        and reference.get("preview_only") is False
        and reference.get("photos") == expected_photos,
        "Only complete VAL references are bridgeable",
    )
    require(
        reference.get("manifest_sha256") == snapshot["manifest_sha256"]
        and reference.get("prepared_content_sha256") == snapshot["canonical_sha256"],
        "Measured reference does not belong to parent corpus",
    )
    entries = reference["per_photo"]
    lookup = {r["photo_id"]: r for r in entries}
    pids = {r["photo_id"] for r in snapshot["rows"]}
    require(
        len(lookup) == len(entries) and set(lookup) == pids,
        "Reference VAL photograph inventory differs",
    )
    for pid, entry in lookup.items():
        rows = [r for r in snapshot["rows"] if r["photo_id"] == pid]
        require(
            len(entry["record_ids"]) == len(rows)
            and set(entry["record_ids"]) == {r["id"] for r in rows},
            "Measured reference observer/record domain differs",
        )
        for head, source in entry["channel_datasets"].items():
            require(
                head in CHANNEL_TASKS
                and any(r["dataset"] == source and r["task"] in CHANNEL_TASKS[head] for r in rows),
                "Measured reference source/head domain is ineligible",
            )


def build_bridge(
    parent,
    target,
    original_reference,
    *,
    expected_parent_content,
    expected_target_content,
    expected_photos=770,
):
    """Prove VAL identity; production CLI fixes expected_photos=770."""
    left = _root_snapshot(parent, expected_parent_content, expected_photos)
    right = _root_snapshot(target, expected_target_content, expected_photos)
    require(left["root"] != right["root"], "Require distinct corpus contexts")
    require(
        encoded(left["rows"]) == encoded(right["rows"]),
        "VAL record/source/group/preparation/observer metadata changed",
    )
    require(
        encoded(left["photos"]) == encoded(right["photos"]),
        "VAL photograph geometry/FOV/source metadata changed",
    )
    raw_reference = Path(original_reference).read_bytes()
    reference = json.loads(raw_reference)
    _reference(reference, left, expected_photos)
    states = [{rel: _state(path) for rel, path in s["assets"].items()} for s in (left, right)]
    require(set(states[0]) == set(states[1]), "VAL asset inventory changed")
    inventory = []
    for relative in states[0]:
        hashes = [
            _file_sha(s["assets"][relative], expected_state=st[relative])
            for s, st in zip((left, right), states, strict=True)
        ]
        require(hashes[0] == hashes[1], "VAL RGB/mask/FOV bytes changed: " + relative)
        inventory.append({"path": relative, "bytes": states[0][relative][2], "sha256": hashes[0]})
    for snapshot, state in zip((left, right), states, strict=True):
        require(
            all(_state(snapshot["assets"][rel]) == value for rel, value in state.items()),
            "VAL assets changed during whole bridge operation",
        )
        _stable_metadata(snapshot)
    require(
        Path(original_reference).read_bytes() == raw_reference,
        "Original measured reference changed during bridge check",
    )
    bridge = {
        "schema_version": 1,
        "kind": KIND,
        "scope": "val",
        "photo_count": expected_photos,
        "record_count": len(left["rows"]),
        "split_group_count": len({r["split_group_id"] for r in left["rows"]}),
        "parent": _domain(left),
        "target": _domain(right),
        "original_measured_report_sha256": digest(raw_reference),
        "original_measured_report_semantic_sha256": digest(encoded(reference)),
        "val_rows_sha256": digest(encoded(left["rows"])),
        "val_photo_metadata_sha256": digest(encoded(left["photos"])),
        "val_asset_inventory_sha256": digest(encoded(inventory)),
        "val_assets": inventory,
        "new_forward": False,
        "assets_decoded_or_scored": False,
        "test_bridge_allowed": False,
        "global_identity_scope": "Expected independently audited schema2 canonical pins are bound; only VAL assets are rehashed, no TRAIN/TEST certification",
        "stability_scope": "Before/after asset identity and metadata checks detect changes during verification; no guarantee against future storage changes",
        "verifier_code_sha256": digest(Path(__file__).read_bytes()),
    }
    return bridge, wrap_reference(reference, bridge)


def wrap_reference(original, bridge):
    """Keep original measurement body/pins untouched inside a typed wrapper."""
    return {
        "schema_version": 1,
        "family": WRAPPER,
        "scope": "val",
        "bridge_sha256": digest(encoded(bridge)),
        "parent_domain": copy.deepcopy(bridge["parent"]),
        "target_domain": copy.deepcopy(bridge["target"]),
        "original_measured_report_sha256": bridge["original_measured_report_sha256"],
        "new_forward": False,
        "original_measured_reference": copy.deepcopy(original),
    }


def validate_target(bridge, target, *, expected_photos=770):
    require(
        bridge.get("schema_version") == 1
        and bridge.get("kind") == KIND
        and bridge.get("scope") == "val",
        "Unknown bridge type/scope",
    )
    require(
        bridge.get("photo_count") == expected_photos
        and bridge.get("test_bridge_allowed") is False
        and bridge.get("new_forward") is False
        and bridge.get("assets_decoded_or_scored") is False,
        "Only unchanged complete VAL is authorized",
    )
    snapshot = _root_snapshot(target, bridge["target"]["prepared_content_sha256"], expected_photos)
    require(
        _domain(snapshot) == bridge["target"], "Target global context differs from trusted bridge"
    )
    require(
        digest(encoded(snapshot["rows"])) == bridge["val_rows_sha256"]
        and digest(encoded(snapshot["photos"])) == bridge["val_photo_metadata_sha256"],
        "Target VAL metadata differs from bridge",
    )
    inventory = bridge["val_assets"]
    require(
        digest(encoded(inventory)) == bridge["val_asset_inventory_sha256"],
        "Bridge inventory hash differs",
    )
    require(
        len(inventory) == len(snapshot["assets"])
        and {r["path"] for r in inventory} == set(snapshot["assets"]),
        "Trusted bridge asset inventory differs",
    )
    states = {rel: _state(path) for rel, path in snapshot["assets"].items()}
    for row in inventory:
        path = snapshot["assets"][row["path"]]
        require(
            _file_sha(path, states[row["path"]]) == row["sha256"]
            and states[row["path"]][2] == row["bytes"],
            "Target VAL asset bytes differ from trusted bridge",
        )
    require(
        all(_state(snapshot["assets"][rel]) == state for rel, state in states.items()),
        "Target VAL assets changed during verification",
    )
    _stable_metadata(snapshot)
    return snapshot


def resolve_reference(
    wrapper_path,
    bridge_path,
    trusted_bridge_sha256,
    original_reference,
    target,
    *,
    expected_photos=770,
):
    """Explicit trusted bridge opt-in; a filename or wrapper's own hash is insufficient."""
    require(_sha(trusted_bridge_sha256), "External trusted bridge SHA256 is required")
    raw_bridge = Path(bridge_path).read_bytes()
    require(
        digest(raw_bridge) == trusted_bridge_sha256, "Bridge differs from external trusted SHA256"
    )
    bridge = json.loads(raw_bridge)
    require(raw_bridge == encoded(bridge), "Require canonical reviewed bridge bytes")
    raw_original = Path(original_reference).read_bytes()
    require(
        digest(raw_original) == bridge["original_measured_report_sha256"],
        "Original measurement artifact SHA differs",
    )
    original = json.loads(raw_original)
    require(
        digest(encoded(original)) == bridge["original_measured_report_semantic_sha256"],
        "Original measurement content differs",
    )
    wrapper = json.loads(Path(wrapper_path).read_bytes())
    require(
        encoded(wrapper) == encoded(wrap_reference(original, bridge)),
        "Recontextualized reference differs from exact approved wrapper",
    )
    snapshot = validate_target(bridge, target, expected_photos=expected_photos)
    parent_view = {
        **snapshot,
        "manifest_sha256": bridge["parent"]["manifest_sha256"],
        "canonical_sha256": bridge["parent"]["prepared_content_sha256"],
    }
    _reference(original, parent_view, expected_photos)
    # Derive only an internal strict-comparison view after the explicit proof.
    result = copy.deepcopy(original)
    result.update(
        manifest_sha256=snapshot["manifest_sha256"],
        prepared_content_sha256=snapshot["canonical_sha256"],
    )
    provenance = {
        "kind": KIND,
        "trusted_bridge_sha256": trusted_bridge_sha256,
        "original_measured_report_sha256": digest(raw_original),
        "reference_context_artifact_sha256": digest(Path(wrapper_path).read_bytes()),
        "parent_domain": bridge["parent"],
        "target_domain": bridge["target"],
        "new_forward": False,
        "scope": "val",
    }
    return result, provenance


def publish(directory, bridge, wrapper):
    """Publish a new artifact directory; never overwrite an earlier reference."""
    directory = Path(directory)
    require(not directory.exists(), "Choose a new bridge output directory")
    directory.mkdir(parents=True)
    for name, value in (("bridge.json", bridge), ("reference-for-target.json", wrapper)):
        with (directory / name).open("xb") as handle:
            handle.write(encoded(value))
    return {
        "bridge_sha256": digest(encoded(bridge)),
        "reference_for_target_sha256": digest(encoded(wrapper)),
        "scope": "val",
        "new_forward": False,
    }
