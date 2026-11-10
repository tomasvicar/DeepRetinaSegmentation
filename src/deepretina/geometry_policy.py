"""Portable, explicitly declared corpus-v2 nominal camera rules.

No new camera angles are legacy defaults. The checkpoint carries the original
policy JSON bytes and their manifest-bound SHA; no policy file is read at inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

NEW_SOURCES = frozenset({"drions", "gamma", "idrid_disc", "biomisa"})
MAX_POLICY_BYTES = 65536
SHAPE_ORDER = "native_height_x_width_before_any_resize_or_rotation"
UNSEEN_ACTION = "use_explicit_nominal_default_and_flag_unseen_shape"


def _angle(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 180
    ):
        raise ValueError("Geometry policy requires a finite angle in (0,180]")
    return float(value)


def _shape(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(x) is not int or x <= 0 for x in value)
    ):
        raise ValueError("Geometry policy requires original native integer H,W")
    return list(value)


def _rule(rule):
    if not isinstance(rule, dict) or not {
        "field_of_view_degrees",
        "nominal_assumption",
        "per_image_angle_measured",
    } <= set(rule):
        raise ValueError("Incomplete geometry policy rule")
    _angle(rule["field_of_view_degrees"])
    if (
        type(rule["nominal_assumption"]) is not bool
        or rule["per_image_angle_measured"] is not False
    ):
        raise ValueError(
            "Geometry policy must retain nominal/reported, not per-image-measured qualifications"
        )
    for key in ("status", "camera", "qualification"):
        if not isinstance(rule.get(key), str) or not rule[key]:
            raise ValueError("Missing geometry qualification: " + key)
    for key in ("sources", "evidence_ids"):
        if (
            not isinstance(rule.get(key), list)
            or not rule[key]
            or any(not isinstance(x, str) or not x for x in rule[key])
        ):
            raise ValueError("Missing geometry evidence: " + key)


def validate_geometry_policy(embedded):
    """Validate both portable-wrapper and policy-document versions and byte hash."""
    if (
        not isinstance(embedded, dict)
        or set(embedded) != {"format_version", "sha256", "document_json"}
        or type(embedded["format_version"]) is not int
        or embedded["format_version"] != 1
    ):
        raise ValueError("Unsupported embedded geometry policy format")
    if not isinstance(embedded["document_json"], str):
        raise ValueError("Embedded geometry policy must be UTF-8 JSON text")  # noqa: TRY004 - malformed checkpoint contract
    if not isinstance(embedded["sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", embedded["sha256"]
    ):
        raise ValueError("Geometry policy requires a SHA256 identity")
    raw = embedded["document_json"].encode("utf-8")
    if len(raw) > MAX_POLICY_BYTES or hashlib.sha256(raw).hexdigest() != embedded["sha256"]:
        raise ValueError("Geometry policy byte hash/size differs")

    def invalid_constant(value):
        raise ValueError("Nonfinite geometry JSON constant: " + value)

    document = json.loads(raw, parse_constant=invalid_constant)
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("target_pixels_per_degree") != 25
        or document.get("physical_measurement_claim") is not False
    ):
        raise ValueError("Unsupported geometry policy schema/physical coordinate contract")
    if not isinstance(document.get("policy_id"), str) or not document["policy_id"]:
        raise ValueError("Geometry policy ID is missing")
    policies = document.get("policies", {})
    if not isinstance(policies, dict) or not policies or not set(policies) <= NEW_SOURCES:
        raise ValueError("Geometry policy may declare only the new corpus-v2 source names")
    for dataset, entry in policies.items():
        if not isinstance(entry, dict):
            raise ValueError("Geometry policy entry must be a rule object")  # noqa: TRY004 - malformed policy contract
        if dataset == "gamma":
            if (
                entry.get("shape_key_order") != SHAPE_ORDER
                or entry.get("unseen_shape_action") != UNSEEN_ACTION
                or not isinstance(entry.get("per_shape"), dict)
                or not entry.get("per_shape")
                or "default" not in entry
            ):
                raise ValueError("GAMMA requires exact native HxW rules and declared fallback")
            for shape, rule in entry["per_shape"].items():
                if not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", shape):
                    raise ValueError("Invalid native HxW geometry rule")
                _rule(rule)
            _rule(entry["default"])
        else:
            _rule(entry)
    return document


def geometry_rule(embedded, dataset, native_shape):
    """Same native-shape-first selection contract as prepare_corpus_v2.angle_for."""
    document = validate_geometry_policy(embedded)
    shape = _shape(native_shape)
    if dataset not in document["policies"]:
        raise ValueError("Dataset has no declared checkpoint geometry rule")
    entry = copy.deepcopy(document["policies"][dataset])
    if "per_shape" in entry:
        key = f"{shape[0]}x{shape[1]}"
        matched = key in entry["per_shape"]
        entry = entry["per_shape"][key] if matched else entry["default"]
        method = "native_shape_rule" if matched else "explicit_unseen_shape_nominal_fallback"
    else:
        method = "dataset_acquisition_description_or_explicit_nominal_research_policy"
    entry.update(
        policy_id=document["policy_id"], native_shape_hw=shape, camera_attribution_method=method
    )
    return entry


def capture_geometry_policy(root, rows):
    """Bind prepared manifest metadata to its sidecar before training/device setup."""
    declared = []
    for row in rows:
        metadata = row.get("metadata", {})
        sha = metadata.get("geometry_policy_sha256")
        if row["dataset"] in NEW_SOURCES or sha is not None:
            if (
                not isinstance(sha, str)
                or not re.fullmatch("[0-9a-f]{64}", sha)
                or metadata.get("policy_sha256") != sha
            ):
                raise ValueError("New-source training row lacks matching geometry policy hashes")
            declared.append(row)
    if not declared:
        return None  # v1 retains its behavior even if an unrelated sidecar exists.
    hashes = {r["metadata"]["geometry_policy_sha256"] for r in declared}
    if len(hashes) != 1:
        raise ValueError("Training manifest mixes geometry policy identities")
    sidecar = Path(root) / "geometry-policy.json"
    if sidecar.stat().st_size > MAX_POLICY_BYTES:
        raise ValueError("Geometry policy sidecar exceeds portable size bound")
    raw = sidecar.read_bytes()
    embedded = {
        "format_version": 1,
        "sha256": next(iter(hashes)),
        "document_json": raw.decode("utf-8"),
    }
    validate_geometry_policy(embedded)
    for row in declared:
        metadata = row["metadata"]
        cal = metadata["physical_calibration"]
        shape = _shape(cal["native_shape_hw"])
        if metadata["original_geometry"]["original_shape"] != shape:
            raise ValueError("Native geometry shape differs from recorded calibration")
        rule = geometry_rule(embedded, row["dataset"], shape)
        if (
            any(cal.get(k) != v for k, v in rule.items())
            or row["field_of_view_degrees"] != rule["field_of_view_degrees"]
        ):
            raise ValueError("Training row geometry differs from its pinned policy rule")
    return embedded


def prediction_geometry(embedded, dataset, native_shape, explicit_angle=None):
    """Return effective explicit angle and honest policy/override provenance."""
    if embedded is None:
        return explicit_angle, None  # old checkpoints: unchanged legacy behavior.
    document = validate_geometry_policy(embedded)
    name = dataset.lower() if dataset else None
    meta = {
        "policy_id": document["policy_id"],
        "policy_sha256": embedded["sha256"],
        "dataset": name,
        "native_shape_hw": _shape(native_shape),
        "policy_applied": False,
    }
    if name not in document["policies"]:
        meta["selection"] = "outside_new_source_scope; existing_legacy_or_explicit_angle"
        return explicit_angle, meta
    rule = geometry_rule(embedded, name, native_shape)
    meta["declared_rule"] = rule
    meta["unseen_native_shape"] = (
        rule["camera_attribution_method"] == "explicit_unseen_shape_nominal_fallback"
    )
    if explicit_angle is not None:
        meta.update(
            selection="explicit_field_of_view_degrees_override",
            field_of_view_degrees=_angle(explicit_angle),
            override_measurement_claim=False,
            unseen_native_shape_fallback=False,
        )
        return float(explicit_angle), meta
    meta.update(
        selection="checkpoint_recorded_geometry_policy",
        policy_applied=True,
        field_of_view_degrees=rule["field_of_view_degrees"],
        unseen_native_shape_fallback=rule["camera_attribution_method"]
        == "explicit_unseen_shape_nominal_fallback",
    )
    return rule["field_of_view_degrees"], meta
