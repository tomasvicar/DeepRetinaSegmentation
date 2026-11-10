"""Fixed strongest-original VAL reproduction and separately reviewed TEST execution.

This research evaluator never changes original prediction defaults or old VAL reports.
Historical training exposure and raw photometry residuals remain unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from compose_original_disc_reference import aggregate
from evaluate_original import confusion, geometry, raw_photo
from PIL import Image

from deepretina import selection as gate
from deepretina.geometry_policy import capture_geometry_policy
from deepretina.inference import resolve_device
from deepretina.integrity import verify_prepared_content
from deepretina.legacy.av import load_av, predict_av_input, restore_classes
from deepretina.legacy.checkpoints import load_single_net
from deepretina.legacy.matlab_av import prepare_vesclass_matlab_calibrated
from deepretina.legacy.matlab_fov import calibrate_provided_fov
from deepretina.legacy.single_net import tiled_logits
from deepretina.unified import CHANNELS, SOURCE_CHANNELS, UnifiedConfig, UnifiedDataset, _metrics

MODEL_SHA = {
    "universal": "9fa4f9ffddb2fa39828f32bde85299ffaff74b7b8817a70c70243fa11362569f",
    "vessel": "dae07480ec048aa8bb065c40d1dc07a3b9eb62b8e2f5ef9e66fe5acc70130d01",
    "disc": "7d6ddc3bf144e970cc7f5e011e834e241b8d3762464aeacf08d6a87287cdc51f",
    "av": "824289039bad6c4c4bdc05ca289e157813c5307aaf3d31b49775499e227f270b",
}
POLICY = {
    "name": "original-raw-vessel-calibrated-provided-fov-av-disc0-universal-cup-in-disc0-v1",
    "single_net": "raw RGB/255 minus0.5; native checkpoint tiling; strict logits>0",
    "cup": "universal/raw cup>0 intersect SAME-forward dedicated disc0/raw>0 in canonical frame before any NN restoration",
    "av": "vesclass_matlab_provided_fov_calibrated; canonical predicted vessel and provided FOV NN-restored to native frame; strict AV ties→0; original NN restore_classes then canonical prepare_mask",
    "gt": "UnifiedDataset full_data: one photo once; deterministic first sorted annotation per task; AV vessel GT priority; unknown validity excluded",
    "model_sha256": MODEL_SHA,
}
LIMITATIONS = [
    "Historical original photograph-to-training exposure cannot be reconstructed; no prior blindness is proven",
    "Raw/normalized original training provenance unresolved; this declared selected pipeline uses raw single_net inputs",
    "Provided FOV is canonical FOV restored to raw geometry, not recovered historical native FOV",
    "Calibrated geometric primitives have MATLAB evidence; raw photometry retains documented residuals",
    "Fresh backend forwards may drift from historical composed VAL counts; actual measured VAL reproduction must be reported",
    "Common canonical prepared geometry, not original publication coordinates; no independent clinical claim",
]


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    return gate.json_snapshot(path)[1]


def source_identity():
    root = Path(__file__).parent
    names = (
        "evaluate_original_frozen.py",
        "evaluate_original.py",
        "compose_original_disc_reference.py",
    )
    return {
        "runtime": gate.evaluation_source_identity(),
        "scripts": {name: sha_bytes((root / name).read_bytes()) for name in names},
    }


def metadata_identity(manifest):
    """Read JSON metadata only; never open image/NPZ inputs here."""
    manifest = Path(manifest)
    rows = read_json(manifest)
    photos = read_json(manifest.parent / "photos.json")
    audit = read_json(manifest.parent / "integrity.json")
    manifest_sha = sha_bytes(manifest.read_bytes())
    if audit["manifest_sha256"] != manifest_sha or audit["fingerprint_schema"] != 2:
        raise ValueError("Manifest integrity metadata differs")
    photo_lookup = {p["photo_id"]: p for p in photos}
    if len(photo_lookup) != len(photos):
        raise ValueError("Duplicate photo metadata")
    groups = {split: {} for split in ("train", "val", "test")}
    identities, record_ids = {}, set()
    for row in rows:
        if row["id"] in record_ids:
            raise ValueError("Duplicate record ID")
        record_ids.add(row["id"])
        for key in ("photo_id", "source_id", "split_group_id"):
            value = row.get(key)
            if (
                value is not None
                and identities.setdefault((key, value), row["split"]) != row["split"]
            ):
                raise ValueError("Metadata identity crosses splits")
        if (
            row["task"] not in SOURCE_CHANNELS
            or not row.get("usable_for_training", True)
            or row.get("exclude_from_training", False)
        ):
            continue
        if (
            row.get("preparation_preview")
            or row.get("preprocessing_profile") != "legacy25_raw"
            or row.get("pixels_per_degree") != 25
        ):
            raise ValueError("Require full canonical legacy25 inputs")
        photo = photo_lookup[row["photo_id"]]
        if photo["split"] != row["split"]:
            raise ValueError("Photo/manifest split differs")
        if row["dataset"] == "idrid_disc" and (
            row["field_of_view_degrees"] != 50 or photo["field_of_view_degrees"] != 50
        ):
            raise ValueError("IDRiD disc nominal acquisition geometry must remain50 degrees")
        groups[row["split"]].setdefault(row["photo_id"], []).append(row["id"])
    if not all(groups.values()):
        raise ValueError("Require nonempty TRAIN/VAL/TEST metadata")
    embedded = capture_geometry_policy(manifest.parent, rows)
    return {
        "manifest_sha256": manifest_sha,
        "prepared_content_sha256": audit["prepared_content_sha256"],
        "photos_sha256": sha_bytes((manifest.parent / "photos.json").read_bytes()),
        "geometry_policy_sha256": embedded["sha256"] if embedded else None,
        "eligible_photo_counts": {s: len(g) for s, g in groups.items()},
        "inventory_sha256": sha_bytes(
            canonical({s: {p: sorted(ids) for p, ids in g.items()} for s, g in groups.items()})
        ),
    }, groups


def model_snapshots(paths):
    if set(paths) != set(MODEL_SHA):
        raise ValueError("Require exactly the four selected original models")
    snapshots = {name: Path(path).read_bytes() for name, path in paths.items()}
    if {name: sha_bytes(data) for name, data in snapshots.items()} != MODEL_SHA:
        raise ValueError("Selected original checkpoint identity differs")
    return snapshots


def checked_test_policy(policy_path, expected_sha, manifest, paths):
    """All review gates precede device resolution, GT/image decoding and loading."""
    blob, policy = gate.json_snapshot(policy_path)
    if sha_bytes(blob) != expected_sha:
        raise ValueError("Frozen original policy differs from externally reviewed SHA")
    if (
        policy.get("kind") != "frozen_original_reference_policy"
        or policy.get("schema_version") != 1
    ):
        raise ValueError("Unsupported original selection artifact")
    if policy.get("policy") != POLICY or policy.get("source") != source_identity():
        raise ValueError("Original recipe/source/runtime differs from frozen selection")
    # Reject malformed freeze evidence before even reading canonical metadata.
    for key in ("validation_report_sha256", "selected_unified_selection_sha256"):
        value = policy.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Missing frozen VAL/unified decision digest")
    if gate.utc(policy.get("frozen_at_utc")) > datetime.now(UTC):
        raise ValueError("Original policy freeze is dated in the future")
    if not policy.get("rationale"):
        raise ValueError("Missing VAL selection rationale")
    identity, groups = metadata_identity(manifest)
    if policy.get("data") != identity:
        raise ValueError("Original selected canonical data differs")
    snapshots = model_snapshots(paths)
    return policy, identity, groups, snapshots


def clip_same_forward(masks):
    """Intersection happens in canonical coordinates, using that same disc forward."""
    result = np.asarray(masks, dtype=bool).copy()
    if result.ndim != 3 or result.shape[0] != 5:
        raise ValueError("Require five canonical masks")
    result[4] &= result[3]
    return result


def native_masks(image, loaded, device):
    result = np.zeros((5, *image.shape[-2:]), dtype=bool)
    for name, channels in (("universal", [0, 3, 4]), ("vessel", [0]), ("disc", [3])):
        model, payload = loaded[name]
        logits = tiled_logits(
            model,
            image - 0.5,
            device=device,
            crop_size=payload["tiling"]["crop_size"],
            border=payload["tiling"]["border"],
        ).numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite original segmentation logits")
        result[channels] = logits > 0
    return clip_same_forward(result)


def calibrated_av(rgb, vessel, fov, photo, av_model, device):
    """Existing refresh_original_av geometric/photometric and NN class kernels."""
    calibration = calibrate_provided_fov(fov, dataset=photo["canonical_dataset"])
    prepared = prepare_vesclass_matlab_calibrated(
        rgb,
        vessel,
        fov=fov,
        fov_diameter_pixels=calibration.diameter_pixels,
        dataset=photo["canonical_dataset"],
        field_of_view_degrees=photo["field_of_view_degrees"],
    )
    prediction = predict_av_input(
        av_model, prepared.inputs(), device=device, original_size=rgb.shape[:2]
    )
    z = prediction["prepared_logits"]
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite original AV logits")
    conditional = np.zeros(z.shape[-2:], np.uint8)
    conditional[z[1] > z[2]] = 1
    conditional[z[2] > z[1]] = 2
    return (
        prediction["classes"],
        restore_classes(conditional, rgb.shape[:2]),
        {
            **asdict(calibration),
            "diameter_pixels": calibration.diameter_pixels,
            "raw_fov_sha256": sha_bytes(fov.tobytes()),
        },
    )


@torch.inference_mode()
def evaluate(
    manifest,
    paths,
    data_root,
    *,
    split="val",
    device="cpu",
    policy_path=None,
    expected_policy_sha256=None,
):
    manifest = Path(manifest)
    if split == "test":
        if policy_path is None or expected_policy_sha256 is None:
            raise ValueError("TEST requires an explicitly reviewed frozen original policy SHA")
        policy, identity, groups, snapshots = checked_test_policy(
            policy_path, expected_policy_sha256, manifest, paths
        )
    elif split == "val":
        policy = None
        identity, groups = metadata_identity(manifest)
        snapshots = model_snapshots(paths)
    else:
        raise ValueError("Only full VAL or reviewed TEST is supported")
    source_before = source_identity()
    # Canonical identity audit only after review checks. No model/CUDA before it.
    verify_prepared_content(manifest, identity["prepared_content_sha256"])
    canonical_before = gate._canonical_metadata(manifest)
    device = resolve_device(device)
    loaded = {
        name: load_single_net(io.BytesIO(snapshots[name]), device)
        for name in ("universal", "vessel", "disc")
    }
    for name, channels in (
        ("universal", ["vessel", "disc", "cup"]),
        ("vessel", ["vessel"]),
        ("disc", ["disc"]),
    ):
        if loaded[name][1]["channels"] != channels:
            raise ValueError("Original selected model head metadata differs")
    av = load_av(io.BytesIO(snapshots["av"]), device)[0]
    for model in [item[0] for item in loaded.values()] + [av]:
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Nonfinite original parameters")
    dataset = UnifiedDataset(manifest, split, UnifiedConfig(purpose="full_data"))
    if dataset.photo_ids != sorted(groups[split]):
        raise ValueError("Evaluator inventory differs from frozen metadata")
    photos = {p["photo_id"]: p for p in read_json(manifest.parent / "photos.json")}
    results, sources = [], {}
    with ExitStack() as stack:
        for index, rows in enumerate(dataset.groups):
            image, target, valid, trace = dataset.sample(
                index, random_annotator=False, return_trace=True
            )
            target, valid = target.numpy() >= 0.5, valid.numpy().copy()
            masks = native_masks(image, loaded, device)
            pid = dataset.photo_ids[index]
            known = valid[1] & valid[2] & (target[1] ^ target[2])
            counts = {
                "correct": 0,
                "scored_pixels": 0,
                "eligible_pixels": int(known.sum()),
                "end_to_end_correct": 0,
            }
            calibration = None
            if known.any():
                photo = photos[pid]
                transform = geometry(photo)
                rgb = raw_photo(photo, Path(data_root), sources, stack)
                vessel = transform.restore(masks[0], order=0)
                if vessel.any() and not vessel.all():
                    with Image.open(manifest.parent / photo["fov"]) as handle:
                        fov = transform.restore(np.asarray(handle) > 0, order=0)
                    classes, conditional, calibration = calibrated_av(
                        rgb, vessel, fov, photo, av, device
                    )
                    classes = transform.prepare_mask(classes)
                    conditional = transform.prepare_mask(conditional)
                    masks[1], masks[2] = classes == 1, classes == 2
                    correct = (conditional == 1) & target[1] | (conditional == 2) & target[2]
                    counts.update(
                        correct=int((correct & known).sum()),
                        scored_pixels=int(known.sum()),
                        end_to_end_correct=int(
                            ((masks[1] & target[1] | masks[2] & target[2]) & known).sum()
                        ),
                    )
            values = confusion(masks, target, valid)
            lookup = {row["id"]: row for row in rows}
            attribution = {
                CHANNELS[ch]: lookup[record]["dataset"]
                for record, task in trace
                for ch in SOURCE_CHANNELS[task]
            }
            results.append(
                {
                    "photo_id": pid,
                    "dataset": rows[0]["dataset"],
                    "record_ids": [r["id"] for r in rows],
                    "channel_datasets": attribution,
                    "channels": {c: _metrics(v) for c, v in zip(CHANNELS, values, strict=True)},
                    "av_conditional_counts": counts,
                    "av_calibration": calibration,
                }
            )
            if len(results) % 25 == 0:
                print(json.dumps({"split": split, "photos": len(results)}), flush=True)
    if {r["photo_id"]: sorted(r["record_ids"]) for r in results} != {
        p: sorted(ids) for p, ids in groups[split].items()
    }:
        raise ValueError("Incomplete original per-photo/record coverage")
    pooled, by_dataset, macro, score = aggregate(
        results, sorted({r["dataset"] for r in dataset.rows})
    )
    sums = {
        k: sum(r["av_conditional_counts"][k] for r in results)
        for k in ("correct", "scored_pixels", "eligible_pixels", "end_to_end_correct")
    }
    # Repeat source and full-content leases before exclusive report publication.
    verify_prepared_content(manifest, identity["prepared_content_sha256"])
    if (
        source_identity() != source_before
        or gate._canonical_metadata(manifest) != canonical_before
        or metadata_identity(manifest)[0] != identity
    ):
        raise ValueError("Original evaluation source/canonical inputs changed during evaluation")
    return {
        "schema_version": 1,
        "family": "frozen_original_reference",
        "split": split,
        **identity,
        "source": source_before,
        "policy_spec": POLICY,
        "photos": len(results),
        "preview_only": False,
        "checkpoint_sha256": MODEL_SHA["universal"],
        "vessel_checkpoint_sha256": MODEL_SHA["vessel"],
        "disc_checkpoint_sha256": MODEL_SHA["disc"],
        "av_checkpoint_sha256": MODEL_SHA["av"],
        "channels": pooled,
        "by_dataset": by_dataset,
        "macro_dataset_channel_dice": macro,
        "macro_dataset_mean_dice": score,
        "av_conditional_accuracy": sums["correct"] / sums["scored_pixels"]
        if sums["scored_pixels"]
        else None,
        "av_end_to_end_accuracy": sums["end_to_end_correct"] / sums["eligible_pixels"]
        if sums["eligible_pixels"]
        else None,
        "av_labeled_pixels": sums["eligible_pixels"],
        "av_conditional_scored_pixels": sums["scored_pixels"],
        "per_photo": results,
        "limitations": LIMITATIONS,
        "independent_quality_improvement_claim": False,
        "reviewed_policy_sha256": expected_policy_sha256 if policy else None,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
    }


def validate_saved_counts(report, manifest):
    """Metadata/count consistency only: never reopen or decode GT pixels."""
    records = {row["id"]: row for row in read_json(manifest)}
    keys = ("tp", "fp", "fn", "valid_pixels")
    for photo in report["per_photo"]:
        selected = sorted((records[rid] for rid in photo["record_ids"]), key=lambda row: row["id"])
        tasks = [task for task in ("av", "vessels") if any(r["task"] == task for r in selected)][:1]
        if any(r["task"] == "disc_cup" for r in selected):
            tasks.append("disc_cup")
        attribution = {}
        for task in tasks:
            first = next(row for row in selected if row["task"] == task)
            attribution.update({CHANNELS[ch]: first["dataset"] for ch in SOURCE_CHANNELS[task]})
        if photo["dataset"] != selected[0]["dataset"] or photo["channel_datasets"] != attribution:
            raise ValueError("Saved annotation-source policy differs")
        for name in CHANNELS:
            counts = photo["channels"][name]
            gate._counts(counts)
            if name not in attribution and any(counts[k] for k in keys):
                raise ValueError("Unsupported saved channel contains scored pixels")
        av = photo["av_conditional_counts"]
        if set(av) != {"correct", "scored_pixels", "eligible_pixels", "end_to_end_correct"} or any(
            type(v) is not int or v < 0 for v in av.values()
        ):
            raise ValueError("Invalid saved conditional AV counts")
        a, v = [photo["channels"][name] for name in ("artery", "vein")]
        if (
            av["eligible_pixels"] != a["tp"] + a["fn"] + v["tp"] + v["fn"]
            or av["end_to_end_correct"] != a["tp"] + v["tp"]
            or not 0
            <= av["end_to_end_correct"]
            <= av["correct"]
            <= av["scored_pixels"]
            <= av["eligible_pixels"]
        ):
            raise ValueError("Saved conditional AV support differs")
    pooled, grouped, macro, score = aggregate(report["per_photo"], report["by_dataset"])
    if (
        report["channels"] != pooled
        or report["by_dataset"] != grouped
        or report["macro_dataset_channel_dice"] != macro
        or report["macro_dataset_mean_dice"] != score
    ):
        raise ValueError("Saved original aggregate counts differ")


def freeze(validation_report, manifest, output, selected_unified_selection_sha256, rationale):
    blob, report = gate.json_snapshot(validation_report)
    identity, groups = metadata_identity(manifest)
    if (
        report.get("family") != "frozen_original_reference"
        or report.get("split") != "val"
        or report.get("preview_only") is not False
        or report.get("source") != source_identity()
        or report.get("policy_spec") != POLICY
        or report.get("photos") != len(groups["val"])
        or any(report.get(k) != v for k, v in identity.items())
        or {r["photo_id"]: sorted(r["record_ids"]) for r in report["per_photo"]}
        != {p: sorted(ids) for p, ids in groups["val"].items()}
        or len(report["per_photo"]) != len(groups["val"])
    ):
        raise ValueError("Require complete current-source fixed-policy VAL evidence")
    validate_saved_counts(report, manifest)
    if gate.utc(report.get("evaluated_at_utc")) > datetime.now(UTC):
        raise ValueError("VAL evidence is dated after policy freeze")
    if (
        not isinstance(selected_unified_selection_sha256, str)
        or not rationale
        or len(selected_unified_selection_sha256) != 64
        or any(c not in "0123456789abcdef" for c in selected_unified_selection_sha256)
    ):
        raise ValueError("Require rationale and externally frozen unified selection SHA")
    policy = {
        "schema_version": 1,
        "kind": "frozen_original_reference_policy",
        "policy": POLICY,
        "data": identity,
        "source": source_identity(),
        "validation_report_sha256": sha_bytes(blob),
        "selected_unified_selection_sha256": selected_unified_selection_sha256,
        "rationale": rationale,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "limitations": LIMITATIONS,
    }
    write_new(output, policy)
    return sha_bytes(Path(output).read_bytes())


def write_new(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(report) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        cmd = commands.add_parser(name)
        for option in ("manifest", "data-root", "universal", "vessel", "disc", "av", "output"):
            cmd.add_argument("--" + option, type=Path, required=True)
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--cpu-threads", type=int, default=1)
        if name == "test":
            cmd.add_argument("--policy", type=Path, required=True)
            cmd.add_argument("--expected-policy-sha256", required=True)
            cmd.add_argument("--execute-reviewed-heldout", action="store_true", required=True)
    cmd = commands.add_parser("freeze")
    for option in ("validation-report", "manifest", "output"):
        cmd.add_argument("--" + option, type=Path, required=True)
    cmd.add_argument("--selected-unified-selection-sha256", required=True)
    cmd.add_argument("--rationale", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if Path(args["output"]).exists():
        parser.error("Preserve previous output")
    if command == "freeze":
        print(freeze(**args))
        return
    threads = args.pop("cpu_threads")
    if threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(threads)
    args.pop("execute_reviewed_heldout", None)
    paths = {name: args.pop(name) for name in MODEL_SHA}
    output = args.pop("output")
    args["policy_path"] = args.pop("policy", None)
    result = evaluate(paths=paths, split=command, **args)
    write_new(output, result)
    print(output)


if __name__ == "__main__":
    main()
