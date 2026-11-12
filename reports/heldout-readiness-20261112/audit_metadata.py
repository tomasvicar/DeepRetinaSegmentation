"""Read JSON/source metadata only; never opens images, NPZ, weights or a database."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
ORIGINAL = ROOT / "reports/frozen-original-val-nearest/validation.json"
STRONG = (
    ROOT
    / "reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json"
)
FRESH = ROOT / "reports/frozen-original-val-reproduction/validation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(manifest):
    assert sha(manifest) == MANIFEST
    rows = json.loads(manifest.read_bytes())
    partitions, inventory = (
        {},
        defaultdict(lambda: {"photos": set(), "records": [], "groups": set()}),
    )
    for row in rows:
        for key in ["photo_id", "source_id", "split_group_id"]:
            assert isinstance(row[key], str) and row[key]
            identity = (key, row[key])
            assert partitions.setdefault(identity, row["split"]) == row["split"]
        if (
            row["task"] not in {"vessels", "av", "disc_cup"}
            or not row.get("usable_for_training", True)
            or row.get("exclude_from_training", False)
        ):
            continue
        inventory[row["split"]]["photos"].add(row["photo_id"])
        inventory[row["split"]]["records"].append(row["id"])
        inventory[row["split"]]["groups"].add(row["split_group_id"])
    counts = {s: {k: len(v) for k, v in d.items()} for s, d in inventory.items()}
    assert counts["train"]["photos"] == 2529 and counts["train"]["records"] == 5700
    assert counts["val"]["photos"] == 770 and counts["test"]["photos"] == 856
    strong, original, fresh = [json.loads(p.read_bytes()) for p in [STRONG, ORIGINAL, FRESH]]
    assert sha(STRONG) == "528233fe50876b15f127f8567fc43d7b8a34a8713cfceff54f02956a0a0bfa2c"
    assert sha(ORIGINAL) == "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"
    assert len(strong["per_photo"]) == len(original["per_photo"]) == 770
    a, b = [{p["photo_id"]: p for p in d["per_photo"]} for d in [strong, original]]
    assert set(a) == set(b) == inventory["val"]["photos"]
    diff = {
        h: {
            "photo_confusion_tuples_changed": 0,
            "pooled_count_differences": {
                k: original["channels"][h][k] - strong["channels"][h][k]
                for k in ["tp", "fp", "fn", "valid_pixels"]
            },
            "strong528_dice": strong["channels"][h]["dice"],
            "original182_dice": original["channels"][h]["dice"],
        }
        for h in ["vessel", "artery", "vein", "disc", "cup"]
    }
    av_changed = 0
    fresh_lookup = {p["photo_id"]: p for p in fresh["per_photo"]}
    assert set(fresh_lookup) == set(a)
    for pid in a:
        left, right = a[pid], b[pid]
        assert set(left["record_ids"]) == set(right["record_ids"])
        assert (
            left["dataset"] == right["dataset"]
            and left["channel_datasets"] == right["channel_datasets"]
        )
        for head, summary in diff.items():
            l, r = left["channels"][head], right["channels"][head]
            assert l["valid_pixels"] == r["valid_pixels"] and l["tp"] + l["fn"] == r["tp"] + r["fn"]
            summary["photo_confusion_tuples_changed"] += any(
                l[k] != r[k] for k in ["tp", "fp", "fn", "valid_pixels"]
            )
            assert all(
                l[k] == fresh_lookup[pid]["channels"][head][k]
                for k in ["tp", "fp", "fn", "valid_pixels"]
            )
        av_changed += any(
            left.get("av_conditional_counts", {}).get(k, 0)
            != right.get("av_conditional_counts", {}).get(k, 0)
            for k in ["correct", "scored_pixels", "eligible_pixels", "end_to_end_correct"]
        )
    for key in [
        "checkpoint_sha256",
        "vessel_checkpoint_sha256",
        "disc_checkpoint_sha256",
        "av_checkpoint_sha256",
    ]:
        assert strong[key] == original[key]
    result = {
        "scope": "JSON/source metadata only; no TEST/pixel/GT/model decode, no NN/GPU",
        "manifest_sha256": MANIFEST,
        "eligible_inventory": counts,
        "all_photo_source_known_group_identities_confined_to_one_split": True,
        "strong528_sha256": sha(STRONG),
        "original182_sha256": sha(ORIGINAL),
        "fresh_original_reproduction53b_sha256": sha(FRESH),
        "same4_component_weight_SHA": True,
        "same770_photo_record_source_and_GT_support": True,
        "strong528_vs_original182": diff,
        "AV_photos_with_any_numeric_count_change": av_changed,
        "strong528_numeric_confusion_counts_exact_to_independent_fresh53b": True,
        "original182_declared_policy": original["policy_spec"],
        "strong528_declared_cup": strong["fresh_paired_cup_component"],
        "strong528_global_pins": {
            k: strong[k] for k in ["manifest_sha256", "prepared_content_sha256"]
        },
        "original182_global_pins": {
            k: original[k] for k in ["manifest_sha256", "prepared_content_sha256"]
        },
        "source_code_SHA": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                ROOT / "scripts/unified_validation.py",
                ROOT / "scripts/unified_test.py",
                ROOT / "scripts/evaluate_original_frozen.py",
                ROOT / "scripts/compare_reviewed_test.py",
                ROOT / "src/deepretina/heldout.py",
                ROOT / "src/deepretina/selection.py",
                ROOT / "src/deepretina/freeze_eligibility.py",
                ROOT / "src/deepretina/curated_completion.py",
            ]
        },
        "no_report_retagged_or_quality_measurement_run": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    main(p.parse_args().manifest)
