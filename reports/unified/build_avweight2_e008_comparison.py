"""Recompute metadata-only paired E8 loss-weight summaries from saved counts."""

import hashlib
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
new = json.loads((folder / "old-avweight2-e008-counts.json").read_text())
base = json.loads(
    (folder / "paired-e008-strong-reference/candidate-counts/lr0003-e008-counts.json").read_text()
)
qa = json.loads((folder / "old-avweight2-e008-verified.json").read_text())
left = {r["photo_id"]: r for r in base["per_photo"]}
right = {r["photo_id"]: r for r in new["per_photo"]}
assert left.keys() == right.keys() and len(right) == 770
channels = list(new["channels"])
for pid, a in left.items():
    b = right[pid]
    for k in ["record_ids", "dataset", "channel_datasets"]:
        assert a[k] == b[k]
    for c in channels:
        assert a["channels"][c]["valid_pixels"] == b["channels"][c]["valid_pixels"]
        assert (
            a["channels"][c]["tp"] + a["channels"][c]["fn"]
            == b["channels"][c]["tp"] + b["channels"][c]["fn"]
        )
new.update(
    manifest_sha256=qa["manifest_sha256"],
    prepared_content_sha256=qa["canonical_fingerprint"],
    checkpoint_sha256=qa["checkpoint_sha256"]["best.pt"],
    selection_epoch=8,
    counts_provenance="Copied validation_metrics from independently verified terminal best/last E8; no new inference",
)
newpath = folder / "old-avweight2-e008-counts.json"
newpath.write_text(json.dumps(new, indent=2) + "\n")
sources = {}
for source, vals in new["by_dataset"].items():
    sources[source] = {}
    for c, counts in vals.items():
        photos = sum(
            1
            for row in new["per_photo"]
            if row["channel_datasets"].get(c) == source and row["channels"][c]["valid_pixels"] > 0
        )
        prior = base["by_dataset"][source][c]
        assert (
            counts["valid_pixels"] == prior["valid_pixels"]
            and counts["tp"] + counts["fn"] == prior["tp"] + prior["fn"]
        )
        sources[source][c] = {
            "supported_photos": photos,
            "av_weight_1_counts": prior,
            "av_weight_2_counts": counts,
            "dice_delta": None if counts["dice"] is None else counts["dice"] - prior["dice"],
        }
summary = {
    "split": "val",
    "epoch": 8,
    "all_photos": 770,
    "all_records": 993,
    "same_photo_annotations_and_GT_support_verified": True,
    "no_forward": True,
    "no_test_evaluation": True,
    "av_weight_2_plan_sha256": qa["plan_sha256"],
    "canonical_fingerprint": qa["canonical_fingerprint"],
    "counts_sha256": hashlib.sha256(newpath.read_bytes()).hexdigest(),
    "channels": {
        c: {
            "av_weight_1_pooled": base["channels"][c]["dice"],
            "av_weight_2_pooled": new["channels"][c]["dice"],
            "pooled_delta": new["channels"][c]["dice"] - base["channels"][c]["dice"],
            "av_weight_1_macro": base["macro_dataset_channel_dice"][c],
            "av_weight_2_macro": new["macro_dataset_channel_dice"][c],
            "macro_delta": new["macro_dataset_channel_dice"][c]
            - base["macro_dataset_channel_dice"][c],
        }
        for c in channels
    },
    "av_conditional_accuracy": {
        "av_weight_1": base["av_conditional_accuracy"],
        "av_weight_2": new["av_conditional_accuracy"],
        "delta": new["av_conditional_accuracy"] - base["av_conditional_accuracy"],
    },
    "av_end_to_end_accuracy": {
        "av_weight_1": base["av_accuracy"],
        "av_weight_2": new["av_accuracy"],
        "delta": new["av_accuracy"] - base["av_accuracy"],
    },
    "source_counts": sources,
    "strong_calibrated_original_AV_point_reference": {
        "artery_dice": 0.6895286745,
        "vein_dice": 0.6972977562,
        "conditional_accuracy": 0.7485732693,
        "historical_exposure_caveat": True,
    },
    "interpretation": "One-seed E8 validation screen; improved AV point metrics vs matched loss-weight1, lower vessels, still below calibrated original AV; not final superiority",
}
(folder / "old-avweight2-e008-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print("GT supports identical, sources", len(sources))
for source, cs in sources.items():
    print(source, {c: v["supported_photos"] for c, v in cs.items() if v["supported_photos"]})
