import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

p = Path(__file__).with_name("comparison_with_pooled_mean.py")
s = importlib.util.spec_from_file_location("paired_extension", p)
h = importlib.util.module_from_spec(s)
s.loader.exec_module(h)


def reports(tmp_path, improve):
    rows = []
    photos = []
    for i in range(2):
        ids = []
        for task in ["av", "disc_cup"]:
            rid = f"p{i}-{task}"
            ids.append(rid)
            rows.append(
                {
                    "id": rid,
                    "photo_id": f"p{i}",
                    "source_id": f"s{i}",
                    "split_group_id": f"g{i}",
                    "dataset": "toy",
                    "split": "val",
                    "task": task,
                }
            )
        photos.append(
            {
                "photo_id": f"p{i}",
                "dataset": "toy",
                "record_ids": ids,
                "channel_datasets": {c: "toy" for c in h.CHANNELS},
                "channels": {c: {"tp": 1, "fp": 1, "fn": 0, "valid_pixels": 2} for c in h.CHANNELS},
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(json.dumps({"split": "val", "manifest_sha256": sha, "per_photo": photos}))
    if improve:
        for r in photos:
            for v in r["channels"].values():
                v["fp"] = 0
    candidate.write_text(json.dumps({"split": "val", "manifest_sha256": sha, "per_photo": photos}))
    return reference, candidate, manifest


@pytest.mark.parametrize("improve", [False, True])
def test_pooled_and_macro_share_paired_draws_and_closed_form_difference(tmp_path, improve):
    r = h.compare_reports(*reports(tmp_path, improve), repeats=100, seed=42)
    expected = 1 / 3 if improve else 0
    for metric in [
        r["macro_dataset_mean_dice"],
        r["pooled_mean_dice"],
        *[v["pooled_dice"] for v in r["channels"].values()],
    ]:
        assert metric["candidate_minus_reference"] == pytest.approx(expected)
        assert metric["paired_percentile_95ci"] == pytest.approx([expected, expected])
