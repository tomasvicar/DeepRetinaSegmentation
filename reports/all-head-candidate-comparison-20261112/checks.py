"""Read-only projection checks; no ML imports, bootstrap, pixels or model reads."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    d = json.loads((HERE / "comparison.json").read_bytes())
    previous = d["previous_snapshot"]
    assert sha(ROOT / previous["path"]) == previous["sha256"]
    old = json.loads((ROOT / previous["path"]).read_bytes())
    assert d["candidates"][:9] == old["candidates"]
    assert d["inputs"][:21] == old["inputs"]
    assert len(d["candidates"]) == 17
    historical = {x["path"]: x for x in d["historical_input_resolutions"]}
    for item in d["inputs"]:
        path = ROOT / item["path"]
        assert path.resolve().is_relative_to(ROOT)
        if sha(path) != item["sha256"]:
            h = historical[item["path"]]
            raw = subprocess.check_output(
                ["git", "show", h["historical_commit"] + ":" + item["path"]], cwd=ROOT
            )
            assert hashlib.sha256(raw).hexdigest() == item["sha256"] == h["sha256"]
    candidates = {r["id"]: r for r in d["candidates"]}
    reference = candidates["original182"]["pooled_head_counts_and_metrics"]
    for row in d["candidates"]:
        assert row["VAL_photos"] == 770
        for head, counts in row["pooled_head_counts_and_metrics"].items():
            assert counts["valid_pixels"] == reference[head]["valid_pixels"]
            assert counts["tp"] + counts["fn"] == reference[head]["tp"] + reference[head]["fn"]
    for row in d["candidates"][9:]:
        original = json.loads((ROOT / row["source"]["path"]).read_bytes())
        metrics = original["metrics"]
        assert row["checkpoint"] == original["checkpoint"]
        assert row["pooled_head_counts_and_metrics"] == metrics["channels"]
        assert row["conditional_AV_accuracy"] == metrics["av_conditional_accuracy"]
        assert row["macro_dataset_mean_dice"] == metrics["macro_dataset_mean_dice"]
        c = row["own_TRAIN_coverage"]["coverage"]
        assert c["eligible_annotation_records"] == c["contributing_annotation_records"] == 5700
        assert c["eligible_photos"] == c["seen_photos"] == 2529
        assert c["epoch"] == row["checkpoint"]["epoch"]
    assert "TPE7-BEST46" not in candidates
    assert candidates["TPE7-LAST60"]["checkpoint"]["epoch"] == 60
    excluded = d["ineligible_not_selected"]["TPE7-BEST46"]
    assert excluded["eligible"] is False and excluded["selected_for_comparison"] is False
    assert excluded["own_TRAIN_coverage"]["contributing_annotation_records"] == 5699
    for entry in d["dense_vessel_by_candidate_id"].values():
        if entry["pooled"] is not None:
            assert entry["support"]["photos"] == 87
            assert entry["support"]["known_split_groups"] == 55
            assert len(entry["support"]["sources"]) == 8
            ci = entry["existing_paired_vs_original"]
            if ci:
                assert ci["candidate"] == entry["pooled"]["dice"]
                assert (
                    ci["reference"]
                    == d["dense_vessel_by_candidate_id"]["original182"]["pooled"]["dice"]
                )
    assert not re.search(
        r"/var/home/|/tmp/|/auto/|GPU-[0-9a-fA-F-]{8}|d0675605|b60687",
        (HERE / "comparison.json").read_text(),
    )
    print(
        "PASS: prior9/21pins preserved; 17 exact metric/role rows; dense87/55/8; no new statistics"
    )


if __name__ == "__main__":
    main()
