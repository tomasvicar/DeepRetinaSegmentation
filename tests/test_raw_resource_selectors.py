"""Synthetic metadata names only; no real source archives or payloads."""

from pathlib import Path

import pytest

from deepretina.raw35_producer import FrozenPlan, resolve_requests
from deepretina.raw_resource_selectors import refuge_mask_selector, resolve_resource_requests


def resolve(names, split="train", image="REFUGE/Training400/Glaucoma/g0001.jpg"):
    key, selector = refuge_mask_selector(split, image)
    return key, resolve_resource_requests(
        names, [key], selectors={key: selector}, existing_resolver=resolve_requests
    )[key]


def test_nested_mask_matches_literal_decoder_and_sorted_ties():
    folder = "Annotation-Training400/Disc_Cup_Masks"
    names = [
        f"REFUGE/{folder}/Normal/g0001.bmp",
        f"REFUGE/{folder}/Glaucoma/g0001.bmp",
        f"long/prefix/REFUGE/{folder}/Glaucoma/g0001.bmp",
        "REFUGE/other/g0001.bmp",
        f"REFUGE/{folder}/Glaucoma/g0002.bmp",
    ]
    matches = [
        n
        for n in sorted(names)
        if f"/{folder.lower()}/" in n.lower() and Path(n).name.lower() == "g0001.bmp"
    ]
    key, selected = resolve(reversed(names))
    assert selected == min(matches, key=len)
    assert selected.endswith("Normal/g0001.bmp")  # shortest, not inferred image category
    with pytest.raises(ValueError, match="Missing/ambiguous"):
        resolve_requests(names, [key])  # proves the original failure mode


def test_equal_length_tie_uses_source_sorted_order():
    base = "REFUGE/Annotation-Training400/Disc_Cup_Masks/"
    assert resolve([base + "BBB/g0001.bmp", base + "AAA/g0001.bmp"])[1] == base + "AAA/g0001.bmp"


@pytest.mark.parametrize(
    "split,folder",
    [
        ("train", "Annotation-Training400"),
        ("val", "REFUGE-Validation400-GT"),
        ("test", "REFUGE-Test-GT"),
    ],
)
def test_exact_split_folder_case_insensitive_basename(split, folder):
    name = f"REFUGE/{folder}/Disc_Cup_Masks/Nested/G0001.BMP"
    assert resolve([name], split)[1] == name


def test_ordinary_requests_delegate_unchanged_and_typed_not_suffix_fallback():
    key, selector = refuge_mask_selector("train", "g0001.jpg")
    calls = []

    def delegate(names, requests, identities):
        calls.append((requests, identities))
        return resolve_requests(names, requests, identities)

    identity = {"rgb.jpg": (1, 2)}
    name = "REFUGE/Annotation-Training400/Disc_Cup_Masks/X/g0001.bmp"
    result = resolve_resource_requests(
        [name, "rgb.jpg"], [key, "rgb.jpg"], identity, {key: selector}, existing_resolver=delegate
    )
    assert result == {key: name, "rgb.jpg": "rgb.jpg"}
    assert calls == [(["rgb.jpg"], identity)]
    with pytest.raises(ValueError, match="Missing REFUGE"):
        resolve(["OTHER/Disc_Cup_Masks/X/g0001.bmp"])


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "unknown"},
        {"folder": "../Disc_Cup_Masks"},
        {"basename": "../g0001.bmp"},
        {"basename": "x\\g0001.bmp"},
        {"extra": True},
        {"basename": "other.bmp"},
    ],
)
def test_invalid_selector_rejects_before_resolver_payload(change):
    key, selector = refuge_mask_selector("train", "g0001.jpg")
    selector.update(change)
    with pytest.raises(ValueError, match="Invalid REFUGE"):
        resolve_resource_requests(
            [], [key], selectors={key: selector}, existing_resolver=resolve_requests
        )


def test_frozen_plan_selector_addition_preserves_rows_photos_membership():
    row = {
        "id": "r",
        "dataset": "refuge",
        "split": "train",
        "task": "disc_cup",
        "photo_id": "p",
        "pixels_per_degree": 25,
        "metadata": {
            "source_image": "REFUGE/Training400/Glaucoma/g0001.jpg",
            "source_split": "train",
        },
        "split_group_id": "g",
    }
    photo = {
        "photo_id": "p",
        "split": "train",
        "pixels_per_degree": 25,
        "canonical_dataset": "refuge",
        "sources": [{"dataset": "refuge", "resource": row["metadata"]["source_image"]}],
    }
    plan = FrozenPlan([row], [photo])
    assert plan.rows == {"r": row} and plan.photos == {"p": photo}
    summary = plan.summary()
    key, selector = refuge_mask_selector("train", row["metadata"]["source_image"])
    assert plan.resource_selectors == {"REFUGE.zip": {key: selector}}
    assert key in plan.requests["REFUGE.zip"]
    assert summary["membership_sha256"] == plan.signature
    assert summary["resource_selectors"] == plan.resource_selectors
    assert summary["TEST_inventory_records_unmaterialized"] == 0
