"""Frozen6b bit/RNG compatibility and isolated vascular-key weighting; no GT files."""

import hashlib
import importlib.util
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from deepretina.unified import CoverageSampler, UnifiedConfig, UnifiedDataset

ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "tests/reference/vascular_sampling/frozen6b.py"
spec = importlib.util.spec_from_file_location("frozen6b_sampling", REFERENCE)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def dataset(policy="dataset_task_balanced", priority=1.0):
    groups = [
        [{"id": "drive-v", "dataset": "drive"}, {"id": "rite-av", "dataset": "rite"}],
        [{"id": "disc1", "dataset": "dc"}, {"id": "disc2", "dataset": "dc"}],
        [{"id": "mixed-v", "dataset": "mixed"}, {"id": "mixed-dc", "dataset": "mixed"}],
        [{"id": "av2", "dataset": "vascular"}],
        [{"id": "disc-only", "dataset": "dc"}],
        [{"id": "missing-cup", "dataset": "dc"}],
        [{"id": "no-selected-task", "dataset": "empty"}],
        [{"id": "declared-group", "dataset": "source", "sampling_group": "explicit-owner"}],
    ]
    flags = {
        "drive-v": {"vessel"},
        "rite-av": {"vessel", "av"},
        "disc1": {"disc", "cup"},
        "disc2": {"disc"},
        "mixed-v": {"vessel"},
        "mixed-dc": {"disc"},
        "av2": {"vessel", "av"},
        "disc-only": {"disc"},
        "missing-cup": {"disc"},
        "no-selected-task": set(),
        "declared-group": {"vessel"},
    }
    value = object.__new__(UnifiedDataset)
    value.config = UnifiedConfig(sampling=policy, vascular_sampling_weight=priority)
    value.config.validate()
    value.groups = groups
    value._row_tasks = flags
    value.support_summary = lambda: None
    return value


def test_frozen_reference_bytes_match_provenance():
    provenance = json.loads((REFERENCE.parent / "provenance.json").read_text())
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["source_commit"] == "6b3dfad605a256e904a877efcdcc1f3cc6135dd0"


@pytest.mark.parametrize(
    "policy", ["uniform", "dataset_balanced", "task_balanced", "dataset_task_balanced"]
)
def test_default_one_weights_are_bitwise_frozen6b_for_all_existing_policies(policy):
    value = dataset(policy)
    expected = np.asarray(old.sampling_weights(value), dtype=np.float64)
    actual = np.asarray(value.sampling_weights(), dtype=np.float64)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert [x.hex() for x in actual] == [x.hex() for x in expected]


@pytest.mark.parametrize("seed", [0, 42, 717])
@pytest.mark.parametrize("count", [3, 8, 8192])
@pytest.mark.parametrize("coverage_first", [True, False])
def test_default_weights_and_sampler_preserve_every_draw_and_rng_state(seed, count, coverage_first):
    value = dataset()
    new_generator = torch.Generator().manual_seed(seed)
    old_generator = torch.Generator().manual_seed(seed)
    current = CoverageSampler(value.sampling_weights(), count, new_generator, coverage_first)
    frozen = old.CoverageSampler(old.sampling_weights(value), count, old_generator, coverage_first)
    for _ in range(3):
        assert list(current) == list(frozen)
        assert torch.equal(new_generator.get_state(), old_generator.get_state())


def test_priority_two_multiplies_only_existing_vascular_key_contributions():
    value = dataset(priority=2)
    memberships = []
    for rows in value.groups:
        owner = rows[0].get("sampling_group", rows[0]["dataset"])
        tasks = set().union(*(value._row_tasks[r["id"]] for r in rows))
        memberships.append({(owner, t) for t in tasks})
    populations = Counter(key for keys in memberships for key in keys)
    expected = [
        sum(
            (2 if task in {"vessel", "av"} else 1) / populations[(source, task)]
            for source, task in keys
        )
        for keys in memberships
    ]
    assert value.sampling_weights() == expected
    baseline = dataset().sampling_weights()
    weighted = value.sampling_weights()
    assert weighted[1] == baseline[1] and weighted[4] == baseline[4] and weighted[5] == baseline[5]
    assert weighted[0] == 2 * baseline[0] and weighted[3] == 2 * baseline[3]
    assert baseline[2] < weighted[2] < 2 * baseline[2]  # mixed photo: DC component unmodified
    assert weighted[6] == 0  # no new memberships are invented
    generator = torch.Generator().manual_seed(42)
    draws = list(CoverageSampler(weighted, 8192, generator, coverage_first=True))
    assert sorted(draws[: len(weighted)]) == list(range(len(weighted)))


@pytest.mark.parametrize(
    "bad", [True, False, float("nan"), float("inf"), -float("inf"), 0, -1, "2", None]
)
def test_priority_must_be_numeric_positive_finite_and_not_bool(bad):
    with pytest.raises(ValueError, match="vascular_sampling_weight"):
        UnifiedConfig(vascular_sampling_weight=bad).validate()


@pytest.mark.parametrize("policy", ["uniform", "dataset_balanced", "task_balanced"])
def test_nondefault_priority_rejected_for_every_other_sampling_policy(policy):
    with pytest.raises(ValueError, match="requires dataset_task_balanced"):
        UnifiedConfig(sampling=policy, vascular_sampling_weight=2).validate()


def test_sampling_owner_alias_conflict_guard_still_applies():
    value = dataset(priority=2)
    value.groups[0][0]["sampling_group"] = "one"
    value.groups[0][1]["sampling_group"] = "two"
    with pytest.raises(ValueError, match="conflicting sampling_group"):
        value.sampling_weights()


def test_control_and_vascular_presets_differ_only_by_declared_scalar():
    configs = [
        tomllib.loads((ROOT / "configs" / name).read_text())
        for name in ["unified-v2-control384.toml", "unified-v2-vascular2.toml"]
    ]
    assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {
        "vascular_sampling_weight"
    }
    assert set(configs[0]) == set(configs[1])
    assert (
        configs[0]["vascular_sampling_weight"] == 1 and configs[1]["vascular_sampling_weight"] == 2
    )
    for config in configs:
        UnifiedConfig(**config).validate()
        assert (
            config["epochs"],
            config["epoch_samples"],
            config["seed"],
            config["patch_size"],
            config["base_channels"],
            config["augmentation"],
        ) == (60, 8192, 42, 384, 12, "mild")
        assert config["preprocessing_profile"] == "legacy25_raw"


def test_observer_crop_augmentation_loss_and_architecture_are_unchanged():
    import ast

    provenance = json.loads((REFERENCE.parent / "provenance.json").read_text())
    source = (ROOT / "src/deepretina/unified.py").read_text()
    tree = ast.parse(source)
    current = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedDataset"
    )
    for name, expected in provenance["unchanged_source_segment_sha256"].items():
        if name == "train_unified":
            # This research branch intentionally wraps training precision; the
            # pinned original functional parity test covers its default path.
            continue
        nodes = current.body if name in {"sample", "__getitem__"} else tree.body
        node = next(
            n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == expected
    assert (
        hashlib.sha256((ROOT / "src/deepretina/augment.py").read_bytes()).hexdigest()
        == provenance["unchanged_augment_sha256"]
    )


def test_direct_weight_computation_rejects_mutated_nondefault_policy():
    value = dataset(priority=2)
    value.config.sampling = "uniform"
    with pytest.raises(ValueError, match="requires dataset_task_balanced"):
        value.sampling_weights()


def test_new_field_is_appended_to_preserve_existing_positional_configuration_api():
    from dataclasses import fields

    provenance = json.loads((REFERENCE.parent / "provenance.json").read_text())
    names = [field.name for field in fields(UnifiedConfig)]
    assert names == provenance["frozen_config_field_order"] + ["vascular_sampling_weight", "mixed_precision"]
    defaults = UnifiedConfig()
    positional = UnifiedConfig(
        *(getattr(defaults, name) for name in provenance["frozen_config_field_order"])
    )
    assert positional == defaults and positional.vascular_sampling_weight == 1
