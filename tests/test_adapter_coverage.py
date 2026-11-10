"""Prevent legacy dataset migrations from silently disappearing from the CLI."""

import json
from pathlib import Path

from deepretina.prepare import DATASETS

COVERAGE = Path(__file__).parents[1] / "src/deepretina/adapters/coverage.json"
LEGACY_MATLAB_LOADERS = {
    "aria",
    "avrdb",
    "avrdb_VA",
    "chasedb1",
    "diaretdb0",
    "diaretdb1",
    "drhagis",
    "drhagis_VA",
    "drishtigs",
    "drive",
    "drive_VA",
    "eophtha",
    "erlangen",
    "fire",
    "g1020",
    "heimed",
    "hrf",
    "hrf_VA",
    "idrid",
    "inspireavr",
    "iostar",
    "iostar_VA",
    "messidor",
    "onhsd",
    "rcrgbma",
    "refuge",
    "review",
    "ridb",
    "riga",
    "rimone",
    "roc",
    "stare",
    "ubmi_VA",
    "uoadr",
    "vicavr",
}
LEGACY_PYTHON_LOADERS = {
    "Aria",
    "Avrdb",
    "Chasedb1",
    "Drhagis",
    "Drishtigs",
    "Drive",
    "G1020",
    "Hrf",
    "Iostar",
    "Onhsd",
    "Refuge",
    "Riga",
    "Stare",
    "Uoa_dr",
}


def test_inventory_covers_every_historical_adapter():
    coverage = json.loads(COVERAGE.read_text())
    assert coverage["schema_version"] == 1
    entries = coverage["entries"]
    paths = {path for entry in entries for path in entry["legacy_paths"]}
    assert {
        Path(path).stem.removeprefix("load_") for path in paths if path.endswith(".m")
    } == LEGACY_MATLAB_LOADERS
    assert {
        Path(path).stem.removeprefix("Loader") for path in paths if path.endswith(".py")
    } == LEGACY_PYTHON_LOADERS
    selectors = [entry["selector"] for entry in entries]
    assert len(selectors) == len(set(selectors)) == 31
    for entry in entries:
        assert entry["required_resources"]
        assert entry["legacy_outputs"]
        assert entry["supported_tasks"]
        assert entry["status"] in {
            "inventoried",
            "implemented",
            "validated",
            "requires_external_data",
        }
        assert isinstance(entry["verification"]["real_sample"], bool)


def test_every_migrated_selector_is_exposed_by_prepare():
    entries = json.loads(COVERAGE.read_text())["entries"]
    assert {entry["selector"] for entry in entries} <= set(DATASETS)


def test_sparse_and_unlabelled_sources_do_not_claim_dense_vessel_ground_truth():
    entries = {entry["selector"]: entry for entry in json.loads(COVERAGE.read_text())["entries"]}
    for selector in ("review", "roc", "vicavr", "inspireavr", "ridb", "ubmi"):
        assert "vessels" not in entries[selector]["supported_tasks"]
        assert "av" not in entries[selector]["supported_tasks"]
    assert "vessels" in entries["erlangen"]["supported_tasks"]
    assert "disc_cup" in entries["aria"]["supported_tasks"]
