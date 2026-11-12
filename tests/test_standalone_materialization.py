"""Synthetic namespace/byte-contract checks only: no model, corpus or TEST reads."""

import copy
import hashlib
import json

import pytest

from deepretina import freeze_eligibility as f
from deepretina.selection import SelectionError


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def contracts(tmp_path):
    profile = f.PROFILES["standalone_72"]
    config = {"device": "cuda:0", "base": 24, "tile_size": 512, "overlap": 64}
    protocol = {
        "schema_version": 1,
        "kind": "reviewed_standalone_unified_protocol",
        "source": {
            "root": "/workspace",
            "commit": profile[1],
            "execution_code_sha256": profile[2],
            "training_source_sha256": profile[3],
        },
        "run": "/runs/train",
        "config": config,
        "data": {
            "manifest_sha256": f.MANIFEST,
            "prepared_content_sha256": f.CONTENT,
            "geometry_policy_sha256": f.GEOMETRY,
            "derived_cache": None,
        },
        "budget": {
            "epochs": 60,
            "epoch_samples": 8192,
            "train_photos": 2529,
            "train_annotation_records": 5700,
            "dataset_count": 21,
        },
        "inference": {"threshold": 0.5, "tile_size": 512, "overlap": 64},
    }
    raw_protocol = encoded(protocol)
    terminal = {
        "schema_version": 1,
        "kind": "bounded_unified_supervisor_terminal",
        "status": "completed",
        "returncode": 0,
        "observed_returncode": True,
        "timed_out": False,
        "test_evaluation_performed": False,
        "source": protocol["source"],
        "run": protocol["run"],
        "data": protocol["data"],
        "budget": protocol["budget"],
        "protocol_sha256": f.gate.digest(raw_protocol),
        "config_sha256": f._sha_json(config),
        "completed_epochs": 60,
        "steps": 122880,
        "delivered_samples": 491520,
    }
    raw_terminal = encoded(terminal)
    values = {k: v for k, v in config.items() if k != "device"}
    qa = {
        "verifier_sha256": f.VERIFIER,
        "source_git_verified": True,
        "reviewed_protocol_sha256": f.gate.digest(raw_protocol),
        "original_supervisor_terminal_sha256": f.gate.digest(raw_terminal),
        "original_protocol_text": raw_protocol.decode(),
        "original_supervisor_terminal_text": raw_terminal.decode(),
        "configuration": {"values": values, "sha256": f._sha_json(values)},
    }
    root = tmp_path / "new-controller-namespace"
    root.mkdir()
    paths = {}
    for name, raw in (("protocol", raw_protocol), ("terminal", raw_terminal), ("qa", encoded(qa))):
        paths[name] = root / (name + ".json")
        paths[name].write_bytes(raw)
    return profile, protocol, terminal, qa, paths


def checked(contracts):
    profile, protocol, terminal, qa, paths = contracts
    protocol_raw, actual_protocol = f._pinned_json(
        paths["protocol"], f.gate.digest(encoded(protocol)), "protocol"
    )
    terminal_raw, actual_terminal = f._pinned_json(
        paths["terminal"], f.gate.digest(encoded(terminal)), "terminal"
    )
    _, actual_qa = f._pinned_json(paths["qa"], f.gate.digest(encoded(qa)), "QA")
    return f._standalone(
        actual_qa,
        profile,
        {"config": protocol["config"]},
        {"config": protocol["config"], "steps": 122880, "derived_cache": None},
        actual_protocol,
        protocol_raw,
        actual_terminal,
        terminal_raw,
    )


def test_original_sif_path_text_survives_materialization_without_retagging(contracts, tmp_path):
    expected = checked(contracts)
    paths = contracts[-1]
    before = {key: path.read_bytes() for key, path in paths.items()}
    relocated = tmp_path / "another-host" / "readonly-contracts"
    relocated.mkdir(parents=True)
    relocated_paths = {}
    for key, path in paths.items():
        target = relocated / path.name
        target.write_bytes(path.read_bytes())
        target.chmod(0o444)
        relocated_paths[key] = target
    result = checked((*contracts[:-1], relocated_paths))
    assert result == expected
    assert b'"root":"/workspace"' in before["protocol"]
    assert b'"run":"/runs/train"' in before["terminal"]
    assert all(path.read_bytes() == before[key] for key, path in relocated_paths.items())


@pytest.mark.parametrize("name", ["protocol", "terminal", "qa"])
def test_rewriting_original_path_or_text_without_external_repin_is_rejected(contracts, name):
    paths = contracts[-1]
    raw = paths[name].read_bytes()
    paths[name].write_bytes(raw.replace(b"/workspace", b"/new/source"))
    with pytest.raises(SelectionError, match="Externally pinned bytes differ"):
        checked(contracts)


def test_repinning_only_relocated_protocol_does_not_rewrite_original_qa(contracts):
    profile, protocol, terminal, qa, paths = copy.deepcopy(contracts)
    protocol["source"] = {**protocol["source"], "root": "/new/source"}
    paths["protocol"].write_bytes(encoded(protocol))
    with pytest.raises(SelectionError, match="Original standalone protocol/terminal bytes differ"):
        checked((profile, protocol, terminal, qa, paths))


def test_seven_run_files_plus_original_qa_keep_role_and_byte_pins(tmp_path):
    # Marker bytes deliberately are NOT checkpoints and are never deserialized.
    names = (
        "best.pt",
        "last.pt",
        "metrics.csv",
        "training_summary.json",
        "run.json",
        "coverage.json",
        "validation.json",
    )
    artifacts = {name: ("synthetic-original:" + name).encode() for name in names}
    terminal = {
        "artifact_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()
        }
    }
    qa = {
        "checkpoint_sha256": {
            name: terminal["artifact_sha256"][name] for name in ("best.pt", "last.pt")
        },
        "sidecar_sha256": {
            name: terminal["artifact_sha256"][name] for name in names if not name.endswith(".pt")
        },
    }
    for name, raw in artifacts.items():
        (tmp_path / name).write_bytes(raw)
    (tmp_path / "full60-QA.json").write_bytes(encoded(qa))

    def verify():
        for name in names:
            raw = (tmp_path / name).read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            assert actual == terminal["artifact_sha256"][name]
            assert (
                actual
                == (qa["checkpoint_sha256"] if name.endswith(".pt") else qa["sidecar_sha256"])[name]
            )

    verify()
    (tmp_path / "best.pt").write_bytes(artifacts["last.pt"])
    with pytest.raises(AssertionError):
        verify()
