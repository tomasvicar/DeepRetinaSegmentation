"""Standalone, root-reviewed full60 completion QA on an actual Git source host.

CPU safe loading and encoded-byte/metadata checks only; no forward or GT decoding.
This is not a pool receipt adapter and never creates a supervisor terminal record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
CONTENT = "00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52"
POLICY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
ARTIFACTS = (
    "best.pt",
    "last.pt",
    "metrics.csv",
    "training_summary.json",
    "run.json",
    "coverage.json",
    "validation.json",
)
BUDGET = {
    "epochs": 60,
    "epoch_samples": 8192,
    "train_photos": 2529,
    "train_annotation_records": 5700,
    "dataset_count": 21,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_pinned(path, expected, label):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, "Externally reviewed original bytes differ: " + label)

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    return raw, json.loads(raw, parse_constant=invalid, object_pairs_hook=pairs)


def hex_digest(value, length=64):
    return isinstance(value, str) and len(value) == length and set(value) <= set("0123456789abcdef")


def check_contracts(protocol, terminal, *, protocol_sha, source, run):
    require(
        protocol.get("schema_version") == 1
        and protocol.get("kind") == "reviewed_standalone_unified_protocol",
        "Require reviewed standalone protocol family",
    )
    require(
        terminal.get("schema_version") == 1
        and terminal.get("kind") == "bounded_unified_supervisor_terminal",
        "Require original standalone supervisor terminal, never a pool/waiter receipt",
    )
    require(
        terminal.get("status") == "completed"
        and type(terminal.get("returncode")) is int
        and terminal["returncode"] == 0
        and terminal.get("timed_out") is False
        and terminal.get("test_evaluation_performed") is False
        and terminal.get("completed_epochs") == 60,
        "Require actual clean completed60 supervisor terminal",
    )
    require(
        canonical(protocol["budget"]) == canonical(BUDGET), "Require full V2 TRAIN/budget inventory"
    )
    require(
        protocol["source"]["root"] == str(source)
        and protocol["run"] == str(run)
        and terminal["protocol_sha256"] == protocol_sha
        and terminal["source"] == protocol["source"]
        and terminal["run"] == protocol["run"]
        and terminal["budget"] == protocol["budget"]
        and terminal["config_sha256"] == sha(canonical(protocol["config"])),
        "Supervisor source/run/protocol/config/budget binding differs",
    )
    source_pins = protocol["source"]
    require(
        hex_digest(source_pins["commit"], 40)
        and all(
            hex_digest(source_pins[k]) for k in ["execution_code_sha256", "training_source_sha256"]
        ),
        "Full reviewed source SHA pins required",
    )
    data = protocol["data"]
    require(
        data["manifest_sha256"] == MANIFEST
        and data["prepared_content_sha256"] == CONTENT
        and data["geometry_policy_sha256"] == POLICY,
        "Require exact reviewed V2 canonical/policy identities",
    )
    require(terminal["data"] == data, "Supervisor canonical/policy/cache binding differs")
    cache = data["derived_cache"]
    require(
        cache is None
        or cache
        == {
            "schema": 1,
            "fingerprint": CACHE,
            "canonical_fingerprint": CONTENT,
            "allow_canonical_fallback": False,
        },
        "Require pinned V2 cache or explicit canonical None",
    )
    config = protocol["config"]
    require(
        config["purpose"] == "full_data"
        and config["epochs"] == 60
        and config["epoch_samples"] == 8192
        and config["coverage_first"] is True
        and config["legacy_initialization"] is None
        and config["allow_historical_exposure"] is False
        and config["max_steps"] is None,
        "Require independent full-data60 recipe",
    )
    require(
        protocol["inference"]
        == {"threshold": 0.5, "tile_size": config["tile_size"], "overlap": config["overlap"]},
        "Inference must retain reviewed fixed0.5 config",
    )
    require(
        set(terminal["artifact_sha256"]) == set(ARTIFACTS)
        and all(hex_digest(v) for v in terminal["artifact_sha256"].values()),
        "Supervisor final artifact SHA inventory differs",
    )


def check_source(source, pins):
    """Actual Git and execution bytes; no container/fake-Git fallback."""
    top = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    require(Path(top).resolve() == source, "Require actual source Git working tree")
    require(
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        == pins["commit"],
        "Reviewed source commit differs",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "Tracked source checkout is dirty",
    )
    digest = hashlib.sha256()
    for path in sorted(
        [
            *source.glob("src/**/*.py"),
            *source.glob("scripts/*.py"),
            source / "pyproject.toml",
            source / "uv.lock",
        ]
    ):
        digest.update(path.relative_to(source).as_posix().encode() + b"\0" + path.read_bytes())
    require(
        digest.hexdigest() == pins["execution_code_sha256"],
        "Reviewed executable source bytes differ",
    )


def optimizer_binding(gate, snapshot, last, config):
    """Use the safe strict loader's native parameter order; no model forward."""
    with gate.torch.device("cpu"), gate.torch.random.fork_rng(devices=[]):
        model, _, _ = gate.load_unified(io.BytesIO(snapshot), device="cpu")
    optimizer = last["training_state"]["optimizer"]
    groups = optimizer["param_groups"]
    require(len(groups) == 1, "Require actual train_unified single AdamW parameter group")
    ids = groups[0]["params"]
    parameters = list(model.parameters())
    require(
        len(ids) == len(set(ids)) == len(parameters) and set(optimizer["state"]) == set(ids),
        "Optimizer/model parameter identity inventory differs",
    )
    for pid, parameter in zip(ids, parameters, strict=True):
        value = optimizer["state"][pid]
        require(float(value["step"]) == last["steps"], "Stale optimizer parameter step")
        require(
            all(
                gate.torch.is_tensor(value[name])
                and value[name].shape == parameter.shape
                and value[name].dtype == parameter.dtype
                for name in ["exp_avg", "exp_avg_sq"]
            ),
            "Optimizer moment shape differs from model",
        )
    minimum = config["lr"] * config["min_lr_ratio"]
    require(
        config["scheduler"] == "cosine"
        and math.isclose(groups[0]["lr"], minimum, rel_tol=1e-10, abs_tol=1e-12)
        and groups[0]["weight_decay"] == config["weight_decay"],
        "Terminal AdamW/schedule parameters differ",
    )
    require(gate._finite(last["training_state"]), "Nonfinite last optimizer/scheduler/RNG state")


def verify(
    *,
    source,
    run,
    manifest,
    protocol_path,
    trusted_protocol_sha256,
    terminal_path,
    trusted_terminal_sha256,
    output,
):
    require(
        not Path(output).exists() and not Path(output).is_symlink(),
        "Preserve existing completion output",
    )
    source, run, manifest, output = (Path(p).resolve() for p in [source, run, manifest, output])
    require(
        all(not output.is_relative_to(root) for root in [source, run, manifest.parent]),
        "Write completion evidence outside source/run/canonical inputs",
    )
    protocol_raw, protocol = read_pinned(protocol_path, trusted_protocol_sha256, "protocol")
    terminal_raw, terminal = read_pinned(
        terminal_path, trusted_terminal_sha256, "original supervisor terminal"
    )
    check_contracts(
        protocol, terminal, protocol_sha=trusted_protocol_sha256, source=source, run=run
    )
    check_source(source, protocol["source"])
    raw = {name: (run / name).read_bytes() for name in ARTIFACTS}
    require(
        all(sha(blob) == terminal["artifact_sha256"][name] for name, blob in raw.items()),
        "Original supervisor artifact bytes changed",
    )
    # Source/terminal checks above precede Torch loading and all canonical reading.
    import torch

    from deepretina import selection as gate
    from deepretina.unified import UnifiedConfig, _training_source_sha256

    require(
        Path(gate.__file__).resolve() == source / "src/deepretina/selection.py",
        "Selection imported from a different source checkout",
    )
    require(
        _training_source_sha256() == protocol["source"]["training_source_sha256"],
        "Reviewed training source bytes differ",
    )
    require(not torch.cuda.is_initialized(), "CUDA must not already be initialized")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

    def blocked(*args, **kwargs):
        raise AssertionError("Standalone completion verifier forbids CUDA initialization")

    config = protocol["config"]
    checked_config = UnifiedConfig(**config)
    checked_config.validate()
    require(
        canonical(config) == canonical(gate.asdict(checked_config)),
        "Reviewed protocol must materialize the exact full configuration",
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, "init", blocked))
        stack.enter_context(patch.object(torch.cuda, "_lazy_init", blocked))
        best_raw, best = gate.checkpoint_snapshot(run / "best.pt")
        last_raw, last = gate.checkpoint_snapshot(run / "last.pt")
        require(
            best_raw == raw["best.pt"] and last_raw == raw["last.pt"],
            "Checkpoint changed during safe snapshot",
        )
        config = protocol["config"]
        UnifiedConfig(**config).validate()
        for payload in [best, last]:
            require(
                canonical(payload["config"]) == canonical(config),
                "Checkpoint exact reviewed configuration differs",
            )
            require(
                payload["manifest_sha256"] == MANIFEST
                and payload["prepared_content_sha256"] == CONTENT
                and payload["training_source_sha256"]
                == protocol["source"]["training_source_sha256"]
                and payload.get("derived_cache") == protocol["data"]["derived_cache"],
                "Checkpoint source/data/cache identity differs",
            )
            require(
                payload.get("geometry_policy", {}).get("sha256") == POLICY,
                "Checkpoint embedded geometry policy differs",
            )
            require(
                payload["lineage"].get("initialization") == "scratch",
                "Standalone recipe requires explicit scratch lineage",
            )
        require(best["lineage"] == last["lineage"], "Best/last lineage differs")
        data, groups = gate.data_identity(manifest, best)
        lease = gate._canonical_metadata(manifest)
        require(
            data["manifest_sha256"] == MANIFEST
            and data["prepared_content_sha256"] == CONTENT
            and len(groups["train"]) == 2529,
            "Exact canonical TRAIN domain differs",
        )
        rows = [row for records in groups["train"].values() for row in records]
        require(
            len(rows) == 5700 and len({r["dataset"] for r in rows}) == 21,
            "Exact5700/21-source eligible TRAIN inventory differs",
        )
        completion = gate.completion_evidence(
            best, last, raw["metrics.csv"], raw["training_summary.json"], groups["train"]
        )
        coverage = last["coverage"]
        visits = coverage["photo_visits"]
        annotations = coverage["annotation_visits"]
        require(
            set(annotations) == {r["id"] for r in rows}
            and coverage["eligible_annotation_records"]
            == coverage["contributing_annotation_records"]
            == 5700
            and all(type(v) is int and v > 0 for v in annotations.values()),
            "Every eligible TRAIN annotation must actually contribute",
        )
        require(
            min(visits.values()) >= 60 and sum(visits.values()) == 60 * 8192,
            "Actual full60 photo coverage/sample budget differs",
        )
        steps = 60 * 8192 // config["batch_size"]
        require(
            last["steps"] == steps == terminal["steps"]
            and terminal["delivered_samples"] == 60 * 8192,
            "Terminal actual optimizer/sample budget differs",
        )
        csv_rows = list(gate.csv.DictReader(gate.io.StringIO(raw["metrics.csv"].decode())))
        require(
            all(
                int(row["steps"]) == epoch * 8192 // config["batch_size"]
                for epoch, row in enumerate(csv_rows, 1)
            ),
            "Per-epoch optimizer step budget differs",
        )
        scores = [float(row["val_selection_score"]) for row in csv_rows]
        require(
            best["epoch"] == scores.index(max(scores)) + 1,
            "Best checkpoint is not first strict maximum",
        )
        optimizer_binding(gate, last_raw, last, config)
        run_metadata = gate.parse_json(raw["run.json"])
        for key in [
            "config",
            "manifest_sha256",
            "prepared_content_sha256",
            "training_source_sha256",
            "derived_cache",
            "geometry_policy",
        ]:
            expected = config if key == "config" else best[key]
            require(run_metadata[key] == expected, "Run metadata differs: " + key)
        require(
            gate.parse_json(raw["coverage.json"]) == coverage, "Actual coverage sidecar differs"
        )
        require(
            gate.parse_json(raw["validation.json"]) == best["validation_metrics"],
            "Best VAL sidecar metadata differs",
        )
        require(not torch.cuda.is_initialized(), "CUDA initialized during completion verification")
        require(
            gate._canonical_metadata(manifest) == lease,
            "Canonical encoded inputs/metadata changed during verification",
        )
    check_source(source, protocol["source"])
    require(
        all((run / name).read_bytes() == blob for name, blob in raw.items()),
        "Run changed before completion publication",
    )
    require(
        Path(protocol_path).read_bytes() == protocol_raw
        and Path(terminal_path).read_bytes() == terminal_raw,
        "Reviewed original artifacts changed during verification",
    )
    result = {
        "kind": "standalone_unified_completion_qa",
        "schema_version": 1,
        "terminal": True,
        "last_epoch": 60,
        "finite_best_last_weights_and_last_optimizer": True,
        "best_epoch": best["epoch"],
        "best_score": best["val_selection_score"],
        "steps": steps,
        "delivered_samples": 60 * 8192,
        "eligible_photos": 2529,
        "seen_photos": 2529,
        "minimum_photo_visits": min(visits.values()),
        "eligible_annotation_records": 5700,
        "contributing_annotation_records": 5700,
        "minimum_annotation_visits": min(annotations.values()),
        "all_eligible_annotations_contributed": True,
        "by_dataset": coverage["by_dataset"],
        "checkpoint_sha256": {name: sha(raw[name]) for name in ["best.pt", "last.pt"]},
        "sidecar_sha256": {name: sha(raw[name]) for name in ARTIFACTS if not name.endswith(".pt")},
        "source_commit": protocol["source"]["commit"],
        "execution_code_sha256": protocol["source"]["execution_code_sha256"],
        "checkpoint_training_source_sha256": protocol["source"]["training_source_sha256"],
        "manifest_sha256": MANIFEST,
        "canonical_fingerprint": CONTENT,
        "geometry_policy_sha256": POLICY,
        "derived_cache": protocol["data"]["derived_cache"],
        "configuration": gate.config_identity(best),
        "training_completion": completion,
        "reviewed_protocol_sha256": trusted_protocol_sha256,
        "original_supervisor_terminal_sha256": trusted_terminal_sha256,
        "original_protocol_text": protocol_raw.decode(),
        "original_supervisor_terminal_text": terminal_raw.decode(),
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "source_git_verified": True,
        "no_gpu_forward": True,
        "no_gt_array_decode": True,
        "no_test_evaluation": True,
        "scope": "Standalone full60 finite completion QA only; not a pool/Optuna receipt or selection decision; canonical encoded bytes hashed without GT decoding",
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "run", "manifest", "protocol", "terminal", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ["trusted-protocol-sha256", "trusted-terminal-sha256"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    verify(
        source=args.source,
        run=args.run,
        manifest=args.manifest,
        protocol_path=args.protocol,
        trusted_protocol_sha256=args.trusted_protocol_sha256,
        terminal_path=args.terminal,
        trusted_terminal_sha256=args.trusted_terminal_sha256,
        output=args.output,
    )


if __name__ == "__main__":
    main()
