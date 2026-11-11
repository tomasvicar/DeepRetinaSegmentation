"""Paired VAL counts from original standalone QA; never weights/GT/TEST evaluation."""

import argparse
import json
import tempfile
from pathlib import Path

from compare_completed_val import unwrap
from compare_frozen_completed_val import check_inventory
from compare_reviewed_test import canonical, inventory, parse, require, sha
from compare_vascular2_val import (
    CANONICAL,
    CONTROL,
    MANIFEST,
    ORIGINAL,
    REPORTS,
    objective,
    validate_source_aggregates,
)

from deepretina.comparison import compare_reports

VERIFIER = "4b568b7e5e66229e378fd8712000af90a05190364563bad6311687e419eb56fe"
CONTRACTS = {
    "vascular4": (
        "0b8762e667c997a2dfd129dc65988c1f26f6b2ad",
        "3ffe949e787045546a9470750b1aedc27c8ae9bee3603db568f719e4b407f24e",
        "85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f",
    ),
    "source72": (
        "72fb321187eca517823083aca80cedace9605570",
        "347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa",
        "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28",
    ),
}
REFERENCES = {
    "original": ORIGINAL,
    "control_terminal60": CONTROL,
    "vascular2_best": REPORTS["best"],
    "vascular2_terminal60": REPORTS["fixed_terminal60"],
}
POLICY = "2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d"
CACHE = "18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e"
VASCULAR4_PROTOCOL = "29d3a9d8a8f6d3a98c1795e53f84fe3005da7c9f9877320a063a9030a9424ec6"


def checked_candidate(
    evidence, qa, protocol, terminal, *, role, contract, protocol_raw, terminal_raw
):
    require(
        contract in CONTRACTS and role in {"best", "fixed_terminal60"},
        "Explicit supported source/role required",
    )
    source, execution, training = CONTRACTS[contract]
    require(
        qa["kind"] == "standalone_unified_completion_qa"
        and qa["schema_version"] == 1
        and qa["verifier_sha256"] == VERIFIER
        and qa["terminal"] is True
        and qa["last_epoch"] == 60
        and qa["finite_best_last_weights_and_last_optimizer"] is True
        and (
            qa["source_commit"],
            qa["execution_code_sha256"],
            qa["checkpoint_training_source_sha256"],
        )
        == (source, execution, training),
        "Original strict standalone QA required",
    )
    require(
        qa["reviewed_protocol_sha256"] == sha(protocol_raw)
        and qa["original_supervisor_terminal_sha256"] == sha(terminal_raw)
        and qa["original_protocol_text"] == protocol_raw.decode()
        and qa["original_supervisor_terminal_text"] == terminal_raw.decode(),
        "Original QA/protocol/terminal byte bindings differ",
    )
    if contract == "vascular4":
        require(
            sha(protocol_raw) == VASCULAR4_PROTOCOL
            and protocol["config"]["vascular_sampling_weight"] == 4.0,
            "Original vascular4 protocol required",
        )
    pins = protocol["source"]
    require(
        (pins["commit"], pins["execution_code_sha256"], pins["training_source_sha256"])
        == (source, execution, training)
        and protocol["kind"] == "reviewed_standalone_unified_protocol",
        "Protocol source/family differs",
    )
    cache = {
        "schema": 1,
        "fingerprint": CACHE,
        "canonical_fingerprint": CANONICAL,
        "allow_canonical_fallback": False,
    }
    require(
        protocol["data"]["geometry_policy_sha256"] == qa["geometry_policy_sha256"] == POLICY
        and protocol["data"]["derived_cache"] == qa["derived_cache"] == cache,
        "Geometry/cache bindings differ",
    )
    config = protocol["config"]
    require(
        config["epochs"] == 60
        and config["epoch_samples"] == 8192
        and config["coverage_first"] is True
        and config["max_steps"] is None
        and config["legacy_initialization"] is None
        and config["allow_historical_exposure"] is False
        and config["purpose"] == "full_data",
        "Full scratch60 recipe required",
    )
    if contract == "source72":
        require(
            config["mixed_precision"] == "bf16" and config["workers"] == 8,
            "Source72 BF16/worker contract differs",
        )
    require(
        terminal["kind"] == "bounded_unified_supervisor_terminal"
        and terminal["status"] == "completed"
        and type(terminal["returncode"]) is int
        and terminal["returncode"] == 0
        and terminal["observed_returncode"] is True
        and terminal["timed_out"] is False
        and terminal["completed_epochs"] == 60
        and terminal["source"] == pins
        and terminal["protocol_sha256"] == sha(protocol_raw)
        and terminal["config_sha256"] == sha(canonical(config))
        and terminal["data"] == protocol["data"],
        "Original clean full60 terminal required",
    )
    require(
        qa["eligible_photos"] == qa["seen_photos"] == 2529
        and qa["minimum_photo_visits"] >= 60
        and qa["eligible_annotation_records"] == qa["contributing_annotation_records"] == 5700
        and qa["all_eligible_annotations_contributed"] is True
        and qa["delivered_samples"] == terminal["delivered_samples"] == 60 * 8192
        and qa["steps"] == terminal["steps"] == 60 * 8192 // config["batch_size"],
        "Actual terminal full-budget coverage required",
    )
    require(
        role != "best" or qa["selected_best_full_annotations"] is True,
        "BEST own annotation coverage is incomplete",
    )
    key = "best.pt" if role == "best" else "last.pt"
    coverage = qa["best_checkpoint_coverage" if role == "best" else "last_checkpoint_coverage"]
    require(
        coverage["eligible_photos"] == coverage["seen_photos"] == 2529
        and coverage["eligible_annotation_records"]
        == coverage["contributing_annotation_records"]
        == 5700
        and coverage["all_eligible_annotations_contributed"] is True
        and coverage["missing_annotation_ids"] == []
        and coverage["minimum_photo_visits"] >= coverage["epoch"],
        "Selected checkpoint lacks own complete annotation coverage; diagnostic scope must remain separate",
    )
    require(
        evidence["checkpoint"]["sha256"]
        == qa["checkpoint_sha256"][key]
        == terminal["artifact_sha256"][key]
        and evidence["checkpoint"]["epoch"] == coverage["epoch"]
        and (role != "fixed_terminal60" or coverage["epoch"] == 60)
        and evidence["checkpoint"]["steps"] == coverage["epoch"] * 8192 // config["batch_size"],
        "BEST/LAST role identity differs",
    )
    expected = {k: v for k, v in config.items() if k != "device"}
    require(
        evidence["configuration"]
        == qa["configuration"]
        == {"values": expected, "sha256": sha(canonical(expected))}
        and evidence["source"]["training_source_sha256"] == training
        and evidence["data"]["manifest_sha256"]
        == qa["manifest_sha256"]
        == protocol["data"]["manifest_sha256"]
        == MANIFEST
        and evidence["data"]["prepared_content_sha256"]
        == qa["canonical_fingerprint"]
        == protocol["data"]["prepared_content_sha256"]
        == CANONICAL,
        "Source/config/canonical binding differs",
    )
    inference = evidence["inference_protocol"]
    require(
        protocol["inference"] == {"threshold": 0.5, "tile_size": 512, "overlap": 64}
        and inference["threshold"] == 0.5
        and inference["tile_size"] == 512
        and inference["overlap"] == 64
        and inference["precision"] == "float32_no_autocast"
        and inference["cup_containment"] == "cup_and_disc",
        "Inference protocol differs",
    )
    return unwrap(evidence)


def compare(inputs, trusted_sha256, output):
    path, output = Path(inputs), Path(output)
    index_raw = path.read_bytes()
    require(sha(index_raw) == trusted_sha256, "External input manifest SHA differs")
    index = parse(index_raw)
    require(
        index["kind"] == "reviewed_standalone_paired_val_inputs" and index["schema_version"] == 1,
        "Reviewed input type required",
    )
    require(not output.exists(), "Fresh output required; never overwrite results")
    paths, blobs = {}, {}
    for key, entry in index["assets"].items():
        paths[key] = path.parent / entry["path"]
        require(paths[key].is_file() and not paths[key].is_symlink(), "Regular input JSON required")
        blobs[key] = paths[key].read_bytes()
        require(sha(blobs[key]) == entry["sha256"], "Input bytes differ: " + key)
    require(
        set(blobs) == {"manifest", "candidate", "qa", "protocol", "terminal", *index["references"]}
        and index["references"]
        and len(set(index["references"])) == len(index["references"]),
        "Explicit input inventory required",
    )
    require(sha(blobs["manifest"]) == MANIFEST, "Canonical manifest differs")
    groups, counts, _, _ = inventory(blobs["manifest"])
    require(counts == {"train": 2529, "val": 770, "test": 856}, "Canonical inventory differs")
    evidence, qa, protocol, terminal = [
        parse(blobs[k]) for k in ("candidate", "qa", "protocol", "terminal")
    ]
    require(
        "owncloud" not in str(output.resolve()).lower()
        and all(
            not output.resolve().is_relative_to(root.resolve())
            for root in (
                paths["manifest"].parent,
                Path(protocol["source"]["root"]),
                Path(protocol["run"]),
            )
        ),
        "Comparison output must not touch canonical/source/training inputs",
    )
    views = {
        "candidate": checked_candidate(
            evidence,
            qa,
            protocol,
            terminal,
            role=index["role"],
            contract=index["source_contract"],
            protocol_raw=blobs["protocol"],
            terminal_raw=blobs["terminal"],
        )
    }
    for reference in index["references"]:
        require(
            reference in REFERENCES and sha(blobs[reference]) == REFERENCES[reference],
            "Known pinned reference required",
        )
        value = parse(blobs[reference])
        if reference == "original":
            require(
                value["family"] == "frozen_original_reference", "Original reference family differs"
            )
            views[reference] = value
        else:
            views[reference] = unwrap(value)
    for view in views.values():
        require(
            view["manifest_sha256"] == MANIFEST and view["prepared_content_sha256"] == CANONICAL,
            "Comparison domain differs",
        )
        check_inventory(view, groups["val"])
        validate_source_aggregates(view)
    results = {}
    with tempfile.TemporaryDirectory(prefix="standalone-val-counts-") as temp:
        tmp = Path(temp)
        for name, view in views.items():
            (tmp / name).write_bytes(canonical(view))
        for reference in index["references"]:
            result = compare_reports(
                tmp / reference,
                tmp / "candidate",
                paths["manifest"],
                repeats=2000,
                seed=42,
                include_sources=True,
                include_av=True,
            )
            require(
                result["photos"] == 770 and result["split_groups"] == 612,
                "Exact grouped VAL inventory differs",
            )
            matched = index["role"] == "fixed_terminal60" and reference.endswith("terminal60")
            result.update(
                kind="standalone_paired_val_counts",
                candidate_role=index["role"],
                reference_role=reference,
                epoch_matching="LAST60_vs_LAST60"
                if matched
                else "different_checkpoint_role_or_original_reference",
                original_standalone_qa_kind=qa["kind"],
                original_standalone_qa_sha256=sha(blobs["qa"]),
                original_supervisor_terminal_sha256=sha(blobs["terminal"]),
                reviewed_standalone_protocol_sha256=sha(blobs["protocol"]),
                source_commit=qa["source_commit"],
                execution_code_sha256=qa["execution_code_sha256"],
                checkpoint_training_source_sha256=qa["checkpoint_training_source_sha256"],
                candidate_configuration=qa["configuration"],
                candidate_checkpoint=evidence["checkpoint"],
                selected_checkpoint_coverage=qa[
                    "best_checkpoint_coverage"
                    if index["role"] == "best"
                    else "last_checkpoint_coverage"
                ],
                full_input_sha256={k: sha(v) for k, v in blobs.items()},
                input_manifest_sha256=trusted_sha256,
                analysis_helper_sha256=sha(Path(__file__).read_bytes()),
                metadata_only=True,
                selection_authorized=False,
                test_evaluation_performed=False,
            )
            result["qualifications"].extend(
                [
                    "612 recoverable split groups do not prove complete biological patient independence.",
                    "VAL is development data; multiple heads/sources/comparisons are not multiplicity-corrected or seed-level uncertainty.",
                    "Original historical exposure and preprocessing differ; distinct CUDA/CPU runtimes are not causally isolated.",
                    "BEST is not a matched-terminal comparison; full60 training completion is not final model selection or TEST authorization.",
                ]
            )
            if reference == "original":
                result["v3_worst6ratio"] = objective(result)
            results[reference] = result
    require(
        path.read_bytes() == index_raw
        and all(p.read_bytes() == blobs[k] for k, p in paths.items()),
        "Inputs changed during comparison",
    )
    output.mkdir(parents=True)
    for name, result in results.items():
        with (output / (index["role"] + "-vs-" + name + ".json")).open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--trusted-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.inputs, args.trusted_input_sha256, args.output)
