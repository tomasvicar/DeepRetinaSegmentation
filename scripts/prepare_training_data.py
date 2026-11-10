"""Build the 21-source training corpus from shared RAW archives, without a private parent.

The intermediate 17-source parent is retained for audit/recovery. Neither it nor
an existing final output is overwritten. This creates a new corpus identity; it
is not a promise to reproduce the authors' historical corpus fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ADDITIONAL_DATASETS = ("drions", "gamma", "idrid_disc", "biomisa")


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "_raw_training_v2", Path(__file__).with_name("prepare_corpus_v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(data_root: Path, parent: Path, output: Path, policy: Path, *, biomisa_policy: str):
    """Prepare, repair known patient groups, audit, append, and audit all sources."""
    data_root, parent, output, policy = [
        Path(p).resolve() for p in (data_root, parent, output, policy)
    ]
    if parent.exists() or output.exists():
        raise ValueError(
            "Parent and final output must both be new directories; use new paths after failure"
        )
    if any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a)
        for a, b in ((parent, output), (parent, data_root), (output, data_root))
    ):
        raise ValueError(
            "RAW, intermediate parent, and final output must be independent directories"
        )
    v2 = _load_builder()
    if biomisa_policy != v2.FIXED_POLICY:
        raise ValueError("Explicit reviewed BIOMISA filename-role policy is required")
    if hashlib.sha256(policy.read_bytes()).hexdigest() != v2.GEOMETRY_POLICY_SHA256:
        raise ValueError("Geometry policy differs from the reviewed version")
    # Fail early before spending time decoding the parent sources.
    missing = [
        name
        for name in ("DRIONS-DB.zip", "GAMMA.zip", "IDRiD.zip", "BIOMISA.zip")
        if not (data_root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing additional RAW archives: {', '.join(missing)}")
    v2.base.prepare_unified_data(data_root, parent)
    v2.base.regroup_patients(parent, data_root, seed=42, train_only_datasets=["riga", "uoadr"])
    parent_audit = v2.base.audit_prepared(parent)
    parent_pin = parent_audit["prepared_content_sha256"]
    final_audit = v2.append(
        parent,
        output,
        data_root,
        policy,
        biomisa_policy=biomisa_policy,
        expected_parent=parent_pin,
    )
    expected_sources = set(v2.base.DEFAULT_DATASETS) | set(ADDITIONAL_DATASETS)
    supported_sources = {
        count["dataset"]
        for count in final_audit["head_counts"]
        if count["split"] == "train" and count["positive_pixels"] > 0
    }
    missing_support = expected_sources - supported_sources
    if missing_support:
        raise ValueError(
            f"No positive TRAIN supervision for required sources: {sorted(missing_support)}"
        )
    proof = {
        "schema_version": 1,
        "workflow": "shared_raw_to_21_sources",
        "historical_identity_reproduction_claim": False,
        "parent_content_sha256": parent_pin,
        "final_content_sha256": final_audit["prepared_content_sha256"],
        "geometry_policy_sha256": v2.GEOMETRY_POLICY_SHA256,
        "biomisa_label_policy": biomisa_policy,
        "seed": 42,
        "training_sources": sorted(supported_sources),
        "parent": str(parent),
        "final": str(output),
        "parent_audit": parent_audit,
        "final_audit": final_audit,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    v2.write_json(output / "raw-workflow.json", proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--parent-output",
        type=Path,
        required=True,
        help="New intermediate 17-source corpus; retained for audit and recovery",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry-policy",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports/data/corpus-v2-geometry-policy/policy.json",
    )
    parser.add_argument(
        "--biomisa-label-policy", required=True, choices=["named_cup_is_disc_named_disc_is_cup"]
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                args.data_root,
                args.parent_output,
                args.output,
                args.geometry_policy,
                biomisa_policy=args.biomisa_label_policy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
