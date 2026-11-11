"""Metadata-only postcheck example, not an evaluator/selection/completion gate."""

SOURCE = "0b8762e667c997a2dfd129dc65988c1f26f6b2ad"
EXECUTION = "3ffe949e787045546a9470750b1aedc27c8ae9bee3603db568f719e4b407f24e"
TRAINING = "85441afac96f47165df55505d06a08742183b9686fb1c608225166810a23a00f"
PROTOCOL = "f4ca4a1aaa7f2a01ad3f6dc818e5a61087fd299cd9ad436941b938bbfd76ec8c"


def postcheck(report, qa, role):
    if (
        qa["kind"] != "standalone_unified_completion_qa"
        or qa["terminal"] is not True
        or qa["last_epoch"] != 60
        or qa["source_commit"] != SOURCE
        or qa["execution_code_sha256"] != EXECUTION
        or qa["checkpoint_training_source_sha256"] != TRAINING
        or qa["reviewed_protocol_sha256"] != PROTOCOL
    ):
        raise ValueError("Require original strict standalone QA identity, never retag pool/partial")
    if role not in ("best", "fixed_terminal60"):
        raise ValueError("Explicit checkpoint role required")
    key = "best.pt" if role == "best" else "last.pt"
    coverage = qa["best_checkpoint_coverage" if role == "best" else "last_checkpoint_coverage"]
    if (
        report["kind"] != "unified_validation_evidence"
        or report["metrics"]["split"] != "val"
        or report["metrics"]["photos"] != 770
        or len(report["metrics"]["per_photo"]) != 770
        or len({p["photo_id"] for p in report["metrics"]["per_photo"]}) != 770
        or report["checkpoint"]["sha256"] != qa["checkpoint_sha256"][key]
        or report["checkpoint"]["epoch"] != coverage["epoch"]
        or role == "fixed_terminal60"
        and report["checkpoint"]["epoch"] != 60
    ):
        raise ValueError("Actual nested VAL/checkpoint identity differs")
    return {
        "role": role,
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "selected_checkpoint_coverage": coverage,
        "selected_best_full_annotations": qa["selected_best_full_annotations"],
        "original_qa_kind": qa["kind"],
        "selection_or_test_authorized": False,
    }
