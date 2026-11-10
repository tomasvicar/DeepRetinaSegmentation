"""Paired saved-count E8 loss-weight comparison only; no model or image reads."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["reference", "candidate", "manifest", "helper", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    assert sha(a.reference) == "3c0af8cd48405ce2d2d04a1769ed14a3e463edd410f65dab6f44432062016fec"
    assert sha(a.candidate) == "8cec99ae99c39cb054f059693f461bd93744d8746642550f0ab17dc1a8d67032"
    assert sha(a.helper) == "72aaa618c3b75d7c1abc7993aa8a66e411630f90a4b59f90e37552d049415221"
    assert sha(a.manifest) == "529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69"
    assert not a.output.exists()
    spec = importlib.util.spec_from_file_location("paired_av_weight", a.helper)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    reference = json.loads(a.reference.read_text())
    candidate = json.loads(a.candidate.read_text())
    assert reference["selection_epoch"] == candidate["selection_epoch"] == 8
    assert not any(
        "av_conditional_counts" in row for r in [reference, candidate] for row in r["per_photo"]
    )
    start = time.perf_counter()
    r = h.compare_reports(a.reference, a.candidate, a.manifest, repeats=2000, seed=42)
    assert r["photos"] == 770 and r["split_groups"] == 612
    r.update(
        performed_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - start,
        python=platform.python_version(),
        numpy=np.__version__,
        comparison_scope="Same87 oldGTX1060 LR.0003 seed42 mildaugmentation data6103; sole planned configdelta av_weight1to2; E8screens, not finalmodels",
        original_comparison_module_sha256="10cbf98db427be2d2e841b59af7b3fd6777ef1835ec30148a2eca3430a392086",
        comparison_extension="Added pooled mean of5 Dice to SAME shared paired bootstrap draws; original macro/perhead algorithms retained",
        protocol_sha256=sha(__file__),
        conditional_av_point_only={
            "reference": reference["av_conditional_accuracy"],
            "candidate": candidate["av_conditional_accuracy"],
            "difference": candidate["av_conditional_accuracy"]
            - reference["av_conditional_accuracy"],
            "paired_ci": None,
            "reason": "Source87 saved per-photo reports lack ungated conditional correct/scored counts; gated TP/FP/FN cannot recover them",
        },
        qualifications=[
            "Exploratory E8 validation screens selected among multiple configurations; intervals are not multiplicity-adjusted",
            "Source-stratified reviewed split_group IDs preserve known aliases; they are not proven independent patients",
            "Uncertainty is conditional on these two seed42 checkpoint realizations, not training-seed robustness or future60epoch outcome",
            "One task or overall gain does not establish all-task superiority; vascular regressions must be disclosed",
            "No TEST scores, images, masks, models or new forwards were read/computed; only saved counts and manifestgroup metadata",
            "Promotion merits a reviewed follow-up, not a final superiority claim",
        ],
        no_new_inference=True,
        no_test_evaluation=True,
        patients_proven_independent=False,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "macro": r["macro_dataset_mean_dice"],
                "pooled_mean": r["pooled_mean_dice"],
                "channels": r["channels"],
                "seconds": r["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
