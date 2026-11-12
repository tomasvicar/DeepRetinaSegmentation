"""Count-only controlled loss2 LAST60 comparison; never runs an evaluator/model."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAST = "1cd10ac6faa9f4096e5a9e0cd045deceefa40015a63db883a9add1e25aba5286"
CONTROL_REPORT = "7570255b5eac5855d6bd3a1068ff12effbc66c7f05c6699ba26f82fc25c022be"
ORIGINAL_REPORT = "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"
MATH_SHA = "8fcd87112ba25a6093337088ad5dfe2af57ae8337cc2f4a05eb031acbbf7fe4e"
DENSE_SHA = "13bc973a6c209877e2a250b7f66a79bd65c12029fb183d2c3e355d8327fdb304"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, expected):
    assert sha(path) == expected, "Original byte pin differs"
    return json.loads(Path(path).read_bytes())


def load(name, path, expected):
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def matched_scope(candidate, control):
    assert candidate["kind"] == control["kind"] == "unified_validation_evidence"
    assert candidate["checkpoint"]["epoch"] == control["checkpoint"]["epoch"] == 60
    assert candidate["checkpoint"]["sha256"] == LAST
    assert candidate["source"] == control["source"]
    assert candidate["data"] == control["data"]
    assert candidate["inference_protocol"] == control["inference_protocol"]
    p = candidate["inference_protocol"]
    assert p["precision"] == "float32_no_autocast" and p["threshold"] == 0.5
    assert p["tile_size"] == 512 and p["overlap"] == 64 and p["pixels_per_degree"] == 25
    assert p["preprocessing_profile"] == "legacy25_raw" and "tta" not in p
    a, b = candidate["configuration"]["values"], control["configuration"]["values"]
    assert a == dict(b, vessel_loss_weight=2.0) and b["vessel_loss_weight"] == 1.0
    assert a["vessel_pos_weight"] == b["vessel_pos_weight"] == 1.0
    assert (
        candidate["source"]["training_source_sha256"]
        == "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28"
    )
    assert (
        candidate["source"]["evaluation_source_sha256"]
        == "b269a5e5214b8422a5b8028ff6922ff5c19137d7e0354afcd8fbef1c74e3662f"
    )


def observed_zero(terminal, outer):
    assert terminal["status"] == "completed_fixed_terminal60_fresh_VAL"
    assert len(terminal["stages"]) == 1 and terminal["stages"][0]["role"] == "fixed_terminal60"
    for record in [outer, terminal["stages"][0]]:
        assert type(record["returncode"]) is int and record["returncode"] == 0
        assert record["observed_returncode"] is True
    assert outer["timed_out"] is False and outer.get("error") is None
    assert terminal.get("error") is None


def main(args):
    cmp = load("reviewed_count_comparison", ROOT / "src/deepretina/comparison.py", MATH_SHA)
    dense = load(
        "reviewed_dense_supplement",
        ROOT / "reports/dense-vessel-support-20261112/analyze.py",
        DENSE_SHA,
    )
    candidate = pinned(args.fresh_report, args.fresh_report_sha256)
    control = pinned(args.control_report, CONTROL_REPORT)
    original_path = ROOT / "reports/frozen-original-val-nearest/validation.json"
    original = pinned(original_path, ORIGINAL_REPORT)
    terminal = pinned(args.terminal, args.terminal_sha256)
    outer = pinned(args.outer_exit, args.outer_exit_sha256)
    observed_zero(terminal, outer)
    postcheck = pinned(args.postcheck, args.postcheck_sha256)
    runtime = pinned(args.runtime, args.runtime_sha256)
    assert postcheck["role"] == "fixed_terminal60" and postcheck["checkpoint_sha256"] == LAST
    assert postcheck["report_sha256"] == args.fresh_report_sha256
    coverage = postcheck["own_coverage"]
    assert (
        coverage["epoch"] == 60 and coverage["eligible_photos"] == coverage["seen_photos"] == 2529
    )
    assert (
        coverage["eligible_annotation_records"]
        == coverage["contributing_annotation_records"]
        == 5700
    )
    assert (
        coverage["all_eligible_annotations_contributed"] is True
        and coverage["missing_annotation_ids"] == []
    )
    assert coverage["minimum_photo_visits"] >= 60
    assert (
        runtime["intra"] == 8
        and runtime["interop"] == 1
        and runtime["inference_precision"] == "FP32_no_autocast"
    )
    assert runtime["cuda_autocast"] is False and runtime["cpu_autocast"] is False
    matched_scope(candidate, control)
    assert sha(args.manifest) == dense.MANIFEST
    manifest = json.loads(args.manifest.read_bytes())
    rows = manifest["records"] if isinstance(manifest, dict) else manifest
    metadata = defaultdict(list)
    for r in rows:
        if (
            r["split"] == "val"
            and r["task"] in {"vessels", "av", "disc_cup"}
            and r.get("usable_for_training", True)
            and not r.get("exclude_from_training", False)
        ):
            metadata[r["photo_id"]].append(r)
    assert len(metadata) == 770
    records = {"loss2_LAST60": candidate, "TPE5_LAST60": control, "original182": original}
    lookups, points, input_pins = {}, {}, {}
    originals = {
        "loss2_LAST60": (args.fresh_report, args.fresh_report_sha256),
        "TPE5_LAST60": (args.control_report, CONTROL_REPORT),
        "original182": (original_path, ORIGINAL_REPORT),
    }
    for name, d in records.items():
        m = d.get("metrics", d)
        assert (
            d.get("data", {}).get("manifest_sha256") or m.get("manifest_sha256")
        ) == dense.MANIFEST
        assert (
            d.get("data", {}).get("prepared_content_sha256") or m.get("prepared_content_sha256")
        ) == dense.CANONICAL
        lookup = {p["photo_id"]: p for p in m["per_photo"]}
        assert len(lookup) == len(m["per_photo"]) == 770 and set(lookup) == set(metadata)
        for p, item in lookup.items():
            assert set(item["record_ids"]) == {r["id"] for r in metadata[p]}
        lookups[name] = lookup
        points[name] = {
            **{h: m["channels"][h]["dice"] for h in cmp.CHANNELS},
            "conditional_AV": m["av_conditional_accuracy"],
            "macro": m["macro_dataset_mean_dice"],
        }
        input_pins[name] = {
            "whole_original_sha256": originals[name][1],
            "kind": d.get("kind", d.get("family")),
            "checkpoint": d.get("checkpoint"),
            "source": d.get("source"),
            "inference_protocol": d.get("inference_protocol"),
            "configuration": d.get("configuration"),
        }
    base = lookups["original182"]
    for lookup in lookups.values():
        dense.validate_pair(base, lookup)
    vessel = sorted(p for p, v in base.items() if v["channels"]["vessel"]["valid_pixels"] > 0)
    sparse = [p for p in vessel if base[p]["channel_datasets"]["vessel"] == "avrdb"]
    keep = [p for p in vessel if p not in set(sparse)]
    assert (len(vessel), len(keep), len(sparse)) == (107, 87, 20)
    support = dense.support(keep, base, metadata)
    assert support["known_split_groups"] == 55 and len(support["sources"]) == 8
    domains = {}
    for domain, ids in [
        ("denseOnly_PRIMARY", keep),
        ("includeAll", vessel),
        ("positiveOnly_AVRDB_SEPARATE", sparse),
    ]:
        d = {"support": dense.support(ids, base, metadata), "models": {}, "contrasts": {}}
        for name, lookup in lookups.items():

            def pooled(selected, lookup=lookup):
                return dense.scores(
                    sum(
                        (dense.checked_counts(lookup[p]["channels"]["vessel"]) for p in selected),
                        np.zeros(4, dtype=np.int64),
                    )
                )

            if domain == "positiveOnly_AVRDB_SEPARATE":
                assert all(
                    (c := dense.checked_counts(lookup[p]["channels"]["vessel"]))[3] == c[0] + c[2]
                    and c[1] == 0
                    for p in ids
                )
            if domain == "denseOnly_PRIMARY":
                assert all(
                    (c := dense.checked_counts(lookup[p]["channels"]["vessel"]))[3] > c[0] + c[2]
                    for p in ids
                )
            d["models"][name] = {
                "pooled": pooled(ids),
                "by_source": {
                    source: pooled(
                        [p for p in ids if lookup[p]["channel_datasets"]["vessel"] == source]
                    )
                    for source in d["support"]["sources"]
                },
            }
        if domain != "positiveOnly_AVRDB_SEPARATE":
            for ref in ["TPE5_LAST60", "original182"]:
                d["contrasts"]["loss2_LAST60__minus_" + ref] = dense.bootstrap(
                    ids, lookups[ref], lookups["loss2_LAST60"], metadata
                )
        domains[domain] = d
    assert not args.output.exists(), (
        "Fresh exclusive analysis output required; never overwrite old comparisons"
    )
    args.output.mkdir(parents=True)

    def put(name, value):
        with (args.output / name).open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")

    adverse = {}
    with tempfile.TemporaryDirectory(prefix="loss2-derived-count-views-") as temp:
        views = {}
        for name, d in records.items():
            m = d.get("metrics", d)
            view = {
                **m,
                "manifest_sha256": dense.MANIFEST,
                "prepared_content_sha256": dense.CANONICAL,
                "checkpoint_sha256": d.get("checkpoint", {}).get(
                    "sha256", m.get("checkpoint_sha256")
                ),
                "derived_count_view_provenance": {
                    "original_whole_bytes_sha256": originals[name][1],
                    "not_a_replacement_evaluation_or_eligibility_report": True,
                },
            }
            path = Path(temp) / (name + ".json")
            path.write_text(json.dumps(view, indent=2, allow_nan=False) + "\n")
            views[name] = path
            input_pins[name]["derived_view_sha256"] = sha(path)
        for ref in ["TPE5_LAST60", "original182"]:
            result = cmp.compare_reports(
                views[ref],
                views["loss2_LAST60"],
                args.manifest,
                repeats=2000,
                seed=42,
                include_sources=True,
                include_av=True,
            )
            assert result["photos"] == 770 and result["split_groups"] == 612
            put("loss2_LAST60__minus_" + ref + ".json", result)
            negatives = []
            for source, heads in result["by_dataset"].items():
                for head, value in heads.items():
                    delta, ci = (
                        value["candidate_minus_reference"]["dice"],
                        value["paired_dice_percentile_95ci"],
                    )
                    if delta is not None and delta < 0:
                        negatives.append(
                            {
                                "source": source,
                                "head": head,
                                "delta": delta,
                                "paired95ci": ci,
                                "upper95_below_zero": ci is not None and ci[1] < 0,
                            }
                        )
            for source, value in result["av_accuracy"]["by_dataset"].items():
                value = value["conditional_accuracy"]
                delta, ci = value["candidate_minus_reference"], value["paired_percentile_95ci"]
                if delta is not None and delta < 0:
                    negatives.append(
                        {
                            "source": source,
                            "metric": "conditional_AV",
                            "delta": delta,
                            "paired95ci": ci,
                            "upper95_below_zero": ci is not None and ci[1] < 0,
                        }
                    )
            adverse[ref] = negatives
    put(
        "dense-vessel.json",
        {
            "primary": "dense87photos55knownGroups8sources loss2LAST60 minus matched firstTPE5LAST60",
            "excluded_only_source": ["avrdb"],
            "domains": domains,
            "draws": 2000,
            "seed": 42,
            "not_original_objective_redefinition": True,
        },
    )
    put("adverse-source-results.json", adverse)
    put(
        "scientific-analysis.json",
        {
            "point_metrics": points,
            "input_provenance": input_pins,
            "primary_dense_CI": domains["denseOnly_PRIMARY"]["contrasts"][
                "loss2_LAST60__minus_TPE5_LAST60"
            ],
            "all_head_groups": 612,
            "dense_groups": 55,
            "draws": 2000,
            "seed": 42,
            "comparison_math_sha256": MATH_SHA,
            "dense_math_sha256": DENSE_SHA,
            "analysis_script_sha256": sha(__file__),
            "original_terminal_sha256": args.terminal_sha256,
            "original_outer_sha256": args.outer_exit_sha256,
            "original_postcheck_sha256": args.postcheck_sha256,
            "original_runtime_sha256": args.runtime_sha256,
            "actual_LAST60_own_coverage": coverage,
            "scope": "Count-only LAST60 controlled contrast. No BEST substitution, clinical margin, source causal attribution, universal quality/promotion or TEST claim. Dense positive CI alone does not excuse adverse other heads.",
            "all_prior_comparisons_unchanged": True,
        },
    )
    for path, pin in originals.values():
        assert sha(path) == pin
    print(
        json.dumps(
            {
                "primary_dense": domains["denseOnly_PRIMARY"]["contrasts"][
                    "loss2_LAST60__minus_TPE5_LAST60"
                ],
                "point_metrics": points,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "fresh-report",
        "control-report",
        "terminal",
        "outer-exit",
        "postcheck",
        "runtime",
        "manifest",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in [
        "fresh-report-sha256",
        "terminal-sha256",
        "outer-exit-sha256",
        "postcheck-sha256",
        "runtime-sha256",
    ]:
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
