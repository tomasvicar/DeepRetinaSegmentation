"""Saved VAL JSON confusion counts only; no Torch, GT, image or model access."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = {
    "original182": "reports/frozen-original-val-nearest/validation.json",
    "original_final6808": "reports/original-f13-Old-VAL-20261112/validation.json",
    "TPE5_BEST58_rot4_035": "reports/first-TPE-BEST58-rot4-threshold-20261111/threshold-0.35.json",
    "TPE5_BEST58_rot4_050": "reports/first-TPE-BEST58-rot4-threshold-20261111/threshold-0.50.json",
    "raw512_BEST50_none_050": "reports/raw512-typed-fresh-val-20261112/raw512-BEST50-aggregate-counts.json",
    "raw512_BEST50_rot4_035": "reports/raw512-BEST50-rot4-035-fresh-val-20261112/raw512-BEST50-rot4-035-aggregate-counts.json",
}


def counts(value):
    result = {k: value[k] for k in ("tp", "fp", "fn", "valid_pixels")}
    assert all(type(v) is int and v >= 0 for v in result.values())
    assert result["tp"] + result["fp"] + result["fn"] <= result["valid_pixels"]
    return result


def scores(value):
    t, p, n = value["tp"], value["fp"], value["fn"]
    return {
        **value,
        "positive_pixels": t + n,
        "precision": t / (t + p),
        "recall": t / (t + n),
        "dice": 2 * t / (2 * t + p + n),
    }


def main():
    raw = {k: (ROOT / p).read_bytes() for k, p in FILES.items()}
    docs = {k: json.loads(v) for k, v in raw.items()}
    metrics = {k: d.get("metrics", d) for k, d in docs.items()}
    historical = json.loads(
        (ROOT / "reports/dense-vessel-support-20261112/analysis.json").read_bytes()
    )
    support = historical["domains"]["denseOnly"]["support"]
    assert support["photos"] == 87 and support["known_split_groups"] == 55
    sources = support["sources"]
    assert len(sources) == 8 and "avrdb" not in sources
    ids = {
        p["photo_id"]
        for p in metrics["original182"]["per_photo"]
        if p["channels"]["vessel"]["valid_pixels"] and p["channel_datasets"]["vessel"] != "avrdb"
    }
    assert len(ids) == 87
    original = {p["photo_id"]: p for p in metrics["original182"]["per_photo"]}
    for name, d in metrics.items():
        if "per_photo" not in d:
            continue  # raw512 public projection lacks photo IDs
        lookup = {p["photo_id"]: p for p in d["per_photo"]}
        assert len(lookup) == len(d["per_photo"]) == 770 and set(lookup) == set(original)
        for pid, left in original.items():
            right = lookup[pid]
            assert (
                left["record_ids"] == right["record_ids"]
                and left["channel_datasets"] == right["channel_datasets"]
            )
            a, b = counts(left["channels"]["vessel"]), counts(right["channels"]["vessel"])
            assert a["valid_pixels"] == b["valid_pixels"] and a["tp"] + a["fn"] == b["tp"] + b["fn"]
    models = {}
    for name, d in metrics.items():
        by_source = {s: scores(counts(d["by_dataset"][s]["vessel"])) for s in sources}
        total = {
            k: sum(v[k] for v in by_source.values()) for k in ("tp", "fp", "fn", "valid_pixels")
        }
        assert total["valid_pixels"] == 60325745 and total["tp"] + total["fn"] == 5893007
        for s in sources:
            a, b = by_source[s], metrics["original182"]["by_dataset"][s]["vessel"]
            assert (
                a["valid_pixels"] == b["valid_pixels"] and a["positive_pixels"] == b["tp"] + b["fn"]
            )
        models[name] = {"pooled": scores(total), "by_source": by_source}
    # Reproduce previously published supplementary counts, not a new domain.
    previous = historical["domains"]["denseOnly"]["models"]["TPE5_BEST58_rot4_035"]["pooled"]
    assert all(
        models["TPE5_BEST58_rot4_035"]["pooled"][k] == previous[k]
        for k in ("tp", "fp", "fn", "valid_pixels", "dice", "precision", "recall")
    )

    contrasts = {}
    for name in ("TPE5_BEST58_rot4_035", "raw512_BEST50_rot4_035"):
        candidate, reference = models[name]["pooled"], models["original182"]["pooled"]
        contrasts[name] = {
            k: candidate[k] - reference[k]
            for k in ("tp", "fp", "fn", "precision", "recall", "dice")
        }
        contrasts[name]["extra_FP_plus_FN"] = (
            candidate["fp"] + candidate["fn"] - reference["fp"] - reference["fn"]
        )
        contrasts[name]["by_source"] = {
            s: {
                k: models[name]["by_source"][s][k] - models["original182"]["by_source"][s][k]
                for k in ("tp", "fp", "fn", "precision", "recall", "dice")
            }
            for s in sources
        }
    t35, t50 = (
        docs[k]["actual_inference_protocol"].copy()
        for k in ("TPE5_BEST58_rot4_035", "TPE5_BEST58_rot4_050")
    )
    assert t35.pop("threshold") == 0.35 and t50.pop("threshold") == 0.5 and t35 == t50
    assert (
        docs["TPE5_BEST58_rot4_035"]["checkpoint_sha256"]
        == docs["TPE5_BEST58_rot4_050"]["checkpoint_sha256"]
    )
    curve = []
    for threshold in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        path = (
            ROOT
            / "reports/first-TPE-BEST58-rot4-threshold-20261111"
            / f"threshold-{threshold:.2f}.json"
        )
        raw_curve = path.read_bytes()
        d = json.loads(raw_curve)
        assert d["checkpoint_sha256"] == docs["TPE5_BEST58_rot4_035"]["checkpoint_sha256"]
        total = {
            k: sum(d["by_dataset"][s]["vessel"][k] for s in sources)
            for k in ("tp", "fp", "fn", "valid_pixels")
        }
        assert total["tp"] + total["fn"] == 5893007 and total["valid_pixels"] == 60325745
        curve.append(
            {
                "threshold": threshold,
                "dense_vessel": scores(total),
                "report_sha256": hashlib.sha256(raw_curve).hexdigest(),
            }
        )
    summary = json.loads(
        (ROOT / "reports/first-TPE-BEST58-rot4-threshold-20261111/summary.json").read_bytes()
    )
    assert (
        summary["same_forward_native_half_counter_parity_all770"] is True
        and summary["inference_calls_per_photo"] == 1
    )
    result = {
        "kind": "dense_vessel_saved_VAL_precision_recall_triage_NOT_selection",
        "support": support,
        "excluded": ["avrdb"],
        "input_pins": {
            k: {"path": FILES[k], "sha256": hashlib.sha256(b).hexdigest()} for k, b in raw.items()
        },
        "models": models,
        "candidate_minus_original182": contrasts,
        "observed_TPE5_same_forward_dense_curve": curve,
        "observed_dense_grid_peak_NOT_new_selection": max(
            curve, key=lambda x: x["dense_vessel"]["dice"]
        )["threshold"],
        "threshold_isolation": {
            "TPE5_rot4_035_vs_rot4_050": "same checkpoint/protocol except threshold, same saved forward for each photo; threshold-only decoder contrast",
            "raw512_rot4_035_vs_NONE_050": "BOTH threshold and TTA changed; not isolated; published full per-photo typed report remains private",
        },
        "qualifications": [
            "Known55 split groups inherited from pinned prior support audit, not newly inferred patient independence",
            "raw512 projection has source aggregates only; matching source GT counts verified here, full87photo identity/support proven in prior completed original report/analysis, not independently reread here",
            "Final6808 original is reported separately; historical182 never retagged",
            "No probability values or GT/model/image arrays loaded; no NN/training/host/TEST/DB operation",
        ],
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["pooled"] for k, v in models.items()}, indent=2))


if __name__ == "__main__":
    main()
