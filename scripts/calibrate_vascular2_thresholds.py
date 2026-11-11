"""One forward per VAL photo, eight unchanged scalar decoders; no TEST or probability cache."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from deepretina import unified as u
from deepretina.comparison import compare_reports

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CHECKPOINT = "81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b"
MANIFEST = "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
ORIGINAL = "182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def photo_counts(probability, target, valid, rows, trace, photo_id, supported, threshold):
    t, v = target.numpy() >= 0.5, valid.numpy().copy()
    v[~np.asarray(supported, bool)] = False
    masks = u.decode_probabilities(probability, threshold, supported)
    values = np.array(
        [
            [
                int((masks[c] & t[c] & v[c]).sum()),
                int((masks[c] & ~t[c] & v[c]).sum()),
                int((~masks[c] & t[c] & v[c]).sum()),
                int(v[c].sum()),
            ]
            for c in range(5)
        ],
        dtype=np.int64,
    )
    known = v[1] & v[2] & (t[1] ^ t[2])
    n = int(known.sum())
    row_map = {r["id"]: r for r in rows}
    sources = {
        u.CHANNELS[c]: row_map[rid]["dataset"]
        for rid, task in trace
        for c in u.SOURCE_CHANNELS[task]
    }
    return {
        "photo_id": photo_id,
        "record_ids": [r["id"] for r in rows],
        "dataset": rows[0]["dataset"],
        "channel_datasets": sources,
        "av_conditional_counts": {
            "correct": int((((probability[1] >= probability[2]) == t[1]) & known).sum()),
            "scored_pixels": n,
            "eligible_pixels": n,
            "end_to_end_correct": int(((masks[1] & t[1] | masks[2] & t[2]) & known).sum()),
        },
        "channels": {name: u._metrics(x) for name, x in zip(u.CHANNELS, values, strict=True)},
    }


def aggregate(photos, threshold, manifest_sha=MANIFEST):
    totals = np.zeros((5, 4), np.int64)
    sources = {}
    for photo in photos:
        for c, channel in enumerate(u.CHANNELS):
            x = photo["channels"][channel]
            values = np.array([x[k] for k in ("tp", "fp", "fn", "valid_pixels")])
            totals[c] += values
            source = photo["channel_datasets"].get(channel)
            if source is not None:
                sources.setdefault(source, np.zeros_like(totals))[c] += values
    channels = {name: u._metrics(x) for name, x in zip(u.CHANNELS, totals, strict=True)}
    by_source = {
        s: {name: u._metrics(x) for name, x in zip(u.CHANNELS, counts, strict=True)}
        for s, counts in sources.items()
    }
    macro = {
        c: np.mean([r[c]["dice"] for r in by_source.values() if r[c]["dice"] is not None]).item()
        for c in u.CHANNELS
        if any(r[c]["dice"] is not None for r in by_source.values())
    }
    av = {
        k: sum(p["av_conditional_counts"][k] for p in photos)
        for k in ("correct", "scored_pixels", "eligible_pixels", "end_to_end_correct")
    }
    return {
        "split": "val",
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": CHECKPOINT,
        "photos": len(photos),
        "threshold": threshold,
        "channels": channels,
        "by_dataset": by_source,
        "per_photo": photos,
        "macro_dataset_channel_dice": macro,
        "macro_dataset_mean_dice": float(np.mean(list(macro.values()))),
        "av_conditional_accuracy": av["correct"] / av["scored_pixels"]
        if av["scored_pixels"]
        else None,
        "av_accuracy": av["end_to_end_correct"] / av["eligible_pixels"]
        if av["eligible_pixels"]
        else None,
        "av_labeled_pixels": av["eligible_pixels"],
    }


def native_parity(model, dataset, index, probability, config, device):
    """Compare one real/native evaluator execution with this counter, sharing SAME forward."""
    rows = dataset.groups[index]
    image, target, valid, trace = dataset.sample(index, random_annotator=False, return_trace=True)
    current = aggregate(
        [
            photo_counts(
                probability,
                target,
                valid,
                rows,
                trace,
                dataset.photo_ids[index],
                model.supervised_channels,
                0.5,
            )
        ],
        0.5,
    )

    class OnePhoto:
        def __init__(self):
            self.groups = [rows]
            self.photo_ids = [dataset.photo_ids[index]]

        def sample(self, _, **kwargs):
            return image, target, valid, trace

        def __len__(self):
            return 1

    one = OnePhoto()
    one.rows = rows
    with (
        patch.object(u, "UnifiedDataset", return_value=one),
        patch.object(u, "predict_probabilities", return_value=probability),
    ):
        native = u.evaluate_unified_model(
            model, dataset.manifest, dataset.split, device, config, 0.5
        )
    for key in (
        "channels",
        "by_dataset",
        "per_photo",
        "macro_dataset_mean_dice",
        "av_conditional_accuracy",
        "av_accuracy",
    ):
        if native[key] != current[key]:
            raise ValueError("Native same-forward count parity failed: " + key)
    return {
        "status": "PASS",
        "photo_id": dataset.photo_ids[index],
        "same_forward_native_counts_exact": True,
        "split": dataset.split,
        "threshold": 0.5,
    }


def scoped_fingerprint(dataset):
    paths = sorted({row[key] for row in dataset.rows for key in ("image", "mask")})
    result = hashlib.sha256()
    seen = {}
    for relative in paths:
        path = dataset.root / relative
        st = path.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in seen:
            with path.open("rb") as stream:
                seen[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        result.update(json.dumps([relative, seen[key]]).encode())
    return result.hexdigest()


def baseline_drift(current, previous):
    previous = previous.get("metrics", previous)
    lookup = {p["photo_id"]: p for p in previous["per_photo"]}
    assert set(lookup) == {p["photo_id"] for p in current["per_photo"]}
    changes = []
    for p in current["per_photo"]:
        old = lookup[p["photo_id"]]
        assert (
            p["record_ids"] == old["record_ids"]
            and p["channel_datasets"] == old["channel_datasets"]
        )
        delta = {}
        for c in u.CHANNELS:
            a, b = old["channels"][c], p["channels"][c]
            assert a["valid_pixels"] == b["valid_pixels"] and a["tp"] + a["fn"] == b["tp"] + b["fn"]
            d = {k: b[k] - a[k] for k in ("tp", "fp", "fn")}
            if any(d.values()):
                delta[c] = d
        if p["av_conditional_counts"] != old["av_conditional_counts"]:
            delta["av_conditional_counts"] = {
                k: p["av_conditional_counts"][k] - old["av_conditional_counts"][k]
                for k in p["av_conditional_counts"]
            }
        if delta:
            changes.append({"photo_id": p["photo_id"], "count_deltas": delta})
    return {
        "exact_original_producer_counts": not changes,
        "changed_photos": len(changes),
        "changes": changes,
        "gt_support_and_source_ids_exact": True,
        "qualification": "Existing producer used another GPU/runtime; numeric drift is separated from same-forward threshold effects, not hidden by tolerance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "original", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--existing-half", type=Path)
    parser.add_argument("--train-canary-only", action="store_true")
    args = parser.parse_args()
    assert (
        sha(args.checkpoint) == CHECKPOINT
        and sha(args.manifest) == MANIFEST
        and sha(args.original) == ORIGINAL
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model, payload, device = u.load_unified(args.checkpoint, args.device)
    config = u.inference_config(payload)
    started = time.monotonic()
    dataset = u.UnifiedDataset(args.manifest, "train" if args.train_canary_only else "val", config)
    input_fp = None if args.train_canary_only else scoped_fingerprint(dataset)
    if args.train_canary_only:
        index = dataset.photo_ids.index(
            "sha256:d0d30763e7b390dd94bc084e2e8dbeedce22cd0cac1688cade051eb3324b5db2"
        )
        image, _, _, _ = dataset.sample(index, random_annotator=False, return_trace=True)
        probability = u.predict_probabilities(
            model, image, device, config, config.tile_size, config.overlap
        )
        save(
            args.output / "native-train-parity.json",
            native_parity(model, dataset, index, probability, config, device),
        )
        return
    assert len(dataset) == len(set(dataset.photo_ids)) == 770
    gathered = {t: [] for t in THRESHOLDS}
    support = model.supervised_channels
    for index, rows in enumerate(dataset.groups):
        image, target, valid, trace = dataset.sample(
            index, random_annotator=False, return_trace=True
        )
        probability = u.predict_probabilities(
            model, image, device, config, config.tile_size, config.overlap
        )
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite prediction")
        if index == 0:
            save(
                args.output / "native-val-parity.json",
                native_parity(model, dataset, index, probability, config, device),
            )
        counts = [
            photo_counts(
                probability, target, valid, rows, trace, dataset.photo_ids[index], support, t
            )
            for t in THRESHOLDS
        ]
        assert (
            len(
                {
                    (
                        p["av_conditional_counts"]["correct"],
                        p["av_conditional_counts"]["scored_pixels"],
                    )
                    for p in counts
                }
            )
            == 1
        )
        with (args.output / "count-progress.jsonl").open("a") as journal:
            journal.write(
                json.dumps(
                    {
                        "index": index,
                        "photo_id": dataset.photo_ids[index],
                        "counts": dict(zip(map(str, THRESHOLDS), counts, strict=True)),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        for t, value in zip(THRESHOLDS, counts, strict=True):
            gathered[t].append(value)
        if (index + 1) % 25 == 0:
            print(
                json.dumps({"photos": index + 1, "seconds": time.monotonic() - started}), flush=True
            )
    original = json.loads(args.original.read_text())
    original = original.get("metrics", original)
    reference = {c: original["channels"][c]["dice"] for c in u.CHANNELS}
    assert scoped_fingerprint(dataset) == input_fp
    scores = []
    for threshold, photos in gathered.items():
        report = aggregate(photos, threshold)
        ratios = {c: report["channels"][c]["dice"] / reference[c] for c in u.CHANNELS}
        ratios["conditional_av"] = (
            report["av_conditional_accuracy"] / original["av_conditional_accuracy"]
        )
        report["worst6ratio"] = min(ratios.values())
        report["ratios"] = ratios
        save(args.output / f"threshold-{threshold:.2f}.json", report)
        scores.append(
            {
                "threshold": threshold,
                "worst6ratio": report["worst6ratio"],
                "macro": report["macro_dataset_mean_dice"],
                "pooled": {c: report["channels"][c]["dice"] for c in u.CHANNELS},
                "conditional_av": report["av_conditional_accuracy"],
            }
        )
    if args.existing_half is not None:
        assert (
            sha(args.existing_half)
            == "ddc9677f100a61d2d0bc8d7fa0ee3f9212e5142c1329d7f96e9ce6698f6a3fc1"
        )
        current = json.loads((args.output / "threshold-0.50.json").read_text())
        save(
            args.output / "baseline-0.50-existing-producer-drift.json",
            baseline_drift(current, json.loads(args.existing_half.read_text())),
        )
    best = max(scores, key=lambda x: (x["worst6ratio"], -abs(x["threshold"] - 0.5)))
    paired = compare_reports(
        args.output / "threshold-0.50.json",
        args.output / f"threshold-{best['threshold']:.2f}.json",
        args.manifest,
        repeats=2000,
        seed=42,
        include_sources=True,
        include_av=True,
    )
    save(args.output / "best-vs-0.50-paired.json", paired)
    save(
        args.output / "summary.json",
        {
            "scores": scores,
            "selected_val_threshold": best["threshold"],
            "selection_rule": "max worst6ratio vs frozen original, exact ties closest0.5",
            "default_threshold_changed": False,
            "test_evaluation_performed": False,
            "one_forward_per_photo": True,
            "probabilities_persisted": False,
            "photos": 770,
            "seconds": time.monotonic() - started,
            "val_encoded_input_sha256": input_fp,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "actual_cuda_uuid": str(torch.cuda.get_device_properties(0).uuid)
            if device.type == "cuda"
            else None,
            "checkpoint_sha256": CHECKPOINT,
            "manifest_sha256": MANIFEST,
            "original_sha256": ORIGINAL,
            "script_sha256": sha(__file__),
            "torch_threads": 2,
            "interop_threads": 1,
            "device": str(device),
            "qualifications": [
                "VAL threshold exploration with8candidates; pairedCI conditional on chosen threshold and checkpoint, multiplicity/selection bias not removed",
                "Known612split groups are not proven independent biological patients",
                "One global scalar changes vessel/AVgating and disc/cup together; inspect allhead/source tradeoffs",
            ],
        },
    )


if __name__ == "__main__":
    main()
