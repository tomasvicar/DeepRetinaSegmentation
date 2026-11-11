"""CPU safe snapshots only: parameter versus BN-buffer E52/E60 drift."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch


def snapshot(path):
    raw = Path(path).read_bytes()
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True), hashlib.sha256(
        raw
    ).hexdigest()


def compare(best, last):
    a, ah = snapshot(best)
    b, bh = snapshot(last)
    if a["epoch"] != 52 or b["epoch"] != 60 or a["config"] != b["config"]:
        raise ValueError("Require matched actual52/60")
    groups = {}
    details = []
    for k, x in a["state_dict"].items():
        y = b["state_dict"][k]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("State schema differs")
        group = (
            "bn_running_buffers"
            if k.endswith(("running_mean", "running_var"))
            else "bn_counters"
            if k.endswith("num_batches_tracked")
            else "learnable_parameters"
        )
        dx = y.double() - x.double()
        den = float(x.double().square().sum())
        num = float(dx.square().sum())
        row = {
            "key": k,
            "group": group,
            "finite": bool(torch.isfinite(y).all() and torch.isfinite(x).all()),
            "changed": int(torch.count_nonzero(x != y)),
            "max_abs": float(dx.abs().max()),
            "relative_l2": (num / den) ** 0.5 if den else None,
        }
        details.append(row)
        g = groups.setdefault(
            group,
            {
                "elements": 0,
                "changed": 0,
                "delta_squared": 0.0,
                "reference_squared": 0.0,
                "max_abs": 0.0,
            },
        )
        g["elements"] += x.numel()
        g["changed"] += row["changed"]
        g["delta_squared"] += num
        g["reference_squared"] += den
        g["max_abs"] = max(g["max_abs"], row["max_abs"])
    for g in groups.values():
        g["relative_l2"] = (
            (g["delta_squared"] / g["reference_squared"]) ** 0.5 if g["reference_squared"] else None
        )
    return {
        "scope": "Observed state drift; does not establish BN causality or quality gain",
        "best_epoch": 52,
        "last_epoch": 60,
        "best_sha256": ah,
        "last_sha256": bh,
        "groups": groups,
        "per_tensor": details,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best", required=True)
    p.add_argument("--last", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    r = compare(a.best, a.last)
    with a.output.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
    print(json.dumps(r["groups"]))


if __name__ == "__main__":
    main()
