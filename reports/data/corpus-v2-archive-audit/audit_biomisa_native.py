import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from deepretina.adapters.additional import resource, resources
from deepretina.sources import Source

root = Path("/var/home/tom/ownCloud/fundus_databases")
rows = []
with Source(root / "BIOMISA.zip") as source:
    index = resources(source, "BIOMISA")
    keys = [
        k
        for k in index
        if re.fullmatch(
            r"OD/(Glaucoma|Healthy)/P_\d+/(Left|Right) Eye/Fundus_.+_(Disc|Cup)\.jpg", k
        )
    ]
    folders = sorted({str(Path(k).parent) for k in keys})
    for folder in folders:
        labels = {}
        for structure in ["Disc", "Cup"]:
            ks = [
                k
                for k in keys
                if str(Path(k).parent) == folder and k.endswith("_" + structure + ".jpg")
            ]
            if len(ks) != 1:
                continue
            name = resource(source, index[ks[0]])
            data = source.bytes(name)
            im = Image.open(io.BytesIO(data))
            a = np.asarray(im.convert("RGBA"))
            assert np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 1], a[:, :, 2])
            assert set(np.unique(a[:, :, :3])) <= {0, 255}
            mask = (a[:, :, 0] == 255) & (a[:, :, 3] > 0)
            labels[structure] = {
                "resource": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "actual_format": im.format,
                "mode": im.mode,
                "shape": list(mask.shape),
                "positive_pixels": int(mask.sum()),
                "alpha_values": np.unique(a[:, :, 3]).tolist(),
                "unfilled_hole_pixels": int((binary_fill_holes(mask) & ~mask).sum()),
                "_mask": mask,
            }
        images = [
            k
            for k in index
            if k.startswith(folder + "/") and "_Color_" in k and k.lower().endswith(".jpg")
        ]
        raw = []
        for k in images:
            name = resource(source, index[k])
            data = source.bytes(name)
            im = Image.open(io.BytesIO(data))
            raw.append(
                {
                    "resource": name,
                    "shape": [im.height, im.width],
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        nesting = {}
        if (
            "Disc" in labels
            and "Cup" in labels
            and labels["Disc"]["shape"] == labels["Cup"]["shape"]
        ):
            d = labels["Disc"]["_mask"]
            c = labels["Cup"]["_mask"]
            nesting = {
                "cup_outside_named_disc_pixels": int((c & ~d).sum()),
                "named_disc_outside_named_cup_pixels": int((d & ~c).sum()),
                "named_cup_fraction_of_disc": int(c.sum()) / max(int(d.sum()), 1),
            }
        for info in labels.values():
            info.pop("_mask")
        rows.append(
            {
                "folder": folder,
                "labels": labels,
                "raw_color_candidates": raw,
                "nesting": nesting,
                "status": "quarantined_pending_semantic_and_matching_review",
                "patient_group": ":".join(folder.split("/")[:3]),
                "data_split_policy": "candidate_train_only",
                "physical_angle_verified": False,
            }
        )
report = {
    "scope": "All native BIOMISA OD fundus label resources; no OCT GT, no clinical metadata values; no training use or label swaps",
    "folders": rows,
    "counts": {
        "folders": len(rows),
        "disc_labels": sum("Disc" in r["labels"] for r in rows),
        "cup_labels": sum("Cup" in r["labels"] for r in rows),
        "both_labels": sum(len(r["labels"]) == 2 for r in rows),
        "contradictory_cup_not_inside_disc": sum(
            r["nesting"].get("cup_outside_named_disc_pixels", 0) > 0 for r in rows
        ),
        "reverse_nesting": sum(
            r["nesting"].get("cup_outside_named_disc_pixels", 0) > 0
            and r["nesting"].get("named_disc_outside_named_cup_pixels", 1) == 0
            for r in rows
        ),
        "missing_or_ambiguous_raw": sum(len(r["raw_color_candidates"]) != 1 for r in rows),
    },
    "all_resources_preserved": True,
    "swapped_labels_automatically": False,
}
Path("/tmp/biomisa-native-label-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["counts"]), flush=True)
