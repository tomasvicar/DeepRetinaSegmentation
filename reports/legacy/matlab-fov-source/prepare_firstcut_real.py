import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from deepretina.sources import Source
from PIL import Image

parser = argparse.ArgumentParser(
    description="Extract only three raw calibration frames, never annotations"
)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.data_root
out = args.output
if out.resolve().is_relative_to(root.resolve()):
    raise ValueError("Scratch must be outside source database")
out.mkdir(parents=True, exist_ok=True)
specs = [
    ("stare", "STARE.zip", "all-images/im0002.ppm", "train canonical metadata"),
    ("drishtigs", "Drishti-GS.zip", "Training/Images/drishtiGS_002.png", "official train"),
    (
        "diaretdb",
        "DiaRetDB0.zip",
        "resources/images/diaretdb0_fundus_images/image001.png",
        "image-only dataset, no GT/split quality evaluation",
    ),
]
meta = []
for name, archive, suffix, partition in specs:
    with Source(root / archive) as source:
        path = source.find(suffix)
        raw = source.bytes(path)
        rgb = source.read(path, mode="RGB")
        target = out / f"firstcut_real_{name}.png"
        if target.exists():
            raise FileExistsError(target)
        Image.fromarray(rgb).save(target)
        meta.append(
            {
                "dataset": name,
                "archive": archive,
                "resource": path,
                "source_file_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_rgb_sha256": hashlib.sha256(
                    np.asarray(rgb, np.uint8).tobytes()
                ).hexdigest(),
                "shape": list(rgb.shape),
                "partition_scope": partition,
                "scratch_file": target.name,
            }
        )
(out / "firstcut_real_sources.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
