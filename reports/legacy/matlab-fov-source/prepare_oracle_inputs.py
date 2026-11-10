"""Prepare external scratch inputs from owned archives and the original helper."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

HELPER_SHA256 = "6f47763d8c1b1821d0f6613bc5c823b910577d668857bf617f54d01dcb1fcf9b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    if root.is_relative_to(args.data_root.resolve()):
        raise ValueError("Oracle scratch must be outside the owned source database directory")
    helper = args.helper.read_bytes()
    if hashlib.sha256(helper).hexdigest() != HELPER_SHA256:
        raise ValueError("Provide unchanged user-owned souradnice.m from7bb0539b")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose new empty oracle scratch; existing inputs are preserved")
    root.mkdir(parents=True, exist_ok=True)
    (root / "souradnice.m").write_bytes(helper)
    for name, archive, output, metadata in [
        ("erlangen", "Erlangen.zip", "real_erlangen.png", "real_source.json"),
        ("rimone", "RIM-ONE.zip", "real_rimone.jpg", "real_rimone_source.json"),
    ]:
        with zipfile.ZipFile(args.data_root / archive) as source:
            if name == "erlangen":
                files = sorted(
                    n
                    for n in source.namelist()
                    if "/FundusImages/" in n
                    and n.lower().endswith(".png")
                    and "fov" not in Path(n).name.lower()
                )
            else:
                files = sorted(
                    n
                    for n in source.namelist()
                    if "stereo images" in n.lower() and n.lower().endswith(".jpg")
                )
            if not files:
                raise FileNotFoundError(f"No expected raw source image in{archive}")
            resource = files[0]
            data = source.read(resource)
        with Image.open(io.BytesIO(data)) as image:
            shape = [image.height, image.width]
        (root / output).write_bytes(data)
        (root / metadata).write_text(
            json.dumps(
                {
                    "archive": archive,
                    "resource": resource,
                    "file_sha256": hashlib.sha256(data).hexdigest(),
                    "shape": shape,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps({"scratch": str(root), "helper_sha256": HELPER_SHA256}))


if __name__ == "__main__":
    main()
