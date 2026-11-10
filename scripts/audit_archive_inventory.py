"""Read archive central directories; never decode images or annotation contents."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from deepretina.prepare import ARCHIVES
from deepretina.sources import canonical_path


def inventory(root):
    registered = {v.casefold(): k for k, v in ARCHIVES.items()}
    rows = []
    for path in sorted(Path(root).iterdir()):
        if not path.is_file() or (path.suffix.lower() != ".zip" and path.name != "ziajP7YB"):
            continue
        row = {
            "archive": path.name,
            "bytes": path.stat().st_size,
            "registered_selector": registered.get(path.name.casefold()),
        }
        if not zipfile.is_zipfile(path):
            row.update(status="unreadable_central_directory", files=None)
            rows.append(row)
            continue
        with zipfile.ZipFile(path) as archive:
            members = [
                m for m in archive.infolist() if not m.is_dir() and "__MACOSX" not in m.filename
            ]
            logical = {}
            conflicts = []
            for m in members:
                name = canonical_path(m.filename)
                parts = name.split("/")
                if parts[0].casefold() == path.stem.casefold():
                    name = "/".join(parts[1:])
                signature = (m.file_size, m.CRC)
                if name in logical and logical[name] != signature:
                    conflicts.append(name)
                logical[name] = signature
            row.update(
                status="indexed",
                central_files=len(members),
                logical_files=len(logical),
                duplicate_transport_members=len(members) - len(logical),
                conflicting_logical_members=conflicts,
                extensions=dict(Counter(Path(n).suffix.lower() for n in logical).most_common()),
                annotation_name_candidates=sum(
                    any(
                        t in n.lower()
                        for t in (
                            "mask",
                            "annot",
                            "contour",
                            "expert",
                            "groundtruth",
                            "ground truth",
                        )
                    )
                    for n in logical
                ),
            )
        rows.append(row)
    return {
        "schema_version": 1,
        "scope": "Central-directory names/CRC/size only; counts include auxiliary modalities and are not training eligibility",
        "archives": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = inventory(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
