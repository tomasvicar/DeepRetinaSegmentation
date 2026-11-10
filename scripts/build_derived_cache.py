"""Build lossless, externally pinned derivatives; canonical data remain unchanged."""

import argparse
import json
from pathlib import Path

from deepretina.derived_cache import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-base", type=Path, required=True)
    parser.add_argument(
        "--max-photos", type=int, help="Explicit stratified sample; omitted builds selected splits"
    )
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_cache(
                args.manifest,
                args.cache_base,
                splits=args.splits,
                max_photos=args.max_photos,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
