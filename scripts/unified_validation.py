"""Produce fresh enriched VAL evidence; this command never evaluates test data."""

import argparse

import torch

from deepretina.selection import SelectionError
from deepretina.validation_evidence import produce_validation_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--tta", choices=["none", "flips2", "rot4", "dihedral8"], default="none")
    parser.add_argument("--photometry-cache")
    parser.add_argument("--expected-photometry-cache-fingerprint")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        produce_validation_evidence(
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            output=args.output,
            device=args.device,
            threshold=args.threshold,
            tile_size=args.tile_size,
            overlap=args.overlap,
            **({"tta": args.tta} if args.tta != "none" else {}),
            **(
                {
                    "photometry_cache_root": args.photometry_cache,
                    "expected_photometry_cache_fingerprint": args.expected_photometry_cache_fingerprint,
                }
                if args.photometry_cache is not None
                or args.expected_photometry_cache_fingerprint is not None
                else {}
            ),
        )
    except (SelectionError, ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"unified-validation: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
