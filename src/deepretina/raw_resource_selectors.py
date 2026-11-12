"""Metadata-only selectors matching historical decoder lookups.

Ordinary suffix requests retain their caller's frozen resolver. Typed REFUGE
requests reproduce decode_core's folder/basename predicate and sorted-name order;
multiple matches intentionally use the historical shortest-name selection.
"""

from pathlib import Path

REQUEST_SCHEMA = "frozen_resource_requests_with_refuge_selectors_v1"
FOLDERS = {
    "train": "Annotation-Training400/Disc_Cup_Masks",
    "val": "REFUGE-Validation400-GT/Disc_Cup_Masks",
    "test": "REFUGE-Test-GT/Disc_Cup_Masks",
}


def refuge_mask_selector(source_split, source_image):
    folder = FOLDERS[source_split]
    basename = Path(source_image).stem + ".bmp"
    return folder + "/" + basename, {
        "kind": "refuge_mask_folder_basename_v1",
        "folder": folder,
        "basename": basename,
    }


def resolve_resource_requests(
    names, requests, identities=None, selectors=None, *, existing_resolver
):
    """Resolve the complete whitelist without opening any member payload."""
    selectors = {} if selectors is None else selectors
    if not isinstance(selectors, dict) or not set(selectors) <= set(requests):
        raise ValueError("Selectors must name existing resource requests")
    resolved = existing_resolver(names, [r for r in requests if r not in selectors], identities)
    # Source.names is sorted; retain that order for min(key=len) ties.
    ordered = sorted(names)
    for requested, selector in selectors.items():
        if not isinstance(selector, dict) or set(selector) != {"kind", "folder", "basename"}:
            raise ValueError("Invalid REFUGE selector schema")
        folder, basename = selector["folder"], selector["basename"]
        if (
            selector["kind"] != "refuge_mask_folder_basename_v1"
            or folder not in FOLDERS.values()
            or not isinstance(basename, str)
            or not basename.lower().endswith(".bmp")
            or any(c in basename for c in ("/", "\\", "\x00"))
            or basename in {".bmp", "..bmp"}
            or requested != folder + "/" + basename
        ):
            raise ValueError("Invalid REFUGE folder/basename selector")
        matches = [
            n
            for n in ordered
            if f"/{folder.lower()}/" in n.lower() and Path(n).name.lower() == basename.lower()
        ]
        if not matches:
            raise ValueError("Missing REFUGE folder/basename resource: " + requested)
        resolved[requested] = min(matches, key=len)
    return resolved
