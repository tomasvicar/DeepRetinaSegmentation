"""Exact stdlib resolver functions copied from frozen f9d5e66; no semantic rewrite."""
import unicodedata


def canonical_path(name: str) -> str:
    name = unicodedata.normalize("NFC", name.replace("\\", "/")).lstrip("/")
    marker = "/data/databases/"
    pos = name.lower().find(marker)
    if pos >= 0:
        name = name[pos + len(marker) :]
    return name


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_requests(names, requests, identities=None):
    """Resolve metadata only. Ambiguous transport aliases require reviewed exact closure."""
    resolved = {}
    for requested in requests:
        suffix = canonical_path(requested).lower()
        exact = [n for n in names if n.lower() == suffix]
        found = exact or [n for n in names if n.lower().endswith("/" + suffix)]
        if len(found) > 1 and identities is not None:
            aliases = {identities[n] for n in found}
            if len(aliases) == 1 and None not in aliases:
                found = [min(found, key=lambda n: (len(n), n))]
        require(len(found) == 1, "Missing/ambiguous closure resource: " + requested)
        resolved[requested] = found[0]
    return resolved
