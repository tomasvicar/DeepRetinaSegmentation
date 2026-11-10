"""Portable, checksum-verified pilot checkpoints distributed with the package."""

import hashlib
import json
from contextlib import contextmanager
from importlib import resources


def model_catalog() -> dict:
    return json.loads(
        resources.files("deepretina").joinpath("pretrained/registry.json").read_text()
    )


@contextmanager
def bundled_checkpoint(name: str):
    catalog = model_catalog()["models"]
    if name not in catalog:
        raise ValueError(f"Unknown bundled model {name!r}; choose {', '.join(catalog)}")
    model = catalog[name]
    resource = resources.files("deepretina").joinpath("pretrained", model["filename"])
    with resources.as_file(resource) as path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != model["sha256"]:
            raise ValueError(f"Bundled model {name!r} checksum mismatch; restore the package")
        yield path
