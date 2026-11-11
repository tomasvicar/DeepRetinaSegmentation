import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import pathlib
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

from deepretina import photometry as photo
from deepretina import photometry_cache as cache
from deepretina import unified as current

parser = argparse.ArgumentParser(
    description="Synthetic product/fd44 source and cache interoperability proof; no real corpus or operational model"
)
parser.add_argument(
    "--product-source", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2]
)
parser.add_argument("--authority-source", type=pathlib.Path, required=True)
parser.add_argument("--product-reference", type=pathlib.Path, required=True)
args = parser.parse_args()
product = args.product_source.resolve()
source = args.authority_source.resolve()
base = args.product_reference.resolve()
for root, pin in [
    (source, "fd44cdfa83d1df9ccc42960df6d1d2018db29900"),
    (base, "d028314d6a26c6790d82bdc97219027184c051df"),
]:
    assert (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        == pin
    )
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()


def package(name, root):
    path = root / "src/deepretina"
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return importlib.import_module(name + ".unified"), importlib.import_module(
        name + ".photometry"
    ) if (path / "photometry.py").exists() else None


old, _ = package("deepretina_product_d028_reference", base)
authority, authphoto = package("deepretina_fd44_reference", source)
authcache = importlib.import_module("deepretina_fd44_reference.photometry_cache")


def astfunc(path, name):
    tree = ast.parse(path.read_text())
    n = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.dump(n, include_attributes=False)


mathnames = [
    "build_unified",
    "_joint_probabilities",
    "_conditional_probabilities",
    "_tta_transform",
    "_tta_inverse",
    "tta_protocol",
    "predict_probabilities",
    "_predict_probabilities",
    "decode_probabilities",
    "restore_probabilities",
]
assert all(
    astfunc(product / "src/deepretina/unified.py", n)
    == astfunc(base / "src/deepretina/unified.py", n)
    for n in mathnames
)
assert astfunc(product / "src/deepretina/unified.py", "train_unified") == astfunc(
    source / "src/deepretina/unified.py", "train_unified"
)
assert cache.operator_fingerprint() == authcache.operator_fingerprint()
proof = {
    "product_base_commit": "d028314d6a26c6790d82bdc97219027184c051df",
    "authority_commit": "fd44cdfa83d1df9ccc42960df6d1d2018db29900",
    "unchanged_product_math_function_AST": mathnames,
    "training_function_AST_exactly_matches_authority": True,
    "operator_fingerprint": cache.operator_fingerprint(),
    "actual_frozen_source_imports_checked": True,
    "no_NN_forward_or_training_or_real_images_or_GT_or_GPU": True,
}
rng = np.random.default_rng(42)
raw = rng.integers(10, 225, (75, 97, 3), dtype=np.uint8)
fov = np.zeros(raw.shape[:2], bool)
fov[7:-6, 9:-8] = True
proof["profiles"] = {}
for name in photo.PROFILES:
    states = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    a = photo.apply(raw, name, fov=fov)
    b = authphoto.apply(raw, name, fov=fov)
    assert np.array_equal(a, b) and photo.contract(name) == authphoto.contract(name)
    assert (
        torch.equal(states[0], torch.get_rng_state())
        and states[1] == random.getstate()
        and all(np.array_equal(a, b) for a, b in zip(states[2], np.random.get_state()))
    )
    proof["profiles"][name] = {
        "array_SHA256": hashlib.sha256(a.tobytes()).hexdigest(),
        "descriptor_exact_equal": True,
        "array_exact_equal": True,
        "all_RNG_exact_unchanged": True,
        "outside_FOV_exact": np.array_equal(a[~fov], raw[~fov]),
    }


class Pointwise:
    def eval(self):
        return self

    def __call__(self, x):
        return torch.cat([x[:, :1], x[:, 1:2], -x[:, 1:2], x[:, 2:3], -x[:, 2:3]], 1)


image = torch.from_numpy(raw.transpose(2, 0, 1).copy()).float() / 255
with torch.inference_mode():
    for mode in ["none", "flips2", "rot4", "dihedral8"]:
        a = old.predict_probabilities(
            Pointwise(), image, "cpu", old.UnifiedConfig(), 64, 16, tta=mode
        )
        b = current.predict_probabilities(
            Pointwise(), image, "cpu", current.UnifiedConfig(), 64, 16, tta=mode
        )
        assert np.array_equal(a, b) and np.array_equal(
            old.decode_probabilities(a), current.decode_probabilities(b)
        )
proof["d028_mock_pointwise_probabilities_and_masks_bitwise_all4_TTA"] = True


# Existing bundles/aliases and legacy byte resources are checked through Git object identities only.
def tree(root, pin, path):
    return subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", pin, "--", path], text=True
    )


assert tree(product, "HEAD", "src/deepretina/pretrained") == tree(
    base, "HEAD", "src/deepretina/pretrained"
)
for p in ["src/deepretina/bundled.py", "src/deepretina/legacy", "configs"]:
    assert tree(product, "HEAD", p) == tree(base, "HEAD", p)
proof["all12_bundled_blob_objects_aliases_legacy_configs_unchanged"] = True
proof["cache_cross_source"] = {}
from test_derived_cache import audit
from test_derived_cache import corpus as make_corpus

with tempfile.TemporaryDirectory(prefix="product-photo-synthetic-") as t:
    root = pathlib.Path(t)
    manifest, rows = make_corpus(root / "canonical")
    photos = json.loads((manifest.parent / "photos.json").read_text())
    for p in photos:
        p.update(preprocessing_profile="legacy25_raw", normalized=False, pixels_per_degree=25.0)
    (manifest.parent / "photos.json").write_text(json.dumps(photos))
    audit(manifest.parent)
    content = json.loads((manifest.parent / "integrity.json").read_text())[
        "prepared_content_sha256"
    ]
    raw_samples = []
    for module in [old, current]:
        ds = module.UnifiedDataset(
            manifest,
            config=module.UnifiedConfig(
                augmentation="basic", foreground_probability=0.5, patch_size=32
            ),
        )
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        sample = ds[0]
        raw_samples.append(
            (sample, torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
        )
    assert all(torch.equal(a, b) for a, b in zip(raw_samples[0][0], raw_samples[1][0]))
    assert (
        torch.equal(raw_samples[0][1], raw_samples[1][1]) and raw_samples[0][2] == raw_samples[1][2]
    )
    assert all(np.array_equal(a, b) for a, b in zip(raw_samples[0][3], raw_samples[1][3]))
    proof["d028_raw_Dataset_sample_crops_targets_valid_and_all_RNG_exact"] = True
    for name in photo.PROFILES[1:]:
        # Genuine authority module builds the synthetic cache, including its own original training-source pin.
        report = authcache.build(
            manifest,
            root / name,
            name,
            expected_content_sha256=content,
            expected_descriptor_sha256=authcache.sha256(
                authcache.canonical(authphoto.contract(name))
            ).hexdigest(),
        )
        loaded = cache.VerifiedPhotometryCache(
            report["root"],
            manifest,
            name,
            expected_fingerprint=report["fingerprint"],
            expected_canonical_fingerprint=content,
        )
        assert (
            loaded.index["construction_training_source_sha256"]
            == authority._training_source_sha256()
        )
        assert loaded.verify_all_bytes()["photos"] == 3
        datasets = [
            m.UnifiedDataset(manifest, config=m.UnifiedConfig(photometric_profile=name))
            for m in [current, authority]
        ]
        cached = current.UnifiedDataset(
            manifest,
            config=current.UnifiedConfig(photometric_profile=name),
            photometry_cache=loaded,
        )
        for i in range(2):
            assert torch.equal(datasets[0]._image(i), datasets[1]._image(i)) and torch.equal(
                cached._image(i), datasets[0]._image(i)
            )
        proof["cache_cross_source"][name] = {
            "genuine_fd44_builder_current_reader": True,
            "descriptor_and_operator_pins_preserved": True,
            "3synthetic_photos": True,
            "cached_uncached_authority_product_tensors_exact_equal": True,
            "cache_fingerprint": report["fingerprint"],
            "original_construction_training_source_sha256": loaded.index[
                "construction_training_source_sha256"
            ],
        }
proof["actual_no_real_corpus_or_models_accessed"] = True
(product / "reports/product-unified-photometry/cross-source-verification.json").write_text(
    json.dumps(proof, indent=2) + "\n"
)
print(json.dumps(proof, indent=2))
