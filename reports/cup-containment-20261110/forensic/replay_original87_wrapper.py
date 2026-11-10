import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from deepretina.legacy.av import load_av
from deepretina.legacy.checkpoints import load_single_net
from deepretina.unified import UnifiedConfig, UnifiedDataset

root = Path.home() / "code/DeepRetinaSegmentation_unified_run_87cf045"
run = (
    Path.home()
    / "code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z/forensic/original87-wrapper"
)
run.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location("old_original", root / "scripts/evaluate_original.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
torch.set_num_threads(8)
started = time.perf_counter()
manifest = Path.home() / "code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json"
dataset = UnifiedDataset(manifest, "val", UnifiedConfig())
pid = "sha256:402f3347e4c847e0beade05125d2c98d73751d2ddd0e8352da5d2002afec23b2"
i = dataset.photo_ids.index(pid)
image, target, valid, trace = dataset.sample(i, random_annotator=False, return_trace=True)
models = root / "src/deepretina/pretrained"
universal = load_single_net(models / "legacy-single-net-universal.pt", "cpu")
vessel = load_single_net(models / "legacy-single-net-vessels.pt", "cpu")
av = load_av(models / "legacy-av-v9.pt", "cpu")[0]
original = old.tiled_logits
capture = {}


def spy(*args, **kwargs):
    value = original(*args, **kwargs)
    if value.shape[0] == 3:
        capture["cup"] = value[2].numpy().copy()
    return value


old.tiled_logits = spy
with torch.inference_mode():
    masks = old.segmentation_masks(image, universal, "cpu", vessel=vessel)
t = target[4].numpy() >= 0.5
v = valid[4].numpy()
p = masks[4]
np.save(run / "cup-logits.npy", capture["cup"])
np.savez_compressed(run / "cup-gt.npz", target=t, valid=v)
r = {
    "photo_id": pid,
    "source_commit": "87cf045ab89425c4994df577fd510d210a1a0eab",
    "torch": str(torch.__version__),
    "threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "loaded_models": ["universal", "vessel", "av-v9"],
    "called": "ORIGINAL87 scripts/evaluate_original.py segmentation_masks with input image before subtraction",
    "image_tensor_sha256": hashlib.sha256((image - 0.5).numpy().tobytes()).hexdigest(),
    "native_counts": {
        "tp": int((p & t & v).sum()),
        "fp": int((p & ~t & v).sum()),
        "fn": int((~p & t & v).sum()),
        "valid_pixels": int(v.sum()),
    },
    "logits_sha256": hashlib.sha256(capture["cup"].tobytes()).hexdigest(),
    "elapsed_seconds": time.perf_counter() - started,
    "no_gpu": True,
    "no_test": True,
}
(run / "report.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r))
