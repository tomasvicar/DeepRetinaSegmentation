"""CPU-only, checksum-pinned unified inference export; never model selection/TEST."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

REMOVED = (
    "training_state",
    "optimizer",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scheduler",
    "rng",
    "rng_state",
)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _state_digest(state):
    import torch

    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode() + b"\0")
        result.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return result.hexdigest()


def export_unified_inference(source, output, *, expected_original_sha256):
    """Preserve original model/config semantics while removing training-only state.

    Returns the separate provenance mapping. The original/frozen checkpoint SHA
    stays authoritative for selection/TEST; the new artifact has its own SHA.
    No model forward, CUDA operation or dataset access is performed.
    """
    if (
        not isinstance(expected_original_sha256, str)
        or len(expected_original_sha256) != 64
        or set(expected_original_sha256) - set("0123456789abcdef")
    ):
        raise ValueError("Require externally pinned original checkpoint SHA256")
    source, output = Path(source), Path(output)
    sidecar = output.with_name(output.name + ".export.json")
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserve existing export: {path}")
    snapshot = source.read_bytes()
    if _digest(snapshot) != expected_original_sha256:
        raise ValueError("Original checkpoint bytes differ from expected SHA256")
    import torch

    from .unified import load_unified

    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Export validation requires default float32 dtype")
    original = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    if (
        not isinstance(original, dict)
        or original.get("format_version") != 1
        or original.get("family") != "unified"
    ):
        raise ValueError("Require a safe unified format_version1 checkpoint")
    if original.get("inference_only") or "export_provenance" in original:
        raise ValueError("Use the original selected checkpoint, not a previous export")
    state = original.get("state_dict")
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(k, str)
            or not torch.is_tensor(v)
            or not v.numel()
            or not torch.isfinite(v).all()
            for k, v in state.items()
        )
    ):
        raise ValueError("Require nonempty finite named model tensors")
    payload = {k: v for k, v in original.items() if k not in REMOVED and k != "checkpoint_sha256"}
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    # Reject other tensor/non-JSON objects rather than silently dropping hidden
    # state or rewriting inference/data/architecture metadata.
    json.dumps(metadata, allow_nan=False)
    payload["state_dict"] = copy.deepcopy(state)  # retains state_dict version metadata
    for name, tensor in state.items():
        payload["state_dict"][name] = tensor.detach().cpu().clone()
    state_sha = _state_digest(state)
    if _state_digest(payload["state_dict"]) != state_sha:
        raise ValueError("CPU export changed model tensor bytes")
    payload["inference_only"] = True
    payload["export_provenance"] = {
        "schema_version": 1,
        "original_checkpoint_sha256": expected_original_sha256,
        "model_state_sha256": state_sha,
        "removed_training_fields": [k for k in REMOVED if k in original],
        "model_transformation": "CPU storage copies only; dtype/shape/tensor bytes unchanged",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    exported = buffer.getvalue()
    # Actual product grammar and strict state shapes, without forward or global
    # RNG changes. Both constructors/loaders are explicitly CPU-only.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        original_model, _, _ = load_unified(io.BytesIO(snapshot), device="cpu")
        exported_model, reloaded, _ = load_unified(io.BytesIO(exported), device="cpu")
    if (
        _state_digest(original_model.state_dict()) != _state_digest(exported_model.state_dict())
        or _state_digest(reloaded["state_dict"]) != state_sha
    ):
        raise ValueError("Product loader model state changed across export")
    if any(v.device.type != "cpu" for v in reloaded["state_dict"].values()):
        raise ValueError("Exported model is not CPU-resident")
    proof = {
        "schema_version": 1,
        "kind": "unified_CPU_inference_export_provenance_NOT_selection",
        **payload["export_provenance"],
        "artifact_sha256": _digest(exported),
        "original_bytes": len(snapshot),
        "artifact_bytes": len(exported),
        "original_epoch": original.get("epoch"),
        "training_source_sha256": original.get("training_source_sha256"),
        "configuration_preserved": reloaded["config"] == original["config"],
        "CPU_model_state_only": True,
        "NN_forward_or_TEST_performed": False,
        "selection_or_TEST_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(exported)
    with sidecar.open("x") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    print(json.dumps(export_unified_inference(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
