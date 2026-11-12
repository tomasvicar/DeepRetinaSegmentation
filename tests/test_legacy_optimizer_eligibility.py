"""Synthetic AdamW metadata + strict CPU reload, no NN forward or real inputs."""

import copy
import io
from dataclasses import asdict

import pytest
import torch

from deepretina import freeze_eligibility as f
from deepretina.selection import SelectionError
from deepretina.unified import CHANNELS, UnifiedConfig, build_unified


@pytest.fixture(scope="module")
def saved_legacy():
    config = UnifiedConfig(architecture="legacy_universal", base_channels=32)
    model = build_unified(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr * config.min_lr_ratio, weight_decay=config.weight_decay
    )
    unused = []
    for i, (name, p) in enumerate(model.named_parameters()):
        if name in {"up_conv1.1.bn.weight", "up_conv1.1.bn.bias"}:
            unused.append(i)
        else:
            p.grad = torch.ones_like(p)
    optimizer.step()  # Genuine moments without a forward or autograd graph.
    payload = {
        "family": "unified",
        "format_version": 1,
        "channels": list(CHANNELS),
        "normalization": "RGB/255-0.5",
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "steps": 1,
        "training_state": {
            "optimizer": optimizer.state_dict(),
            "rng": dict.fromkeys(
                ["torch", "python", "numpy", "sampler", "augmentation", "cuda"], None
            ),
        },
    }
    snapshot = io.BytesIO()
    torch.save(payload, snapshot)
    assert len(unused) == 2
    return snapshot.getvalue(), payload, unused


def test_exact_unused_BN_pair_and_genuine_used_moments_accept(saved_legacy):
    raw, payload, unused = saved_legacy
    assert set(payload["training_state"]["optimizer"]["state"]) == set(
        payload["training_state"]["optimizer"]["param_groups"][0]["params"]
    ) - set(unused)
    f._optimizer(raw, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_used",
        "extra_id",
        "order",
        "compact_label",
        "fabricated_unused",
        "missing_moment",
        "dtype",
        "step",
        "bool_state_id",
        "empty_moment",
    ],
)
def test_corrupt_legacy_optimizer_rejected(saved_legacy, mutation):
    raw, source, unused = saved_legacy
    p = copy.deepcopy(source)
    o = p["training_state"]["optimizer"]
    used = min(o["state"])
    if mutation == "missing_used":
        o["state"].pop(used)
    if mutation == "extra_id":
        o["state"][99999] = copy.deepcopy(o["state"][used])
    if mutation == "order":
        o["param_groups"][0]["params"][:2] = [1, 0]
    if mutation == "compact_label":
        p["config"]["architecture"] = "compact_unet"
    if mutation == "fabricated_unused":
        o["state"][unused[0]] = copy.deepcopy(o["state"][used])
    if mutation == "missing_moment":
        o["state"][used].pop("exp_avg_sq")
    if mutation == "dtype":
        o["state"][used]["exp_avg"] = o["state"][used]["exp_avg"].double()
    if mutation == "step":
        o["state"][used]["step"] = torch.tensor(0.0)
    if mutation == "bool_state_id":
        o["state"][False] = o["state"].pop(0)
    if mutation == "empty_moment":
        o["state"][used]["exp_avg"] = torch.empty(0)
    with pytest.raises((SelectionError, ValueError, RuntimeError)):
        f._optimizer(raw, p)


def test_actual_BN_flag_required(saved_legacy, monkeypatch):
    original = f.gate.load_unified

    def loaded(*args, **kwargs):
        model, payload, device = original(*args, **kwargs)
        model.up_conv1[1].do_batch = 1
        return model, payload, device

    monkeypatch.setattr(f.gate, "load_unified", loaded)
    with pytest.raises(SelectionError):
        f._optimizer(saved_legacy[0], saved_legacy[1])


@pytest.mark.parametrize(
    "architecture,pin,tag,accept",
    [
        ("compact_unet", f.VERIFIER, None, True),
        ("legacy_universal", f.LEGACY_VERIFIER, f.LEGACY_OPTIMIZER_SCHEMA, True),
        ("legacy_universal", f.VERIFIER, f.LEGACY_OPTIMIZER_SCHEMA, False),
        ("compact_unet", f.LEGACY_VERIFIER, None, False),
        ("legacy_universal", "0" * 64, f.LEGACY_OPTIMIZER_SCHEMA, False),
        ("legacy_universal", f.LEGACY_VERIFIER, None, False),
        ("compact_unet", f.VERIFIER, f.LEGACY_OPTIMIZER_SCHEMA, False),
    ],
)
def test_architecture_specific_verifier_revision(architecture, pin, tag, accept):
    qa = {"verifier_sha256": pin, "source_git_verified": True}
    if tag is not None:
        qa["optimizer_binding_schema"] = tag
    call = lambda: f._standalone_verifier(
        qa, f.PROFILES["standalone_72"], {"config": {"architecture": architecture}}
    )
    if accept:
        call()
    else:
        with pytest.raises(SelectionError):
            call()


def test_real_compact_still_requires_every_parameter_state(saved_legacy):
    config = UnifiedConfig(base_channels=2, depth=2)
    model = build_unified(config)
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    payload = copy.deepcopy(saved_legacy[1])
    payload.update(config=asdict(config), state_dict=model.state_dict())
    payload["training_state"]["optimizer"] = optimizer.state_dict()
    stream = io.BytesIO()
    torch.save(payload, stream)
    f._optimizer(stream.getvalue(), payload)
    payload["training_state"]["optimizer"]["state"].pop(0)
    with pytest.raises(SelectionError):
        f._optimizer(stream.getvalue(), payload)


def test_legacy_revision_cannot_be_claimed_by_other_source_family():
    qa = {
        "verifier_sha256": f.LEGACY_VERIFIER,
        "source_git_verified": True,
        "optimizer_binding_schema": f.LEGACY_OPTIMIZER_SCHEMA,
    }
    with pytest.raises(SelectionError):
        f._standalone_verifier(
            qa, f.PROFILES["standalone_0b"], {"config": {"architecture": "legacy_universal"}}
        )
