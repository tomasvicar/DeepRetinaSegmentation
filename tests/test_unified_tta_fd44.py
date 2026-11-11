"""TTA integration with fd44 BF16 checkpoint metadata and immutable RGB caches."""

from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_photometry_cache import built
from test_photometry_cache import corpus as cache_corpus  # noqa: F401

from deepretina import unified as u


@pytest.mark.parametrize("mode", ["none", "flips2", "rot4", "dihedral8"])
def test_bf16_source72_style_checkpoint_defaults_raw_and_disables_outer_autocast(tmp_path, mode):
    # Synthetic weights only; source72 saved BF16 config predates photometric_profile.
    cfg = u.UnifiedConfig(base_channels=2, depth=1, mixed_precision="bf16", tile_size=32, overlap=8)
    model = u.build_unified(cfg).eval()
    saved = asdict(cfg)
    del saved["photometric_profile"]
    path = tmp_path / "synthetic-source72-metadata.pt"
    torch.save(
        {
            "format_version": 1,
            "family": "unified",
            "channels": list(u.CHANNELS),
            "normalization": "RGB/255-0.5",
            "config": saved,
            "state_dict": model.state_dict(),
            "training_source_sha256": "3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28",
        },
        path,
    )
    restored, payload, device = u.load_unified(path, "cpu")
    inference = u.inference_config(payload)
    assert inference.mixed_precision == "bf16" and inference.photometric_profile == "raw"
    assert "photometric_profile" not in payload["config"]
    assert all(p.dtype == torch.float32 for p in restored.parameters())
    image = torch.linspace(0, 1, 3 * 35 * 49).reshape(3, 35, 49)
    expected = u.predict_probabilities(restored, image, device, inference, 32, 8, tta=mode)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = u.predict_probabilities(restored, image, device, inference, 32, 8, tta=mode)
    assert np.array_equal(actual, expected)
    assert actual.shape == (5, 35, 49) and actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual[1:3].sum(axis=0), 1, rtol=1e-6, atol=1e-7)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile", ["clahe_hsv_v1", "legacy_lc_clahe_v1"])
def test_cached_full_image_once_per_val_photo_before_eight_views(
    tmp_path, cache_corpus, profile, monkeypatch  # noqa: F811 - imported pytest fixture
):
    from test_unified_tta import Pointwise

    manifest, rows, _ = cache_corpus
    _, cache = built(tmp_path, cache_corpus, profile)
    identity = dict(cache.identity)
    calls = []
    rgb = cache.rgb

    def once(pid):
        calls.append(pid)
        return rgb(pid)

    monkeypatch.setattr(cache, "rgb", once)

    def forbidden(*a, **kw):
        raise AssertionError("Already processed full-image cache must never be processed again")

    monkeypatch.setattr(u, "apply_photometry", forbidden)
    cfg = u.UnifiedConfig(
        photometric_profile=profile, tile_size=32, overlap=8, mixed_precision="bf16"
    )
    model = Pointwise()
    model.supervised_channels = [True] * 5
    result = u.evaluate_unified_model(
        model, manifest, config=cfg, photometry_cache=cache, tta="dihedral8"
    )
    eligible = {r["photo_id"] for r in rows if r["split"] == "val"}
    assert set(calls) == eligible and len(calls) == len(eligible)
    assert cache.identity == identity
    assert result["photos"] == len(eligible) and result["tta"]["mode"] == "dihedral8"
