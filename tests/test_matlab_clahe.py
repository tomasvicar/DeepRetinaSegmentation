"""Synthetic MATLAB-oracle comparisons; real-photo arrays remain outside Git."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_clahe import (
    AV_CLIP_LIMIT,
    AV_KERNEL_PIXELS,
    _clip_histogram,
    adapthisteq_uniform,
    av_matlab_clahe_v,
)

REFERENCE = Path(__file__).parent / "reference/matlab_clahe"
METADATA = json.loads((REFERENCE / "metadata.json").read_text())


@pytest.mark.parametrize("case", METADATA["cases"], ids=lambda case: case["name"])
def test_generated_matlab_synthetic_goldens(case):
    with np.load(REFERENCE / "synthetic.npz", allow_pickle=False) as fixture:
        image = fixture[case["input"]]
        expected = fixture[case["output"]]
    original = image.copy()
    result = adapthisteq_uniform(
        image, num_tiles=case["tiles"], nbins=case["bins"], clip_limit=case["clip"]
    )
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(image, original)
    assert result.shape == image.shape and result.dtype == np.float64
    assert 0 <= result.min() <= result.max() <= 1


def test_synthetic_fixture_identity_is_recorded():
    assert (
        hashlib.sha256((REFERENCE / "synthetic.npz").read_bytes()).hexdigest()
        == METADATA["fixture_sha256"]
    )


def test_histogram_redistribution_preserves_mass_and_cap():
    histogram = np.zeros(256, dtype=np.int64)
    histogram[[0, 60, 128, 250]] = [10000, 73, 97, 500]
    original = histogram.copy()
    limit = int(np.ceil(histogram.sum() / len(histogram))) + 14
    clipped = _clip_histogram(histogram, limit)
    assert clipped.sum() == histogram.sum()
    assert clipped.min() >= 0 and clipped.max() <= limit
    np.testing.assert_array_equal(histogram, original)


def test_constant_half_is_not_globally_stretched_to_white():
    result = adapthisteq_uniform(np.full((16, 16), 0.5), num_tiles=(2, 2), clip_limit=0.005)
    np.testing.assert_array_equal(result, np.full((16, 16), 0.515625))


def test_av_stage_uses_fixed_source_kernel_and_matlab_rounding():
    image = np.linspace(0, 1, 200 * 300).reshape(200, 300)
    expected_tiles = np.floor(np.array(image.shape) / AV_KERNEL_PIXELS + 0.5).astype(int)
    expected = adapthisteq_uniform(image, num_tiles=expected_tiles, clip_limit=AV_CLIP_LIMIT)
    np.testing.assert_array_equal(av_matlab_clahe_v(image), expected)
    with pytest.raises(ValueError, match="integers>=2"):
        av_matlab_clahe_v(np.zeros((16, 16), dtype=float))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), np.uint8),
        np.zeros((2, 2, 3)),
        np.empty((0, 2)),
        np.array([[np.nan, 0.0]]),
        np.array([[1.1, 0.0]]),
        np.array([[-0.1, 0.0]]),
    ],
)
def test_undefined_or_unsupported_input_is_rejected(image):
    with pytest.raises(ValueError):
        adapthisteq_uniform(image)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_tiles": (1, 2)},
        {"num_tiles": (2.5, 2)},
        {"num_tiles": (2, 2, 2)},
        {"clip_limit": -1},
        {"clip_limit": np.nan},
        {"clip_limit": 1.1},
        {"nbins": 1},
        {"nbins": True},
        {"nbins": 256.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        adapthisteq_uniform(np.zeros((4, 4), dtype=float), **kwargs)
