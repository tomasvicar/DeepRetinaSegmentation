"""Measured research candidate only; production Sobel remains unchanged."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "reports/legacy/matlab-residual-20261111"))
from candidate_sobel import candidate
from diagnose_sobel import sobel


@pytest.mark.parametrize("index", range(2, 9))
def test_observed_gx_path_model_exact_on_supported_random_oracle_shapes(index):
    with np.load(ROOT / "tests/reference/matlab_sobel/probe.npz", allow_pickle=False) as ref:
        distance = ref[f"{index}_distance"]
        x, y = candidate(distance)
        assert np.array_equal(x, ref[f"{index}_gx"])
        # Only the evidenced terminal interior is replaced for Gy; unmatched
        # left/corner behavior must remain visible rather than rebaselined.
        old = sobel(distance)[1]
        assert np.count_nonzero(y != ref[f"{index}_gy"]) <= np.count_nonzero(
            old != ref[f"{index}_gy"]
        )


def test_tiny_counterexample_and_remaining_gy_error_are_retained():
    with np.load(ROOT / "tests/reference/matlab_sobel/probe.npz", allow_pickle=False) as ref:
        d = ref["0_distance"]
        x, y = candidate(d)
        assert np.array_equal(x, sobel(d)[0])
        assert not np.array_equal(x, ref["0_gx"])
        x, y = candidate(ref["3_distance"])
        assert np.count_nonzero(y != ref["3_gy"]) > 0


def test_constant_has_zero_gradients_without_epsilon_or_bin_adjustment():
    x, y = candidate(np.full((33, 35), 10, np.float32))
    assert np.count_nonzero(x) == np.count_nonzero(y) == 0
