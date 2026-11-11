"""Reference nearest-center coordinates, including the retained DiaRet tie."""

from pathlib import Path

import numpy as np
import pytest

from deepretina.legacy.matlab_resize import imresize_nearest_mask


@pytest.mark.parametrize("index", range(24))
def test_retained_matlab_nearest_goldens(index):
    path = Path(__file__).parent / "reference/matlab_sobel/nearest.npz"
    with np.load(path, allow_pickle=False) as ref:
        expected = ref[f"{index}_expected"]
        assert np.array_equal(
            imresize_nearest_mask(ref[f"{index}_input"], expected.shape), expected
        )


def test_diaret_reference_half_pixel_source_row_without_photo():
    # Retained actual MATLAB resized-FOV disagreement locations: (837,127)
    # and (837,1164), both true. Its one-based source row is 960, not961.
    # A synthetic row label image isolates that coordinate without storing FOV.
    rows = np.arange(1152, dtype=np.int16)[:, None]
    output = imresize_nearest_mask(rows, (1005, 1))
    assert output[837, 0] == 959
    old_row = int(np.floor((837 + 0.5) * 1152 / 1005))
    assert old_row == 960
