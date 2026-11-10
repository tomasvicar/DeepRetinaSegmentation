"""The CPU optimization must preserve the historical disk morphology exactly."""

import numpy as np
import pytest
from skimage.morphology import binary_closing, binary_erosion, disk, footprint_from_sequence


@pytest.mark.parametrize("radius", [6, 10])
def test_crosses_reconstruct_the_original_disk(radius):
    np.testing.assert_array_equal(
        footprint_from_sequence(disk(radius, decomposition="crosses")), disk(radius)
    )


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 8), (31, 45), (123, 157)])
@pytest.mark.filterwarnings("ignore:.*is deprecated.*:FutureWarning")
def test_exact_morphology_including_tiny_empty_full_and_border_masks(shape):
    generator = np.random.default_rng(42)
    masks = [np.zeros(shape, bool), np.ones(shape, bool)]
    masks.extend(
        generator.random(shape) < probability for probability in [0.01, 0.1, 0.5, 0.9, 0.99]
    )
    border = np.zeros(shape, bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    masks.append(border)
    for mask in masks:
        for operation, radius in [(binary_closing, 10), (binary_erosion, 6)]:
            expected = operation(mask, disk(radius))
            actual = operation(mask, disk(radius, decomposition="crosses"))
            np.testing.assert_array_equal(actual, expected)
        expected = binary_erosion(binary_closing(mask, disk(10)), disk(6))
        actual = binary_erosion(
            binary_closing(mask, disk(10, decomposition="crosses")),
            disk(6, decomposition="crosses"),
        )
        np.testing.assert_array_equal(actual, expected)
