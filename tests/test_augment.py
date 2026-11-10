"""Scientific geometry, ignored-support and reproducibility checks for augmentation."""

import pytest
import torch

from deepretina.augment import augment, warp_joint


def example():
    target = torch.zeros(2, 25, 25)
    target[0, 3:10, 4:9] = 1
    target[1, 13:20, 12:20] = 1
    image = torch.cat([target, target[:1]], dim=0)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, 0:5, 15:20] = False
    return image, target, valid


def test_joint_rotation_matches_integer_reference_for_all_channels():
    image, target, valid = example()
    image_out, target_out, valid_out = warp_joint(image, target, valid, angle_degrees=90)
    torch.testing.assert_close(image_out, torch.rot90(image, 1, (-2, -1)), atol=2e-6, rtol=0)
    assert torch.equal(target_out, torch.rot90(target, 1, (-2, -1)))
    assert torch.equal(valid_out, torch.rot90(valid, 1, (-2, -1)))
    assert all(x.is_contiguous() for x in (image_out, target_out, valid_out))


def test_zoom_out_padding_never_becomes_negative_supervision():
    image = torch.ones(3, 31, 41)
    target = torch.ones(2, 31, 41)
    valid = torch.ones_like(target, dtype=torch.bool)
    image_out, target_out, valid_out = warp_joint(image, target, valid, angle_degrees=13, scale=0.7)
    assert not valid_out[:, 0].any() and not valid_out[:, -1].any()
    assert not valid_out[:, :, 0].any() and not valid_out[:, :, -1].any()
    assert (~valid_out).any() and valid_out.any()
    assert torch.equal(target_out[valid_out], torch.ones_like(target_out[valid_out]))
    # Reflected photographic padding does not invent target background validity.
    torch.testing.assert_close(image_out, torch.ones_like(image_out))
    assert not valid_out[target_out == 0].any()


def test_nearest_targets_and_unknown_av_support_stay_paired():
    artery = torch.zeros(29, 29)
    vein = torch.zeros_like(artery)
    artery[4:10, 4:20] = 1
    vein[15:20, 5:25] = 1
    vessel = artery + vein
    vessel[11:13, 5:25] = 1  # Known vessel, unknown artery/vein.
    target = torch.stack([vessel, artery, vein])
    known = (artery + vein).bool()
    valid = torch.stack([torch.ones_like(known), known, known])
    image = target.clone()
    _, target_out, valid_out = warp_joint(image, target, valid, angle_degrees=17, scale=0.85)
    assert set(target_out.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(valid_out[1], valid_out[2])
    assert torch.equal(
        (target_out[1] + target_out[2])[valid_out[1]], torch.ones_like(target_out[1][valid_out[1]])
    )
    unknown_vessel = (target_out[0] == 1) & (target_out[1] == 0) & (target_out[2] == 0)
    assert unknown_vessel.any()
    assert not valid_out[1:, unknown_vessel].any()


@pytest.mark.parametrize("level", ["mild", "strong"])
def test_seed_reproduces_exact_outputs_and_preserves_input(level):
    inputs = example()
    copies = tuple(x.clone() for x in inputs)
    first = augment(*inputs, level=level, generator=torch.Generator().manual_seed(11))
    second = augment(*inputs, level=level, generator=torch.Generator().manual_seed(11))
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    for a, b in zip(inputs, copies):
        assert torch.equal(a, b)
    assert torch.isfinite(first[0]).all()
    assert 0 <= first[0].min() <= first[0].max() <= 1
    assert first[0].dtype == inputs[0].dtype
    assert first[1].dtype == inputs[1].dtype
    assert first[2].dtype == torch.bool
    assert all(x.is_contiguous() for x in first)


def test_global_torch_seed_controls_helper_and_basic_adds_no_transform():
    inputs = example()
    torch.manual_seed(107)
    first = augment(*inputs, level="strong")
    torch.manual_seed(107)
    second = augment(*inputs, level="strong")
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    basic = augment(*inputs, level="basic")
    assert all(torch.equal(a, b) for a, b in zip(inputs, basic))


def test_invalid_inputs_fail_clearly():
    image, target, valid = example()
    with pytest.raises(ValueError, match="Unknown augmentation"):
        augment(image, target, valid, level="unknown")
    with pytest.raises(TypeError, match="valid must be bool"):
        augment(image, target, valid.float())
    with pytest.raises(ValueError, match="positive"):
        warp_joint(image, target, valid, scale=0)
    with pytest.raises(ValueError, match="same nonempty spatial shape"):
        warp_joint(image, target[:, :-1], valid[:, :-1])
    image[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite image"):
        augment(image, target, valid)
