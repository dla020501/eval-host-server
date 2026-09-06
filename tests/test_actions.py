"""액션 조립 규약. 깨지면 모든 참가자 액션이 잘못된 슬롯으로 나간다."""

import numpy as np
import pytest

from evalhost import action_groups, fill_action
from evalhost.actions import ACTION_SLICES


def test_full_chunk_passes_through():
    a = np.ones((5, 22), np.float32)
    out = fill_action(a, 22)
    assert out.shape == (5, 22)
    assert np.array_equal(out, a)


def test_short_vector_is_zero_padded():
    out = fill_action([1.0, 2.0, 3.0], 22)
    assert out.shape == (1, 22)
    assert out.dtype == np.float32
    assert np.array_equal(out[0, :3], [1, 2, 3])
    assert not out[0, 3:].any()


def test_slice_name_dict_fills_named_slots_only():
    out = fill_action({"arm_r": np.arange(7), "gripper_r": -1.0}, 22)
    assert out.shape == (1, 22)
    assert np.array_equal(out[0, ACTION_SLICES["arm_r"]], np.arange(7))
    assert out[0, 14] == -1.0
    assert not out[0, 15:].any()


def test_slice_name_dict_chunk_len():
    out = fill_action({"head": [0.25, 0.5]}, 22, chunk_len=4)
    assert out.shape == (4, 22)
    assert np.array_equal(out[:, 16:18], [[0.25, 0.5]] * 4)
    assert not out[:, :16].any()


def test_group_name_keys_are_treated_as_group_dict():
    """`lift`는 구간 이름이자 v2 그룹 이름이다. 그룹 dict로 보고 그대로 통과시킨다."""
    action = {"lift": 0.3}
    assert fill_action(action, 22) is action


def test_group_dict_passes_through_untouched():
    groups = action_groups(joint_q=np.zeros((3, 18), np.float32))
    out = fill_action(groups, 22)
    assert out is groups


def test_dict_action_without_action_dim_raises():
    with pytest.raises(ValueError):
        fill_action({"arm_r": np.zeros(7)})


def test_action_groups_shapes():
    g = action_groups(joint_q=np.zeros(18), lift=0.5, mobile=np.zeros((2, 3)))
    assert g["joint_q"].shape == (1, 18)
    assert g["lift"].shape == (1, 1)
    assert g["mobile"].shape == (2, 3)
    assert all(v.dtype == np.float32 for v in g.values())


def test_action_groups_omits_missing_groups():
    assert set(action_groups(lift=[0.1, 0.2])) == {"lift"}
    assert action_groups(lift=[0.1, 0.2])["lift"].shape == (2, 1)
