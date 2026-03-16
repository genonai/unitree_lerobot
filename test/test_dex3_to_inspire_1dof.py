"""Tests for Dex3 28D → Inspire 1DOF 16D conversion logic."""
import numpy as np
import pytest

from unitree_lerobot.utils.convert_dex3_to_inspire_1dof import (
    collapse_hand_to_grip,
    convert_28d_to_16d,
    ARM_LEFT_SLICE,
    ARM_RIGHT_SLICE,
    HAND_LEFT_SLICE,
    HAND_RIGHT_SLICE,
)


class TestCollapseHandToGrip:
    def test_average_of_uniform_values(self):
        """All joints at same value → grip equals that value."""
        hand = np.full(7, 0.5)
        assert collapse_hand_to_grip(hand) == pytest.approx(0.5)

    def test_average_of_mixed_values(self):
        """Average of 7 mixed values."""
        hand = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.4])
        expected = np.mean(hand)
        assert collapse_hand_to_grip(hand) == pytest.approx(expected)

    def test_normalize_fully_open(self):
        """All joints at min (open) → grip = 1.0 (Inspire open convention)."""
        hand = np.full(7, 0.0)
        grip = collapse_hand_to_grip(hand, normalize=True, hand_min=0.0, hand_max=1.7)
        assert grip == pytest.approx(1.0)

    def test_normalize_fully_closed(self):
        """All joints at max (closed) → grip = 0.0 (Inspire closed convention)."""
        hand = np.full(7, 1.7)
        grip = collapse_hand_to_grip(hand, normalize=True, hand_min=0.0, hand_max=1.7)
        assert grip == pytest.approx(0.0)

    def test_normalize_midpoint(self):
        """All joints at midpoint → grip = 0.5."""
        hand = np.full(7, 0.85)  # midpoint of [0, 1.7]
        grip = collapse_hand_to_grip(hand, normalize=True, hand_min=0.0, hand_max=1.7)
        assert grip == pytest.approx(0.5)

    def test_normalize_clips_out_of_range(self):
        """Values outside [min, max] are clipped."""
        hand = np.full(7, 2.0)  # beyond max of 1.7
        grip = collapse_hand_to_grip(hand, normalize=True, hand_min=0.0, hand_max=1.7)
        assert grip == pytest.approx(0.0)  # clipped to max → fully closed → 0.0

    def test_no_normalize_returns_raw_average(self):
        """Without normalize, returns raw average."""
        hand = np.array([0.5, 1.0, 1.5, 0.3, 0.7, 0.9, 1.2])
        grip = collapse_hand_to_grip(hand, normalize=False)
        assert grip == pytest.approx(float(np.mean(hand)))


class TestConvert28dTo16d:
    def test_output_shape(self):
        """28D input → 16D output."""
        state = np.random.randn(28).astype(np.float32)
        result = convert_28d_to_16d(state)
        assert result.shape == (16,)

    def test_arm_values_preserved(self):
        """Arm joints (0:14) pass through unchanged."""
        state = np.arange(28, dtype=np.float32)
        result = convert_28d_to_16d(state)
        np.testing.assert_array_equal(result[:7], state[:7])    # left arm
        np.testing.assert_array_equal(result[7:14], state[7:14])  # right arm

    def test_grip_is_mean_of_hand(self):
        """Grip values are mean of hand joints."""
        state = np.zeros(28, dtype=np.float32)
        state[14:21] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # left hand
        state[21:28] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]  # right hand
        result = convert_28d_to_16d(state)
        assert result[14] == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        assert result[15] == pytest.approx(np.mean([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]))

    def test_dtype_is_float32(self):
        """Output is float32."""
        state = np.random.randn(28).astype(np.float64)
        result = convert_28d_to_16d(state)
        assert result.dtype == np.float32

    def test_known_values(self):
        """Full known-value test."""
        left_arm = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        right_arm = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        left_hand = np.full(7, 0.5)   # avg = 0.5
        right_hand = np.full(7, 0.8)  # avg = 0.8
        state = np.concatenate([left_arm, right_arm, left_hand, right_hand]).astype(np.float32)

        result = convert_28d_to_16d(state)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0.5, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestSliceConstants:
    def test_slices_cover_28d(self):
        """Verify slice constants cover full 28D without overlap."""
        indices = set()
        for s in [ARM_LEFT_SLICE, ARM_RIGHT_SLICE, HAND_LEFT_SLICE, HAND_RIGHT_SLICE]:
            new_indices = set(range(s.start, s.stop))
            assert not indices.intersection(new_indices), f"Overlap at {indices.intersection(new_indices)}"
            indices.update(new_indices)
        assert indices == set(range(28))
