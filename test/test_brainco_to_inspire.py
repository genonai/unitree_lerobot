"""Tests for Brainco 26D -> Inspire 26D conversion logic."""
import numpy as np
import pytest

from unitree_lerobot.utils.convert_brainco_to_inspire import (
    BRAINCO_TO_INSPIRE_HAND,
    ARM_SLICE,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    reorder_hand,
    convert_26d_brainco_to_inspire,
    normalize_joint,
)


class TestBraincoToInspireHand:
    """Verify the permutation array maps fingers correctly."""

    def test_permutation_length(self):
        assert len(BRAINCO_TO_INSPIRE_HAND) == 6

    def test_permutation_is_valid(self):
        """All indices 0-5 appear exactly once."""
        assert sorted(BRAINCO_TO_INSPIRE_HAND) == list(range(6))

    def test_pinky_mapping(self):
        """Brainco Pinky (idx 5) -> Inspire Pinky (idx 0)."""
        brainco = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.99])  # only pinky set
        inspire = reorder_hand(brainco)
        assert inspire[0] == pytest.approx(0.99)

    def test_thumb_mapping(self):
        """Brainco Thumb (idx 0) -> Inspire ThumbBend (idx 4)."""
        brainco = np.array([0.77, 0.0, 0.0, 0.0, 0.0, 0.0])  # only thumb set
        inspire = reorder_hand(brainco)
        assert inspire[4] == pytest.approx(0.77)

    def test_thumbaux_mapping(self):
        """Brainco ThumbAux (idx 1) -> Inspire ThumbRotation (idx 5)."""
        brainco = np.array([0.0, 0.55, 0.0, 0.0, 0.0, 0.0])
        inspire = reorder_hand(brainco)
        assert inspire[5] == pytest.approx(0.55)

    def test_full_known_mapping(self):
        """All 6 joints with known distinct values."""
        # Brainco: [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
        brainco = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        inspire = reorder_hand(brainco)
        # Inspire: [Pinky, Ring, Middle, Index, ThumbBend, ThumbRotation]
        expected = np.array([0.6, 0.5, 0.4, 0.3, 0.1, 0.2])
        np.testing.assert_array_almost_equal(inspire, expected)

    def test_reorder_preserves_dtype(self):
        brainco = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        inspire = reorder_hand(brainco)
        assert inspire.dtype == np.float32


class TestConvert26d:
    """Full 26D vector conversion."""

    def test_output_shape(self):
        state = np.random.randn(26).astype(np.float32)
        result = convert_26d_brainco_to_inspire(state)
        assert result.shape == (26,)

    def test_arm_preserved(self):
        """Arm joints [0:14] pass through unchanged."""
        state = np.arange(26, dtype=np.float32)
        result = convert_26d_brainco_to_inspire(state)
        np.testing.assert_array_equal(result[:14], state[:14])

    def test_hands_reordered_independently(self):
        """Left and right hands are reordered independently."""
        state = np.zeros(26, dtype=np.float32)
        # Left hand [14:20]: Brainco order [T, TA, I, M, R, P]
        state[14:20] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # Right hand [20:26]: Brainco order
        state[20:26] = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

        result = convert_26d_brainco_to_inspire(state)
        # Left hand Inspire order: [P, R, M, I, TB, TR]
        np.testing.assert_array_almost_equal(
            result[14:20], [0.6, 0.5, 0.4, 0.3, 0.1, 0.2])
        # Right hand Inspire order
        np.testing.assert_array_almost_equal(
            result[20:26], [1.2, 1.1, 1.0, 0.9, 0.7, 0.8])

    def test_dtype_float32(self):
        state = np.random.randn(26).astype(np.float64)
        result = convert_26d_brainco_to_inspire(state)
        assert result.dtype == np.float32

    def test_wrong_dimension_raises(self):
        """Non-26D input must raise ValueError."""
        with pytest.raises(ValueError, match="Expected shape"):
            convert_26d_brainco_to_inspire(np.zeros(28, dtype=np.float32))

    def test_2d_input_raises(self):
        """2D input must raise ValueError."""
        with pytest.raises(ValueError, match="Expected shape"):
            convert_26d_brainco_to_inspire(np.zeros((1, 26), dtype=np.float32))

    def test_range_map_applied_in_inspire_order(self):
        """range_map indices correspond to Inspire-order joints (post-reorder)."""
        state = np.zeros(26, dtype=np.float32)
        # Left hand Brainco: [Thumb=0.0, ThumbAux=0.0, Index=0.0, Middle=0.0, Ring=0.0, Pinky=1.0]
        state[14:20] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # After reorder: Inspire [Pinky=1.0, Ring=0.0, Middle=0.0, Index=0.0, ThumbBend=0.0, ThumbRotation=0.0]
        # Normalize Inspire[0] (Pinky) from [0,1] → [0,10]
        range_map = {
            "left_hand": [
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 10.0},
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 1.0},
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 1.0},
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 1.0},
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 1.0},
                {"src_min": 0.0, "src_max": 1.0, "dst_min": 0.0, "dst_max": 1.0},
            ],
        }
        result = convert_26d_brainco_to_inspire(state, range_map=range_map)
        # Inspire[0] = Pinky = 1.0 → normalized to 10.0
        assert result[14] == pytest.approx(10.0)
        # Right hand untouched (no "right_hand" key in range_map)
        np.testing.assert_array_equal(result[20:26], np.zeros(6, dtype=np.float32))


class TestNormalizeJoint:
    def test_identity_same_range(self):
        assert normalize_joint(0.5, 0.0, 1.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_scale_up(self):
        assert normalize_joint(0.5, 0.0, 1.0, 0.0, 2.0) == pytest.approx(1.0)

    def test_zero_src_range(self):
        """If src_min == src_max, return dst_min."""
        assert normalize_joint(5.0, 5.0, 5.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_midpoint_mapping(self):
        assert normalize_joint(0.85, 0.0, 1.7, 0.0, 1.0) == pytest.approx(0.5)
