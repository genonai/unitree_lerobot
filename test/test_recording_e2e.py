"""End-to-end test: EpisodeWriter -> data.json -> JsonDataset -> 16D arrays.

Verifies the full recording->conversion pipeline without hardware.
Requires: xr_teleoperate on sys.path (for EpisodeWriter).
Dependencies stubbed in conftest.py (logging_mp, rerun_visualizer).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure xr_teleoperate is importable
# ---------------------------------------------------------------------------
XR_TELEOP_ROOT = Path(__file__).resolve().parent.parent.parent / "xr_teleoperate"
if str(XR_TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_TELEOP_ROOT))

from teleop.utils.episode_writer import EpisodeWriter  # noqa: E402

# Import JsonDataset (conftest.py stubs lerobot deps)
from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------
NUM_FRAMES = 10


def _make_arm_data(frame_idx: int, use_cos: bool = False) -> list[float]:
    """Generate 7D arm joint values using sin/cos wave."""
    fn = np.cos if use_cos else np.sin
    return [float(fn(2 * np.pi * frame_idx / NUM_FRAMES) * 0.5) for _ in range(7)]


def _make_image() -> np.ndarray:
    """64x48 BGR dummy image (small for fast tests)."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


LEFT_EE = [0.8]
RIGHT_EE = [0.3]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRecordingE2E:
    """Full pipeline: EpisodeWriter.add_item -> data.json -> JsonDataset -> 16D."""

    @pytest.fixture()
    def recorded_episode_dir(self, tmp_path: Path) -> Path:
        """Record 10 frames via EpisodeWriter, return the episode directory."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()

        writer = EpisodeWriter(
            task_dir=str(task_dir),
            frequency=30,
            image_size=[64, 48],
            rerun_log=False,
        )
        writer.create_episode()  # REQUIRED: creates episode dir, json_path, color_dir

        for i in range(NUM_FRAMES):
            left_arm = _make_arm_data(i, use_cos=False)
            right_arm = _make_arm_data(i, use_cos=True)
            states = {
                "left_arm": {"qpos": left_arm},
                "right_arm": {"qpos": right_arm},
                "left_ee": {"qpos": LEFT_EE},
                "right_ee": {"qpos": RIGHT_EE},
            }
            # For simplicity, action = state (same values)
            actions = {
                "left_arm": {"qpos": left_arm},
                "right_arm": {"qpos": right_arm},
                "left_ee": {"qpos": LEFT_EE},
                "right_ee": {"qpos": RIGHT_EE},
            }
            colors = {"color_0": _make_image()}

            writer.add_item(colors=colors, states=states, actions=actions)

        writer.save_episode()
        writer.close()

        # Find the episode_XXXX directory
        episode_dirs = sorted(task_dir.glob("episode_*"))
        assert len(episode_dirs) == 1, f"Expected 1 episode, found {len(episode_dirs)}"
        return episode_dirs[0]

    def test_json_structure(self, recorded_episode_dir: Path):
        """Verify data.json has correct structure and field names."""
        json_path = recorded_episode_dir / "data.json"
        assert json_path.exists(), "data.json not created"

        with open(json_path) as f:
            data = json.load(f)

        assert "data" in data, "Missing 'data' key"
        frames = data["data"]
        assert len(frames) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(frames)}"

        frame = frames[0]
        # Check states structure
        assert "states" in frame
        assert "left_arm" in frame["states"]
        assert "qpos" in frame["states"]["left_arm"]
        assert len(frame["states"]["left_arm"]["qpos"]) == 7
        assert "right_arm" in frame["states"]
        assert len(frame["states"]["right_arm"]["qpos"]) == 7
        assert "left_ee" in frame["states"]
        assert len(frame["states"]["left_ee"]["qpos"]) == 1
        assert "right_ee" in frame["states"]
        assert len(frame["states"]["right_ee"]["qpos"]) == 1

        # Check actions structure (same fields)
        assert "actions" in frame
        assert len(frame["actions"]["left_arm"]["qpos"]) == 7

        # Check colors directory has images
        colors_dir = recorded_episode_dir / "colors"
        assert colors_dir.exists(), "colors/ directory not created"
        jpg_files = list(colors_dir.glob("*.jpg"))
        assert len(jpg_files) >= NUM_FRAMES, f"Expected >= {NUM_FRAMES} images, got {len(jpg_files)}"

    def test_conversion_to_16d(self, recorded_episode_dir: Path):
        """Verify JsonDataset loads recorded episode as 16D arrays."""
        # JsonDataset expects root dir whose children are task dirs
        # tmp_path/test_task/episode_XXXX -> pass tmp_path as data_dirs
        root_dir = recorded_episode_dir.parent.parent
        ds = JsonDataset(data_dirs=root_dir, robot_type="Unitree_G1_Inspire_1DOF")

        assert len(ds.episode_ids) == 1

        item = ds.get_item(0)
        states = item["state"]
        actions = item["action"]

        # Shape check: (num_frames, 16)
        assert states.shape == (NUM_FRAMES, 16), f"states shape {states.shape} != ({NUM_FRAMES}, 16)"
        assert actions.shape == (NUM_FRAMES, 16), f"actions shape {actions.shape} != ({NUM_FRAMES}, 16)"

    def test_values_match_input(self, recorded_episode_dir: Path):
        """Verify converted values match what was fed to EpisodeWriter."""
        root_dir = recorded_episode_dir.parent.parent
        ds = JsonDataset(data_dirs=root_dir, robot_type="Unitree_G1_Inspire_1DOF")
        item = ds.get_item(0)
        states = item["state"]

        for i in range(NUM_FRAMES):
            expected_left_arm = _make_arm_data(i, use_cos=False)
            expected_right_arm = _make_arm_data(i, use_cos=True)

            # Indices [0:7] = left_arm, [7:14] = right_arm
            np.testing.assert_allclose(
                states[i, 0:7], expected_left_arm, atol=1e-6,
                err_msg=f"Frame {i} left_arm mismatch",
            )
            np.testing.assert_allclose(
                states[i, 7:14], expected_right_arm, atol=1e-6,
                err_msg=f"Frame {i} right_arm mismatch",
            )

            # Index [14] = left_ee (0.8), [15] = right_ee (0.3)
            assert abs(states[i, 14] - 0.8) < 1e-6, f"Frame {i} left_ee mismatch"
            assert abs(states[i, 15] - 0.3) < 1e-6, f"Frame {i} right_ee mismatch"

    def test_gripper_range(self, recorded_episode_dir: Path):
        """Verify gripper values are in [0, 1] range."""
        root_dir = recorded_episode_dir.parent.parent
        ds = JsonDataset(data_dirs=root_dir, robot_type="Unitree_G1_Inspire_1DOF")
        item = ds.get_item(0)
        states = item["state"]

        grip_values = states[:, 14:16]
        assert np.all(grip_values >= 0.0), "Grip values below 0"
        assert np.all(grip_values <= 1.0), "Grip values above 1"
