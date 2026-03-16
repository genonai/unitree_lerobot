"""Integration test: mock 16D JSON episode → LeRobot conversion."""
import json
import os
import tempfile
import numpy as np
import pytest

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


def _create_mock_episode(episode_dir: str, num_frames: int = 5):
    """Create a minimal mock episode in xr_teleoperate JSON format with 1DOF ee."""
    os.makedirs(os.path.join(episode_dir, "colors"), exist_ok=True)

    data = []
    for i in range(num_frames):
        img_path = f"colors/{i:06d}_color_0.jpg"
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(os.path.join(episode_dir, img_path), img)

        frame = {
            "idx": i,
            "colors": {"color_0": img_path},
            "states": {
                "left_arm":  {"qpos": np.random.randn(7).tolist(), "qvel": [], "torque": []},
                "right_arm": {"qpos": np.random.randn(7).tolist(), "qvel": [], "torque": []},
                "left_ee":   {"qpos": [round(np.random.rand(), 4)], "qvel": [], "torque": []},
                "right_ee":  {"qpos": [round(np.random.rand(), 4)], "qvel": [], "torque": []},
                "body":      {"qpos": []},
            },
            "actions": {
                "left_arm":  {"qpos": np.random.randn(7).tolist(), "qvel": [], "torque": []},
                "right_arm": {"qpos": np.random.randn(7).tolist(), "qvel": [], "torque": []},
                "left_ee":   {"qpos": [round(np.random.rand(), 4)], "qvel": [], "torque": []},
                "right_ee":  {"qpos": [round(np.random.rand(), 4)], "qvel": [], "torque": []},
                "body":      {"qpos": []},
            },
        }
        data.append(frame)

    episode_json = {
        "info": {"version": "1.0.0"},
        "text": {"goal": "test pick and place"},
        "data": data,
    }

    with open(os.path.join(episode_dir, "data.json"), "w") as f:
        json.dump(episode_json, f)


def test_json_dataset_loads_16d():
    """Verify JsonDataset extracts 16D state and action arrays from mock 1DOF JSON."""
    from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = os.path.join(tmpdir, "test_task")
        ep_dir = os.path.join(task_dir, "episode_0000")
        _create_mock_episode(ep_dir, num_frames=3)

        ds = JsonDataset(tmpdir, "Unitree_G1_Inspire_1DOF")
        assert len(ds) == 1

        item = ds.get_item(0)
        assert item["state"].shape == (3, 16), f"state shape: {item['state'].shape}"
        assert item["action"].shape == (3, 16), f"action shape: {item['action'].shape}"
        assert item["data_cfg"]["state_dim"] == 16
        assert item["data_cfg"]["action_dim"] == 16


def test_state_values_are_concatenated_correctly():
    """Verify left_arm(7) + right_arm(7) + left_ee(1) + right_ee(1) = 16 in correct order."""
    from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = os.path.join(tmpdir, "test_task")
        ep_dir = os.path.join(task_dir, "episode_0000")
        os.makedirs(os.path.join(ep_dir, "colors"), exist_ok=True)

        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(ep_dir, "colors/000000_color_0.jpg"), img)

        left_arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        right_arm = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        left_ee = [0.85]
        right_ee = [0.42]

        episode_json = {
            "info": {"version": "1.0.0"},
            "text": {"goal": "test"},
            "data": [{
                "idx": 0,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "states": {
                    "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
                    "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
                    "left_ee": {"qpos": left_ee, "qvel": [], "torque": []},
                    "right_ee": {"qpos": right_ee, "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
                "actions": {
                    "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
                    "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
                    "left_ee": {"qpos": left_ee, "qvel": [], "torque": []},
                    "right_ee": {"qpos": right_ee, "qvel": [], "torque": []},
                    "body": {"qpos": []},
                },
            }],
        }
        with open(os.path.join(ep_dir, "data.json"), "w") as f:
            json.dump(episode_json, f)

        ds = JsonDataset(tmpdir, "Unitree_G1_Inspire_1DOF")
        item = ds.get_item(0)

        expected = np.array(left_arm + right_arm + left_ee + right_ee, dtype=np.float32)
        np.testing.assert_array_almost_equal(item["state"][0], expected, decimal=4)
