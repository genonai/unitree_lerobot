"""Tests for Unitree_G1_Inspire_1DOF ROBOT_CONFIG."""
import pytest
from unitree_lerobot.utils.constants import ROBOT_CONFIGS


def test_inspire_1dof_config_exists():
    assert "Unitree_G1_Inspire_1DOF" in ROBOT_CONFIGS


def test_inspire_1dof_motor_count_is_16():
    cfg = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    assert len(cfg.motors) == 16, f"Expected 16D, got {len(cfg.motors)}D"


def test_inspire_1dof_arm_joints_match_dex1():
    """Arm joints (first 14) must be identical to G1_DEX1_CONFIG."""
    cfg_1dof = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    cfg_dex1 = ROBOT_CONFIGS["Unitree_G1_Dex1"]
    assert cfg_1dof.motors[:14] == cfg_dex1.motors[:14]


def test_inspire_1dof_gripper_names():
    cfg = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    assert cfg.motors[14] == "kLeftGripper"
    assert cfg.motors[15] == "kRightGripper"


def test_inspire_1dof_json_paths():
    cfg = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    expected = ["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"]
    assert cfg.json_state_data_name == expected
    assert cfg.json_action_data_name == expected


def test_inspire_1dof_cameras():
    cfg = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    assert len(cfg.cameras) == 4
    assert "cam_left_high" in cfg.cameras
    assert "cam_right_high" in cfg.cameras
    assert "cam_left_wrist" in cfg.cameras
    assert "cam_right_wrist" in cfg.cameras


def test_inspire_1dof_camera_to_image_key():
    cfg = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"]
    expected = {
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    }
    assert cfg.camera_to_image_key == expected
