"""
Convert a Dex3 28D LeRobot dataset to Inspire 1DOF 16D format.

Collapses Dex3 hand DOFs (7D per hand) into a single grip value (1D per hand)
by averaging all finger joint positions. Keeps arm joints (14D) unchanged.

Dex3 28D layout:
  [0:7]   = Left arm
  [7:14]  = Right arm
  [14:21] = Left hand  (thumb0, thumb1, thumb2, middle0, middle1, index0, index1)
  [21:28] = Right hand (thumb0, thumb1, thumb2, index0, index1, middle0, middle1)

Inspire 1DOF 16D layout:
  [0:7]   = Left arm
  [7:14]  = Right arm
  [14]    = Left gripper  (mean of left hand 7D, normalized to [0,1])
  [15]    = Right gripper (mean of right hand 7D, normalized to [0,1])

Usage:
  python unitree_lerobot/utils/convert_dex3_to_inspire_1dof.py \
      --src-repo-id unitreerobotics/G1_Dex3_ToastedBread_Dataset \
      --dst-repo-id jihun/g1_inspire_1dof_toastedbread \
      --normalize

  # With custom normalization range (if Dex3 joint values aren't [0,1]):
  python unitree_lerobot/utils/convert_dex3_to_inspire_1dof.py \
      --src-repo-id unitreerobotics/G1_Dex3_ToastedBread_Dataset \
      --dst-repo-id jihun/g1_inspire_1dof_toastedbread \
      --normalize --hand-min 0.0 --hand-max 1.7
"""

import shutil
import argparse
import numpy as np
import torch
import tqdm
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


# Dex3 28D index ranges
ARM_LEFT_SLICE = slice(0, 7)
ARM_RIGHT_SLICE = slice(7, 14)
HAND_LEFT_SLICE = slice(14, 21)
HAND_RIGHT_SLICE = slice(21, 28)


def collapse_hand_to_grip(hand_values: np.ndarray, normalize: bool = False,
                           hand_min: float = 0.0, hand_max: float = 1.0) -> float:
    """
    Collapse 7D Dex3 hand joint values to a single grip float.

    Strategy: Average all 7 joint values. Optionally normalize to [0,1].

    Args:
        hand_values: Array of 7 joint positions
        normalize: If True, normalize the average to [0,1] using hand_min/hand_max
        hand_min: Minimum expected joint value (fully open)
        hand_max: Maximum expected joint value (fully closed)

    Returns:
        Single float grip value
    """
    avg = float(np.mean(hand_values))
    if normalize and hand_max != hand_min:
        # Normalize: 0 = open, 1 = closed in Dex3 convention
        # But Inspire convention: 1 = open, 0 = closed
        # So we invert: grip = 1 - normalized
        normalized = np.clip((avg - hand_min) / (hand_max - hand_min), 0.0, 1.0)
        return float(1.0 - normalized)  # Invert to match Inspire convention
    return avg


def convert_28d_to_16d(state_28d: np.ndarray, normalize: bool = False,
                        hand_min: float = 0.0, hand_max: float = 1.0) -> np.ndarray:
    """
    Convert a single 28D state/action vector to 16D.

    Args:
        state_28d: Array of shape (28,)
        normalize: Whether to normalize hand values
        hand_min: Min hand joint value
        hand_max: Max hand joint value

    Returns:
        Array of shape (16,)
    """
    left_arm = state_28d[ARM_LEFT_SLICE]      # 7D
    right_arm = state_28d[ARM_RIGHT_SLICE]    # 7D
    left_hand = state_28d[HAND_LEFT_SLICE]    # 7D
    right_hand = state_28d[HAND_RIGHT_SLICE]  # 7D

    left_grip = collapse_hand_to_grip(left_hand, normalize, hand_min, hand_max)
    right_grip = collapse_hand_to_grip(right_hand, normalize, hand_min, hand_max)

    return np.concatenate([left_arm, right_arm, [left_grip], [right_grip]]).astype(np.float32)


def analyze_hand_range(src_dataset: LeRobotDataset) -> dict:
    """
    Scan dataset to find min/max of hand joint values (useful for normalization).
    """
    all_left_hand = []
    all_right_hand = []

    for idx in range(len(src_dataset)):
        frame = src_dataset[idx]
        state = frame["observation.state"].numpy()
        if len(state) == 28:
            all_left_hand.append(state[HAND_LEFT_SLICE])
            all_right_hand.append(state[HAND_RIGHT_SLICE])

    left_hand = np.array(all_left_hand)
    right_hand = np.array(all_right_hand)

    return {
        "left_hand_min": float(left_hand.min()),
        "left_hand_max": float(left_hand.max()),
        "left_hand_mean": float(left_hand.mean()),
        "right_hand_min": float(right_hand.min()),
        "right_hand_max": float(right_hand.max()),
        "right_hand_mean": float(right_hand.mean()),
        "overall_min": float(min(left_hand.min(), right_hand.min())),
        "overall_max": float(max(left_hand.max(), right_hand.max())),
    }


def convert_dex3_to_inspire_1dof(
    src_repo_id: str,
    dst_repo_id: str,
    *,
    normalize: bool = False,
    hand_min: float = 0.0,
    hand_max: float = 1.0,
    auto_range: bool = False,
    push_to_hub: bool = False,
    max_episodes: int = 0,
):
    """
    Convert a Dex3 28D LeRobot dataset to Inspire 1DOF 16D format.

    Args:
        src_repo_id: Source dataset repo ID (e.g., unitreerobotics/G1_Dex3_ToastedBread_Dataset)
        dst_repo_id: Destination dataset repo ID
        normalize: Whether to normalize hand values to [0,1]
        hand_min: Min hand joint value for normalization
        hand_max: Max hand joint value for normalization
        auto_range: If True, scan dataset to determine hand_min/hand_max automatically
        push_to_hub: Whether to push the converted dataset to HuggingFace Hub
        max_episodes: Max episodes to convert (0 = all)
    """
    print(f"Loading source dataset: {src_repo_id}")
    src_dataset = LeRobotDataset(repo_id=src_repo_id)

    # Verify source is 28D
    src_state_dim = src_dataset.meta.shapes["observation.state"][0]
    assert src_state_dim == 28, f"Expected 28D source, got {src_state_dim}D"

    total_episodes = src_dataset.meta.total_episodes
    total_frames = src_dataset.meta.total_frames
    print(f"Source: {total_episodes} episodes, {total_frames} frames, {src_state_dim}D")

    # Auto-detect hand range if requested
    if auto_range:
        print("Scanning dataset to determine hand joint range...")
        hand_stats = analyze_hand_range(src_dataset)
        hand_min = hand_stats["overall_min"]
        hand_max = hand_stats["overall_max"]
        print(f"  Hand range: [{hand_min:.4f}, {hand_max:.4f}]")
        print(f"  Left  hand: [{hand_stats['left_hand_min']:.4f}, {hand_stats['left_hand_max']:.4f}] (mean: {hand_stats['left_hand_mean']:.4f})")
        print(f"  Right hand: [{hand_stats['right_hand_min']:.4f}, {hand_stats['right_hand_max']:.4f}] (mean: {hand_stats['right_hand_mean']:.4f})")
        normalize = True

    # Determine how many episodes to convert
    num_episodes = min(max_episodes, total_episodes) if max_episodes > 0 else total_episodes
    print(f"Converting {num_episodes}/{total_episodes} episodes to 16D...")

    # Create destination dataset
    dst_path = HF_LEROBOT_HOME / dst_repo_id
    if dst_path.exists():
        shutil.rmtree(dst_path)

    motors_16d = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"].motors
    cameras_16d = ROBOT_CONFIGS["Unitree_G1_Inspire_1DOF"].cameras

    # Build features dict
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [motors_16d],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": [motors_16d],
        },
    }

    # Check which cameras exist in source
    src_cameras = [key.replace("observation.images.", "")
                   for key in src_dataset.meta.shapes.keys()
                   if key.startswith("observation.images.")]
    print(f"Source cameras: {src_cameras}")

    for cam in src_cameras:
        cam_shape = src_dataset.meta.shapes[f"observation.images.{cam}"]
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": cam_shape,
            "names": ["height", "width", "channel"],
        }

    dst_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=src_dataset.fps,
        robot_type="Unitree_G1_Inspire_1DOF",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )

    # Convert episodes
    frame_idx = 0
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Converting episodes"):
        # Get episode boundaries
        ep_from = src_dataset.episode_data_index["from"][ep_idx].item()
        ep_to = src_dataset.episode_data_index["to"][ep_idx].item()
        ep_length = ep_to - ep_from

        for i in range(ep_length):
            src_frame = src_dataset[ep_from + i]

            # Convert state 28D → 16D
            state_28d = src_frame["observation.state"].numpy()
            state_16d = convert_28d_to_16d(state_28d, normalize, hand_min, hand_max)

            # Convert action 28D → 16D
            action_28d = src_frame["action"].numpy()
            action_16d = convert_28d_to_16d(action_28d, normalize, hand_min, hand_max)

            # Build frame
            dst_frame = {
                "observation.state": torch.from_numpy(state_16d),
                "action": torch.from_numpy(action_16d),
            }

            # Copy camera images
            for cam in src_cameras:
                key = f"observation.images.{cam}"
                if key in src_frame:
                    dst_frame[key] = src_frame[key]

            # Task text
            if "task" in src_frame:
                dst_frame["task"] = src_frame["task"]
            else:
                dst_frame["task"] = "pick and place"

            dst_dataset.add_frame(dst_frame)

        dst_dataset.save_episode()
        frame_idx += ep_length

    print(f"\nConversion complete!")
    print(f"  Output: {dst_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Frames: {frame_idx}")
    print(f"  Dimensions: 28D → 16D")

    if push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dst_dataset.push_to_hub(upload_large_folder=True)
        print(f"  Uploaded to: {dst_repo_id}")

    return dst_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Dex3 28D dataset to Inspire 1DOF 16D")
    parser.add_argument("--src-repo-id", type=str, required=True,
                        help="Source Dex3 dataset repo ID (e.g., unitreerobotics/G1_Dex3_ToastedBread_Dataset)")
    parser.add_argument("--dst-repo-id", type=str, required=True,
                        help="Destination dataset repo ID")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize hand values to [0,1] range")
    parser.add_argument("--auto-range", action="store_true",
                        help="Auto-detect hand joint range from data (implies --normalize)")
    parser.add_argument("--hand-min", type=float, default=0.0,
                        help="Min hand joint value (default: 0.0)")
    parser.add_argument("--hand-max", type=float, default=1.0,
                        help="Max hand joint value (default: 1.0)")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Max episodes to convert (0 = all)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push converted dataset to HuggingFace Hub")

    args = parser.parse_args()

    convert_dex3_to_inspire_1dof(
        src_repo_id=args.src_repo_id,
        dst_repo_id=args.dst_repo_id,
        normalize=args.normalize,
        hand_min=args.hand_min,
        hand_max=args.hand_max,
        auto_range=args.auto_range,
        push_to_hub=args.push_to_hub,
        max_episodes=args.max_episodes,
    )
