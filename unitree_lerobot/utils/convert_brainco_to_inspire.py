"""Convert Brainco 26D LeRobot dataset to Inspire 26D format.

Reorders hand joints from Brainco ordering to Inspire ordering.
Arm joints (14D) are identical — pass through unchanged.
Optionally applies per-joint range normalization.

Brainco 26D layout:
  [0:7]   = Left arm  (same as Inspire)
  [7:14]  = Right arm (same as Inspire)
  [14:20] = Left hand  [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
  [20:26] = Right hand [Thumb, ThumbAux, Index, Middle, Ring, Pinky]

Inspire 26D layout:
  [0:7]   = Left arm
  [7:14]  = Right arm
  [14:20] = Left hand  [Pinky, Ring, Middle, Index, ThumbBend, ThumbRotation]
  [20:26] = Right hand [Pinky, Ring, Middle, Index, ThumbBend, ThumbRotation]

Usage:
  python unitree_lerobot/utils/convert_brainco_to_inspire.py \
      --src-repo-id unitreerobotics/G1_Brainco_PickApple_Dataset \
      --dst-repo-id jihun/G1_Inspire_PickApple_from_Brainco

  # With normalization (if scan_joint_ranges.py showed different ranges):
  python unitree_lerobot/utils/convert_brainco_to_inspire.py \
      --src-repo-id unitreerobotics/G1_Brainco_PickApple_Dataset \
      --dst-repo-id jihun/G1_Inspire_PickApple_from_Brainco \
      --range-file /path/to/joint_range_scan.json
"""
import argparse
import json
import shutil
import numpy as np
import torch
import tqdm
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.utils.constants import ROBOT_CONFIGS

# Slice definitions for 26D vectors
ARM_SLICE = slice(0, 14)
LEFT_HAND_SLICE = slice(14, 20)
RIGHT_HAND_SLICE = slice(20, 26)

# Brainco [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
# Inspire [Pinky, Ring, Middle, Index, ThumbBend, ThumbRotation]
# inspire_output[i] = brainco_input[BRAINCO_TO_INSPIRE_HAND[i]]
BRAINCO_TO_INSPIRE_HAND = [5, 4, 3, 2, 0, 1]


def normalize_joint(value: float, src_min: float, src_max: float,
                    dst_min: float = 0.0, dst_max: float = 1.0) -> float:
    """Linear interpolation from source range to destination range."""
    if src_max == src_min:
        return dst_min
    normalized = (value - src_min) / (src_max - src_min)
    return dst_min + normalized * (dst_max - dst_min)


def reorder_hand(brainco_hand: np.ndarray) -> np.ndarray:
    """Reorder a 6D Brainco hand vector to Inspire joint order."""
    return brainco_hand[BRAINCO_TO_INSPIRE_HAND].astype(np.float32)


def convert_26d_brainco_to_inspire(
    state_26d: np.ndarray,
    range_map: dict | None = None,
) -> np.ndarray:
    """Convert a single 26D Brainco state/action vector to Inspire format.

    Args:
        state_26d: Array of shape (26,)
        range_map: Optional dict with per-joint normalization ranges.
                   Keys: "left_hand" and "right_hand", each a list of 6 dicts
                   with {src_min, src_max, dst_min, dst_max}.
    Returns:
        Array of shape (26,) in Inspire joint order.
    """
    if state_26d.shape != (26,):
        raise ValueError(f"Expected shape (26,), got {state_26d.shape}")
    arm = state_26d[ARM_SLICE].copy()
    left_hand = reorder_hand(state_26d[LEFT_HAND_SLICE])
    right_hand = reorder_hand(state_26d[RIGHT_HAND_SLICE])

    if range_map:
        for i in range(6):
            if "left_hand" in range_map:
                rm = range_map["left_hand"][i]
                left_hand[i] = normalize_joint(
                    left_hand[i], rm["src_min"], rm["src_max"],
                    rm.get("dst_min", 0.0), rm.get("dst_max", 1.0))
            if "right_hand" in range_map:
                rm = range_map["right_hand"][i]
                right_hand[i] = normalize_joint(
                    right_hand[i], rm["src_min"], rm["src_max"],
                    rm.get("dst_min", 0.0), rm.get("dst_max", 1.0))

    return np.concatenate([arm, left_hand, right_hand]).astype(np.float32)


def convert_dataset(
    src_repo_id: str,
    dst_repo_id: str,
    *,
    range_file: str | None = None,
    camera_filter: str | None = None,
    max_episodes: int = 0,
    push_to_hub: bool = False,
):
    """Convert a full Brainco LeRobot dataset to Inspire format."""
    print(f"Loading source dataset: {src_repo_id}")
    src_dataset = LeRobotDataset(repo_id=src_repo_id)

    src_dim = src_dataset.meta.shapes["observation.state"][0]
    assert src_dim == 26, f"Expected 26D source, got {src_dim}D"

    total_episodes = src_dataset.meta.total_episodes
    total_frames = src_dataset.meta.total_frames
    print(f"Source: {total_episodes} episodes, {total_frames} frames, {src_dim}D")

    # Load range map if provided
    range_map = None
    if range_file:
        with open(range_file) as f:
            range_map = json.load(f)
        print(f"Using range normalization from {range_file}")

    # Determine cameras
    src_cameras = [key.replace("observation.images.", "")
                   for key in src_dataset.meta.shapes
                   if key.startswith("observation.images.")]
    if camera_filter:
        src_cameras = [c for c in src_cameras if c == camera_filter]
        if not src_cameras:
            print(f"WARNING: camera_filter='{camera_filter}' matched no cameras. "
                  f"Available: {[k.replace('observation.images.', '') for k in src_dataset.meta.shapes if k.startswith('observation.images.')]}")
    print(f"Cameras: {src_cameras}")

    # Build features
    inspire_motors = ROBOT_CONFIGS["Unitree_G1_Inspire"].motors
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (26,),
            "names": [inspire_motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (26,),
            "names": [inspire_motors],
        },
    }
    for cam in src_cameras:
        cam_shape = src_dataset.meta.shapes[f"observation.images.{cam}"]
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": cam_shape,
            "names": ["height", "width", "channel"],
        }

    # Create destination dataset
    dst_path = HF_LEROBOT_HOME / dst_repo_id
    if dst_path.exists():
        shutil.rmtree(dst_path)

    dst_dataset = LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=int(src_dataset.fps),
        robot_type="Unitree_G1_Inspire",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )

    num_episodes = min(max_episodes, total_episodes) if max_episodes > 0 else total_episodes
    print(f"Converting {num_episodes}/{total_episodes} episodes...")

    # Get episode boundaries (LeRobot v0.4.1 API)
    ep_from_indices = src_dataset.meta.episodes["dataset_from_index"]
    ep_to_indices = src_dataset.meta.episodes["dataset_to_index"]

    frame_count = 0
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Converting"):
        ep_from = ep_from_indices[ep_idx]
        ep_to = ep_to_indices[ep_idx]

        for i in range(ep_to - ep_from):
            src_frame = src_dataset[ep_from + i]

            state = convert_26d_brainco_to_inspire(
                src_frame["observation.state"].numpy(), range_map)
            action = convert_26d_brainco_to_inspire(
                src_frame["action"].numpy(), range_map)

            dst_frame = {
                "observation.state": torch.from_numpy(state),
                "action": torch.from_numpy(action),
            }
            for cam in src_cameras:
                key = f"observation.images.{cam}"
                if key in src_frame:
                    img = src_frame[key]
                    # LeRobot returns CHW; add_frame expects HWC
                    if img.dim() == 3 and img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0)
                    dst_frame[key] = img

            if "task" in src_frame:
                dst_frame["task"] = src_frame["task"]
            else:
                dst_frame["task"] = "pick and place"
            dst_dataset.add_frame(dst_frame)

        dst_dataset.save_episode()
        frame_count += ep_to - ep_from

    print(f"\nDone! {num_episodes} episodes, {frame_count} frames")
    print(f"Output: {dst_path}")

    if push_to_hub:
        dst_dataset.push_to_hub(upload_large_folder=True)

    return dst_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Brainco 26D -> Inspire 26D")
    parser.add_argument("--src-repo-id", required=True)
    parser.add_argument("--dst-repo-id", required=True)
    parser.add_argument("--range-file", type=str, default=None,
                        help="JSON file with per-joint normalization ranges")
    parser.add_argument("--camera-filter", type=str, default=None,
                        help="Keep only this camera (e.g., cam_left_high)")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    convert_dataset(
        src_repo_id=args.src_repo_id,
        dst_repo_id=args.dst_repo_id,
        range_file=args.range_file,
        camera_filter=args.camera_filter,
        max_episodes=args.max_episodes,
        push_to_hub=args.push_to_hub,
    )
