"""Convert 26D Inspire dataset to 8D RightArm+Grip dataset.

Extracts right arm (dims 7:14) + mean grip signal (mean of dims 20:26)
from existing 26D LeRobot dataset. Videos are re-linked (not re-encoded).

Usage:
    python convert_26d_to_8d.py \
        --src-repo-id jihun/G1_Inspire_PickPlace_Unified_246ep_fullres \
        --dst-repo-id jihun/G1_Inspire_PickPlace_RightArm8D_246ep_fullres \
        --camera cam_head
"""
import argparse
import shutil

import numpy as np
import torch
import tqdm

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


def convert_26d_to_8d(vec_26d: np.ndarray) -> np.ndarray:
    """Slice 26D → 8D: right_arm(7) + mean(right_hand(6))."""
    right_arm = vec_26d[7:14]  # 7D
    right_hand_grip = np.mean(vec_26d[20:26])  # scalar
    return np.concatenate([right_arm, [right_hand_grip]]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert 26D LeRobot dataset to 8D")
    parser.add_argument("--src-repo-id", required=True, help="Source 26D dataset repo_id")
    parser.add_argument("--dst-repo-id", required=True, help="Destination 8D dataset repo_id")
    parser.add_argument("--camera", default="cam_head", help="Camera key to include (default: cam_head)")
    parser.add_argument("--max-episodes", type=int, default=0, help="Max episodes to convert (0=all)")
    args = parser.parse_args()

    print(f"Loading source: {args.src_repo_id}")
    src = LeRobotDataset(repo_id=args.src_repo_id)

    src_dim = src.meta.shapes["observation.state"][0]
    assert src_dim == 26, f"Expected 26D source, got {src_dim}D"

    total_episodes = src.meta.total_episodes
    total_frames = src.meta.total_frames
    print(f"Source: {total_episodes} episodes, {total_frames} frames, {src_dim}D")

    # Camera shape from source
    cam_key = f"observation.images.{args.camera}"
    assert cam_key in src.meta.shapes, (
        f"Camera '{args.camera}' not found. Available: "
        f"{[k.replace('observation.images.', '') for k in src.meta.shapes if k.startswith('observation.images.')]}"
    )
    cam_shape = src.meta.shapes[cam_key]
    print(f"Camera: {args.camera}, shape: {cam_shape}")

    # Build 8D features
    motors_8d = ROBOT_CONFIGS["Unitree_G1_Inspire_RightArm8D_Mono"].motors
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [motors_8d],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [motors_8d],
        },
        f"observation.images.{args.camera}": {
            "dtype": "video",
            "shape": cam_shape,
            "names": ["height", "width", "channel"],
        },
    }

    # Remove existing destination
    dst_path = HF_LEROBOT_HOME / args.dst_repo_id
    if dst_path.exists():
        print(f"Removing existing: {dst_path}")
        shutil.rmtree(dst_path)

    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=int(src.fps),
        robot_type="Unitree_G1_Inspire_RightArm8D_Mono",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )

    num_episodes = min(args.max_episodes, total_episodes) if args.max_episodes > 0 else total_episodes
    ep_from_indices = src.meta.episodes["dataset_from_index"]
    ep_to_indices = src.meta.episodes["dataset_to_index"]

    frame_count = 0
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Converting"):
        ep_from = ep_from_indices[ep_idx]
        ep_to = ep_to_indices[ep_idx]

        for i in range(ep_to - ep_from):
            src_frame = src[ep_from + i]

            state_8d = convert_26d_to_8d(src_frame["observation.state"].numpy())
            action_8d = convert_26d_to_8d(src_frame["action"].numpy())

            dst_frame = {
                "observation.state": torch.from_numpy(state_8d),
                "action": torch.from_numpy(action_8d),
            }

            # Copy camera image
            if cam_key in src_frame:
                img = src_frame[cam_key]
                # LeRobot returns CHW; add_frame expects HWC
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                dst_frame[cam_key] = img

            # Copy task
            if "task" in src_frame:
                dst_frame["task"] = src_frame["task"]
            else:
                dst_frame["task"] = "pick up pill bottle from table and place on red taped square"

            dst.add_frame(dst_frame)

        dst.save_episode()
        frame_count += ep_to - ep_from

    print(f"\nDone! {num_episodes} episodes, {frame_count} frames → 8D")
    print(f"Output: {dst_path}")


if __name__ == "__main__":
    main()
