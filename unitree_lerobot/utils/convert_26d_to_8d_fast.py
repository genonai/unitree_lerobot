"""Fast 26D→8D converter: rewrites parquet + metadata, symlinks videos.

~1-2 min instead of ~60 min by avoiding video re-encoding.

Usage:
    python convert_26d_to_8d_fast.py \
        --src-repo-id jihun/G1_Inspire_PickPlace_Unified_246ep_fullres \
        --dst-repo-id jihun/G1_Inspire_PickPlace_RightArm8D_246ep_fullres
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_RIGHT_ARM = slice(7, 14)   # 7 dims
SRC_RIGHT_HAND = slice(20, 26) # 6 dims → mean → 1 dim

MOTORS_8D = [
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kRightGrip",
]


def slice_26d_to_8d(values: list[list[float]]) -> list[list[float]]:
    """Vectorized 26D→8D: right_arm(7) + mean(right_hand(6))."""
    arr = np.array(values, dtype=np.float32)
    right_arm = arr[:, SRC_RIGHT_ARM]                    # (N, 7)
    right_grip = arr[:, SRC_RIGHT_HAND].mean(axis=1, keepdims=True)  # (N, 1)
    return np.concatenate([right_arm, right_grip], axis=1).tolist()


def rewrite_parquet(src_path: Path, dst_path: Path):
    """Read parquet, slice state/action to 8D, write new parquet."""
    table = pq.read_table(str(src_path))

    state_col = table.column("observation.state").to_pylist()
    action_col = table.column("action").to_pylist()

    state_8d = slice_26d_to_8d(state_col)
    action_8d = slice_26d_to_8d(action_col)

    # Replace columns
    idx_state = table.schema.get_field_index("observation.state")
    idx_action = table.schema.get_field_index("action")

    table = table.set_column(idx_state, "observation.state",
                             pa.array(state_8d, type=pa.list_(pa.float32())))
    table = table.set_column(idx_action, "action",
                             pa.array(action_8d, type=pa.list_(pa.float32())))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(dst_path))


def rewrite_info(src_info: dict) -> dict:
    """Update info.json for 8D."""
    info = json.loads(json.dumps(src_info))  # deep copy
    info["robot_type"] = "Unitree_G1_Inspire_RightArm8D_Mono"

    for key in ("observation.state", "action"):
        info["features"][key]["shape"] = [8]
        info["features"][key]["names"] = [MOTORS_8D]

    return info


def recompute_stats(dst_data_dir: Path, src_stats: dict) -> dict:
    """Recompute stats for 8D state/action from parquet files."""
    # Collect all 8D data
    all_states = []
    all_actions = []
    for pq_file in sorted(dst_data_dir.rglob("*.parquet")):
        table = pq.read_table(str(pq_file))
        all_states.extend(table.column("observation.state").to_pylist())
        all_actions.extend(table.column("action").to_pylist())

    stats = json.loads(json.dumps(src_stats))  # deep copy

    for key, values in [("observation.state", all_states), ("action", all_actions)]:
        arr = np.array(values, dtype=np.float32)
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [len(arr)] * 8,
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fast 26D→8D converter (no video re-encode)")
    parser.add_argument("--src-repo-id", required=True)
    parser.add_argument("--dst-repo-id", required=True)
    args = parser.parse_args()

    hf_home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    src_root = hf_home / args.src_repo_id
    dst_root = hf_home / args.dst_repo_id

    assert src_root.exists(), f"Source not found: {src_root}"

    # Clean destination
    if dst_root.exists():
        print(f"Removing existing: {dst_root}")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True)
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")

    # 1. Symlink videos (no re-encoding)
    src_videos = src_root / "videos"
    dst_videos = dst_root / "videos"
    if src_videos.exists():
        os.symlink(src_videos.resolve(), dst_videos)
        print(f"Symlinked videos/")

    # 2. Rewrite parquet files
    src_data = src_root / "data"
    dst_data = dst_root / "data"
    pq_files = sorted(src_data.rglob("*.parquet"))
    print(f"Rewriting {len(pq_files)} parquet file(s)...")
    total_rows = 0
    for pq_file in pq_files:
        rel = pq_file.relative_to(src_data)
        dst_pq = dst_data / rel
        rewrite_parquet(pq_file, dst_pq)
        n = pq.read_metadata(str(dst_pq)).num_rows
        total_rows += n
        print(f"  {rel}: {n} rows → 8D")

    # 3. Rewrite metadata
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True)

    # info.json
    with open(src_root / "meta/info.json") as f:
        src_info = json.load(f)
    dst_info = rewrite_info(src_info)
    with open(dst_meta / "info.json", "w") as f:
        json.dump(dst_info, f, indent=4)
    print(f"Wrote info.json (robot_type={dst_info['robot_type']})")

    # stats.json
    with open(src_root / "meta/stats.json") as f:
        src_stats = json.load(f)
    dst_stats = recompute_stats(dst_data, src_stats)
    with open(dst_meta / "stats.json", "w") as f:
        json.dump(dst_stats, f, indent=4)
    print("Recomputed stats.json for 8D")

    # Copy tasks.parquet and episodes/ as-is
    shutil.copy2(src_root / "meta/tasks.parquet", dst_meta / "tasks.parquet")
    src_episodes = src_root / "meta/episodes"
    if src_episodes.exists():
        shutil.copytree(src_episodes, dst_meta / "episodes")
    print("Copied tasks.parquet + episodes/")

    print(f"\nDone! {dst_info['total_episodes']} episodes, {total_rows} frames → 8D")
    print(f"Output: {dst_root}")


if __name__ == "__main__":
    main()
