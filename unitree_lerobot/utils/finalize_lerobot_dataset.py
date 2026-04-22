"""finalize_lerobot_dataset — session-end consolidator for LeRobotEpisodeWriter.

Repacks per-episode staging artifacts emitted by
``xr_teleoperate/teleop/utils/lerobot_episode_writer.py`` into the LeRobotDataset
v3 chunked layout. Invoked after a teleop session (manually or via the
atexit hook in ``teleop_hand_and_arm.py``).

Input layout (written by LeRobotEpisodeWriter):

    <task_dir>/_staging/
      episode_000000/
        data.parquet       # columns: observation.state, action, timestamp,
                           #          frame_index, episode_index
        cam_head.mp4       # 640x480 BGR H.264, -g 10, yuv420p
        meta.json          # {episode_id, n_frames, fps, task.goal, ...}
      episode_000001/
      ...
      episode_NNNNNN.tmp/  # in-flight — skipped

Output: standard LeRobotDataset v3 tree at ``HF_LEROBOT_HOME/<repo_id>`` or
the explicit ``--output-root`` path. Compatible with ``lerobot-train`` for
ACT and Pi0/Pi0.5.

Design: docs/superpowers/specs/2026-04-21-direct-lerobot-recording-design.md §3.3
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pyarrow.parquet as pq
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS

# Schema locked by the spec — see §2.
DEFAULT_ROBOT_TYPE = "Unitree_G1_Inspire_HeadOnly_Mono"
CAM_KEY = "observation.images.cam_head"
STATE_DIM = 26
ACTION_DIM = 26


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Args:
    task_dir: Path
    """Directory containing _staging/episode_NNNNNN/... subdirs (the xr_teleoperate --task-dir)."""

    repo_id: str
    """HuggingFace-style repo id for the resulting dataset, e.g. 'jihun/pick_hold_v04'."""

    robot_type: str = DEFAULT_ROBOT_TYPE
    """Robot config name in ROBOT_CONFIGS; must match the writer's schema."""

    output_root: Path | None = None
    """Explicit dataset root; defaults to HF_LEROBOT_HOME/<repo_id>."""

    fps: int = 30
    """Dataset fps tag; must match the recorded MP4 framerate."""

    task_override: str | None = None
    """If set, overrides per-episode meta.json task.goal (for batch re-labeling)."""

    mark_consumed: bool = True
    """Rename _staging/ -> _staging.consumed.YYYYMMDD-HHMMSS/ on success."""

    compute_quantiles: bool = False
    """Run augment_dataset_quantile_stats post-build (required for Pi0.5)."""

    keep_on_error: bool = True
    """If consolidation fails mid-way, leave the partial dataset on disk for diagnosis."""


# ---------------------------------------------------------------------------
# Episode discovery + loading
# ---------------------------------------------------------------------------


def discover_episodes(staging_dir: Path) -> list[Path]:
    """Return completed episode dirs sorted by numeric ID. Skips .tmp/ and junk."""
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"staging dir not found: {staging_dir}")
    out: list[tuple[int, Path]] = []
    for p in staging_dir.iterdir():
        if not p.is_dir() or p.name.endswith(".tmp") or not p.name.startswith("episode_"):
            continue
        try:
            n = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if not _episode_is_complete(p):
            print(f"  [WARN] skipping incomplete episode dir: {p.name}", file=sys.stderr)
            continue
        out.append((n, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _episode_is_complete(ep_dir: Path) -> bool:
    return (
        (ep_dir / "data.parquet").is_file()
        and (ep_dir / "cam_head.mp4").is_file()
        and (ep_dir / "meta.json").is_file()
    )


def load_episode_meta(ep_dir: Path) -> dict:
    return json.loads((ep_dir / "meta.json").read_text())


def iter_parquet_rows(parquet_path: Path) -> Iterator[dict]:
    """Yield per-frame dicts with fixed-size list<float32> state/action already unpacked."""
    table = pq.read_table(parquet_path)
    needed = {"observation.state", "action", "timestamp", "frame_index"}
    missing = needed - set(table.column_names)
    if missing:
        raise ValueError(f"{parquet_path}: missing columns {missing}")
    state_col = table.column("observation.state").to_pylist()
    action_col = table.column("action").to_pylist()
    ts_col = table.column("timestamp").to_pylist()
    frame_idx_col = table.column("frame_index").to_pylist()
    for i in range(table.num_rows):
        yield {
            "observation.state": np.asarray(state_col[i], dtype=np.float32),
            "action": np.asarray(action_col[i], dtype=np.float32),
            "timestamp": float(ts_col[i]),
            "frame_index": int(frame_idx_col[i]),
        }


def iter_mp4_frames(mp4_path: Path) -> Iterator[np.ndarray]:
    """Yield RGB frames from an MP4 (cv2 decodes BGR, convert to RGB)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {mp4_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Feature spec
# ---------------------------------------------------------------------------


def build_features(robot_type: str, image_size: tuple[int, int]) -> dict:
    """Build the LeRobotDataset features dict from ROBOT_CONFIGS."""
    cfg = ROBOT_CONFIGS[robot_type]
    motors = cfg.motors
    if len(motors) != STATE_DIM:
        raise ValueError(
            f"{robot_type} has {len(motors)} motors; expected {STATE_DIM}"
        )
    W, H = image_size
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": motors,
        },
        CAM_KEY: {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
    }


# ---------------------------------------------------------------------------
# Core consolidation
# ---------------------------------------------------------------------------


def consolidate(args: Args) -> int:
    staging_dir = args.task_dir / "_staging"
    episode_dirs = discover_episodes(staging_dir)
    if not episode_dirs:
        print(f"[finalize] no completed episodes under {staging_dir}", file=sys.stderr)
        return 1

    # Infer image size from the first meta.json (writer always records it).
    first_meta = load_episode_meta(episode_dirs[0])
    W, H = first_meta["image_size"]  # (W, H) as written by LeRobotEpisodeWriter
    if first_meta["fps"] != args.fps:
        raise ValueError(
            f"meta.json fps={first_meta['fps']} != --fps={args.fps}; "
            "refusing to mint a dataset with mismatched framerates"
        )
    if first_meta["robot_type"] != args.robot_type:
        raise ValueError(
            f"meta.json robot_type={first_meta['robot_type']!r} != "
            f"--robot-type={args.robot_type!r}"
        )

    features = build_features(args.robot_type, (W, H))

    root = args.output_root
    print(
        f"[finalize] {len(episode_dirs)} episodes from {staging_dir}\n"
        f"           -> repo_id={args.repo_id}, root={root or '<HF_LEROBOT_HOME>'}"
    )

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type=args.robot_type,
        root=root,
        use_videos=True,
    )

    total_frames = 0
    for i, ep_dir in enumerate(episode_dirs):
        meta = load_episode_meta(ep_dir)
        task = args.task_override if args.task_override is not None else meta["task"]["goal"]
        if not task:
            raise ValueError(f"episode {ep_dir.name}: empty task string")

        n_frames = _add_episode_frames(ds, ep_dir, task=task, expected_fps=args.fps)
        if n_frames != meta["n_frames"]:
            raise ValueError(
                f"{ep_dir.name}: parquet/MP4 yielded {n_frames} frames but meta says {meta['n_frames']}"
            )
        ds.save_episode()
        total_frames += n_frames
        print(f"  [{i+1}/{len(episode_dirs)}] {ep_dir.name}: {n_frames} frames, task={task!r}")

    print(f"[finalize] wrote {len(episode_dirs)} episodes, {total_frames} frames total")

    if args.compute_quantiles:
        _run_quantile_stats(ds.root)

    if args.mark_consumed:
        _mark_staging_consumed(staging_dir)

    return 0


def _add_episode_frames(
    ds: LeRobotDataset, ep_dir: Path, *, task: str, expected_fps: int
) -> int:
    """Stream parquet rows and MP4 frames in lockstep; each row MUST pair with one frame."""
    n = 0
    rows = iter_parquet_rows(ep_dir / "data.parquet")
    frames = iter_mp4_frames(ep_dir / "cam_head.mp4")
    for row, frame in zip(rows, frames):
        if row["frame_index"] != n:
            raise ValueError(
                f"{ep_dir.name}: non-monotonic frame_index at position {n}: "
                f"got {row['frame_index']}"
            )
        # LeRobotDataset.add_frame expects task + timestamp inside the dict;
        # it pops them before running feature validation (see
        # lerobot/datasets/lerobot_dataset.py:1128-1132).
        ds.add_frame(
            {
                "observation.state": row["observation.state"],
                "action": row["action"],
                CAM_KEY: frame,
                "task": task,
                "timestamp": row["timestamp"],
            }
        )
        n += 1

    # Verify both iterators exhausted together — mismatched lengths indicate
    # writer-side temporal desync we should never trust silently.
    extra_row = next(rows, None)
    extra_frame = next(frames, None)
    if extra_row is not None or extra_frame is not None:
        raise ValueError(
            f"{ep_dir.name}: parquet/MP4 length mismatch "
            f"(extra_row={extra_row is not None}, extra_frame={extra_frame is not None})"
        )
    return n


def _run_quantile_stats(dataset_root: Path) -> None:
    """Invoke augment_dataset_quantile_stats (required for Pi0.5 normalization)."""
    try:
        from lerobot.datasets.v30.augment_dataset_quantile_stats import (
            augment_dataset_quantile_stats,
        )
    except ImportError as e:
        print(
            f"[finalize] --compute-quantiles requested but import failed: {e}",
            file=sys.stderr,
        )
        return
    print(f"[finalize] computing quantile stats at {dataset_root}")
    augment_dataset_quantile_stats(dataset_root)


def _mark_staging_consumed(staging_dir: Path) -> None:
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    consumed = staging_dir.parent / f"_staging.consumed.{ts}"
    staging_dir.rename(consumed)
    print(f"[finalize] moved staging -> {consumed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = tyro.cli(Args)
    sys.exit(consolidate(args))


if __name__ == "__main__":
    main()
