"""Tests for finalize_lerobot_dataset — session-end staging consolidator.

The existing test/conftest.py stubs LeRobotDataset with a MagicMock, so the
focus here is on the parts of the consolidator that are load-bearing for
correctness: staging discovery, per-episode parquet/MP4 decoding, and the
shape of the payload handed to LeRobotDataset.add_frame.

Writes synthetic staging artifacts that match the on-disk contract emitted
by LeRobotEpisodeWriter (parquet schema + ffmpeg H.264 MP4 + meta.json).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unitree_lerobot.utils.finalize_lerobot_dataset import (
    ACTION_DIM,
    CAM_KEY,
    STATE_DIM,
    Args,
    build_features,
    consolidate,
    discover_episodes,
    iter_mp4_frames,
    iter_parquet_rows,
    load_episode_meta,
)


# ---------------------------------------------------------------------------
# synthetic staging artifacts matching LeRobotEpisodeWriter's output
# ---------------------------------------------------------------------------


def _write_synthetic_parquet(path: Path, n_frames: int, episode_id: int) -> None:
    state_rows = [[float(episode_id * 100 + i + k * 0.01) for k in range(STATE_DIM)]
                  for i in range(n_frames)]
    action_rows = [[float(episode_id * 200 + i + k * 0.01) for k in range(ACTION_DIM)]
                   for i in range(n_frames)]
    table = pa.table({
        "observation.state": pa.array(state_rows, type=pa.list_(pa.float32(), STATE_DIM)),
        "action":            pa.array(action_rows, type=pa.list_(pa.float32(), ACTION_DIM)),
        "timestamp":         pa.array([i / 30.0 for i in range(n_frames)], type=pa.float32()),
        "frame_index":       pa.array(list(range(n_frames)), type=pa.int64()),
        "episode_index":     pa.array([episode_id] * n_frames, type=pa.int64()),
    })
    pq.write_table(table, path)


def _write_synthetic_mp4(path: Path, n_frames: int, W: int = 640, H: int = 480) -> None:
    """Match the writer's ffmpeg invocation: rawvideo bgr24 -> libx264 yuv420p -g 10."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", "30", "-i", "pipe:",
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-g", "10", "-an",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    # Deterministic per-frame content so decode roundtrip is checkable.
    for i in range(n_frames):
        frame = np.full((H, W, 3), fill_value=(i * 7) % 256, dtype=np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait(timeout=15)
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}: {proc.stderr.read().decode(errors='replace')}")


def _write_synthetic_meta(path: Path, episode_id: int, n_frames: int, task: str) -> None:
    meta = {
        "episode_id": episode_id,
        "n_frames": n_frames,
        "fps": 30,
        "image_size": [640, 480],
        "robot_type": "Unitree_G1_Inspire_HeadOnly_Mono",
        "task": {"goal": task, "desc": "", "steps": ""},
        "created_utc": "2026-04-22T00:00:00Z",
        "writer_version": "1.0.0",
    }
    path.write_text(json.dumps(meta))


@pytest.fixture
def fabricate_staging(tmp_path: Path):
    """Factory that builds a staging dir with the given episode specs."""

    def _build(episodes: list[tuple[int, int, str]]) -> Path:
        """episodes: list of (episode_id, n_frames, task_goal)."""
        staging = tmp_path / "task" / "_staging"
        staging.mkdir(parents=True)
        for ep_id, n, task in episodes:
            ep_dir = staging / f"episode_{ep_id:06d}"
            ep_dir.mkdir()
            _write_synthetic_parquet(ep_dir / "data.parquet", n, ep_id)
            _write_synthetic_mp4(ep_dir / "cam_head.mp4", n)
            _write_synthetic_meta(ep_dir / "meta.json", ep_id, n, task)
        return staging

    return _build


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_episodes_sorts_by_numeric_id(fabricate_staging):
    staging = fabricate_staging([(2, 3, "t"), (0, 3, "t"), (1, 3, "t")])
    dirs = discover_episodes(staging)
    assert [d.name for d in dirs] == ["episode_000000", "episode_000001", "episode_000002"]


def test_discover_episodes_skips_tmp_dirs(fabricate_staging):
    staging = fabricate_staging([(0, 3, "t")])
    (staging / "episode_000001.tmp").mkdir()  # in-flight episode
    dirs = discover_episodes(staging)
    assert [d.name for d in dirs] == ["episode_000000"]


def test_discover_episodes_skips_incomplete_dirs(fabricate_staging, tmp_path, capsys):
    staging = fabricate_staging([(0, 3, "t")])
    # A dir with only meta.json — missing parquet + mp4.
    bad = staging / "episode_000001"
    bad.mkdir()
    _write_synthetic_meta(bad / "meta.json", 1, 3, "t")
    dirs = discover_episodes(staging)
    assert [d.name for d in dirs] == ["episode_000000"]
    assert "skipping incomplete episode dir" in capsys.readouterr().err


def test_discover_episodes_raises_on_missing_staging(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_episodes(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# per-episode loaders
# ---------------------------------------------------------------------------


def test_iter_parquet_rows_yields_expected_schema(fabricate_staging):
    staging = fabricate_staging([(0, 5, "t")])
    rows = list(iter_parquet_rows(staging / "episode_000000" / "data.parquet"))
    assert len(rows) == 5
    for i, row in enumerate(rows):
        assert row["observation.state"].shape == (STATE_DIM,)
        assert row["action"].shape == (ACTION_DIM,)
        assert row["observation.state"].dtype == np.float32
        assert row["frame_index"] == i
        assert row["timestamp"] == pytest.approx(i / 30.0, abs=1e-5)


def test_iter_mp4_frames_yields_correct_count_and_shape(fabricate_staging):
    staging = fabricate_staging([(0, 7, "t")])
    frames = list(iter_mp4_frames(staging / "episode_000000" / "cam_head.mp4"))
    assert len(frames) == 7
    for f in frames:
        assert f.shape == (480, 640, 3)
        assert f.dtype == np.uint8


def test_load_episode_meta_roundtrip(fabricate_staging):
    staging = fabricate_staging([(3, 10, "pick up cube")])
    meta = load_episode_meta(staging / "episode_000003")
    assert meta["episode_id"] == 3
    assert meta["n_frames"] == 10
    assert meta["task"]["goal"] == "pick up cube"
    assert meta["image_size"] == [640, 480]
    assert meta["robot_type"] == "Unitree_G1_Inspire_HeadOnly_Mono"


# ---------------------------------------------------------------------------
# feature spec
# ---------------------------------------------------------------------------


def test_build_features_matches_robot_config():
    features = build_features("Unitree_G1_Inspire_HeadOnly_Mono", (640, 480))
    assert set(features.keys()) == {"observation.state", "action", CAM_KEY}
    assert features["observation.state"]["shape"] == (STATE_DIM,)
    assert features["action"]["shape"] == (ACTION_DIM,)
    assert len(features["observation.state"]["names"]) == STATE_DIM
    # Camera feature: video dtype, (H, W, 3) shape per LeRobot convention.
    assert features[CAM_KEY]["dtype"] == "video"
    assert features[CAM_KEY]["shape"] == (480, 640, 3)


def test_build_features_rejects_wrong_motor_count():
    # Unitree_G1_Inspire_1DOF has 16 motors, not 26 — consolidator must refuse.
    with pytest.raises(ValueError, match="expected 26"):
        build_features("Unitree_G1_Inspire_1DOF", (640, 480))


# ---------------------------------------------------------------------------
# end-to-end consolidate() with a mock LeRobotDataset
# ---------------------------------------------------------------------------


def _make_mock_ds(root: Path):
    """Mimic the minimal LeRobotDataset surface consolidate() touches."""
    ds = MagicMock()
    ds.root = root
    ds.add_frame = MagicMock()
    ds.save_episode = MagicMock()
    return ds


def test_consolidate_calls_add_frame_with_lockstep_rows_and_frames(
    tmp_path, fabricate_staging
):
    staging = fabricate_staging([(0, 5, "pick"), (1, 3, "place")])
    mock_ds = _make_mock_ds(root=tmp_path / "out")

    with patch(
        "unitree_lerobot.utils.finalize_lerobot_dataset.LeRobotDataset.create",
        return_value=mock_ds,
    ):
        rc = consolidate(Args(
            task_dir=tmp_path / "task",
            repo_id="test/fake",
            output_root=tmp_path / "out",
            fps=30,
            mark_consumed=False,
        ))

    assert rc == 0
    # 5 + 3 = 8 frames total, so add_frame called 8 times.
    assert mock_ds.add_frame.call_count == 8
    # save_episode called once per episode.
    assert mock_ds.save_episode.call_count == 2

    # Inspect first add_frame call — should carry the right keys and shapes.
    first_call = mock_ds.add_frame.call_args_list[0]
    frame_dict = first_call.args[0]
    assert set(frame_dict.keys()) == {
        "observation.state", "action", CAM_KEY, "task", "timestamp",
    }
    assert frame_dict["observation.state"].shape == (STATE_DIM,)
    assert frame_dict["action"].shape == (ACTION_DIM,)
    assert frame_dict[CAM_KEY].shape == (480, 640, 3)
    assert frame_dict["task"] == "pick"
    assert frame_dict["timestamp"] == pytest.approx(0.0, abs=1e-6)

    # First frame of episode 1 carries the second task string.
    ep1_first_call = mock_ds.add_frame.call_args_list[5]  # index after 5-frame ep0
    assert ep1_first_call.args[0]["task"] == "place"


def test_consolidate_refuses_fps_mismatch(tmp_path, fabricate_staging):
    staging = fabricate_staging([(0, 3, "t")])
    with patch(
        "unitree_lerobot.utils.finalize_lerobot_dataset.LeRobotDataset.create",
    ):
        with pytest.raises(ValueError, match="fps"):
            consolidate(Args(
                task_dir=tmp_path / "task",
                repo_id="test/fake",
                output_root=tmp_path / "out",
                fps=60,  # meta.json says 30
                mark_consumed=False,
            ))


def test_consolidate_refuses_robot_type_mismatch(tmp_path, fabricate_staging):
    staging = fabricate_staging([(0, 3, "t")])
    with patch(
        "unitree_lerobot.utils.finalize_lerobot_dataset.LeRobotDataset.create",
    ):
        with pytest.raises(ValueError, match="robot_type"):
            consolidate(Args(
                task_dir=tmp_path / "task",
                repo_id="test/fake",
                output_root=tmp_path / "out",
                fps=30,
                robot_type="Unitree_G1_Inspire_1DOF",
                mark_consumed=False,
            ))


def test_consolidate_task_override_replaces_per_episode_task(
    tmp_path, fabricate_staging
):
    staging = fabricate_staging([(0, 3, "original"), (1, 3, "different")])
    mock_ds = _make_mock_ds(root=tmp_path / "out")
    with patch(
        "unitree_lerobot.utils.finalize_lerobot_dataset.LeRobotDataset.create",
        return_value=mock_ds,
    ):
        rc = consolidate(Args(
            task_dir=tmp_path / "task",
            repo_id="test/fake",
            output_root=tmp_path / "out",
            fps=30,
            task_override="unified task",
            mark_consumed=False,
        ))
    assert rc == 0
    tasks = {c.args[0]["task"] for c in mock_ds.add_frame.call_args_list}
    assert tasks == {"unified task"}


def test_consolidate_marks_staging_consumed_on_success(tmp_path, fabricate_staging):
    staging = fabricate_staging([(0, 3, "t")])
    mock_ds = _make_mock_ds(root=tmp_path / "out")
    with patch(
        "unitree_lerobot.utils.finalize_lerobot_dataset.LeRobotDataset.create",
        return_value=mock_ds,
    ):
        rc = consolidate(Args(
            task_dir=tmp_path / "task",
            repo_id="test/fake",
            output_root=tmp_path / "out",
            fps=30,
            mark_consumed=True,
        ))
    assert rc == 0
    assert not staging.exists()
    consumed = list((tmp_path / "task").glob("_staging.consumed.*"))
    assert len(consumed) == 1, f"expected one consumed dir, got: {consumed}"


def test_consolidate_returns_1_when_no_episodes(tmp_path):
    (tmp_path / "task" / "_staging").mkdir(parents=True)
    rc = consolidate(Args(
        task_dir=tmp_path / "task",
        repo_id="test/fake",
        output_root=tmp_path / "out",
        fps=30,
        mark_consumed=False,
    ))
    assert rc == 1
