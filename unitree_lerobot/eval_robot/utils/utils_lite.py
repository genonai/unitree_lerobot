"""Lightweight subset of utils.py for client-side eval scripts.

eval_remote.py runs on MacBook and orchestrates: image client, DDS arm/hand
control, ZMQ inference round-trip. It must NOT import lerobot (no draccus, no
torch policies, no PreTrainedConfig) — those live on the H100 inference server.

This module gives eval_remote.py the two symbols it needs (`EvalRealConfig` for
make_robot.setup_robot_interface, and `cleanup_resources` for shared-memory
teardown) without pulling in the lerobot stack.
"""
from dataclasses import dataclass, field
from typing import Any, Dict


def cleanup_resources(image_info: Dict[str, Any]) -> None:
    """Close + unlink any shared-memory blocks listed under 'shm_resources'.

    eval_remote.py builds its own image_info dict (mono camera, custom keys)
    and runs its own cleanup loop in the finally block, so this function is
    here only to satisfy the import. It tolerates missing keys.
    """
    for shm in image_info.get("shm_resources", []) or []:
        if shm is None:
            continue
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


@dataclass
class EvalRealConfig:
    """Minimal config consumed by make_robot.setup_robot_interface().

    Strips the policy field + lerobot-aware __post_init__ from the heavy
    EvalRealConfig in utils.py. eval_remote.py never loads a local policy
    (inference is remote), so no draccus/PreTrainedConfig is needed.
    """
    repo_id: str = ""
    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Robot config
    arm: str = "G1_29"   # G1_29, G1_23
    ee: str = "dex3"     # dex3, dex1, inspire1, inspire1_1dof, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False
    sim: bool = False

    rename_map: Dict[str, str] = field(default_factory=dict)
