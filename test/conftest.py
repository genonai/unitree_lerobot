"""pytest configuration: stub out heavy lerobot dependencies for unit/integration tests."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so unitree_lerobot is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub lerobot modules before any test imports convert_unitree_json_to_lerobot
# ---------------------------------------------------------------------------
_lerobot = types.ModuleType("lerobot")
_lerobot_utils = types.ModuleType("lerobot.utils")
_lerobot_utils_constants = types.ModuleType("lerobot.utils.constants")
_lerobot_utils_constants.HF_LEROBOT_HOME = Path("/tmp/lerobot_home")
_lerobot_datasets = types.ModuleType("lerobot.datasets")
_lerobot_datasets_lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")
_lerobot_datasets_lerobot_dataset.LeRobotDataset = MagicMock()

sys.modules.setdefault("lerobot", _lerobot)
sys.modules.setdefault("lerobot.utils", _lerobot_utils)
sys.modules.setdefault("lerobot.utils.constants", _lerobot_utils_constants)
sys.modules.setdefault("lerobot.datasets", _lerobot_datasets)
sys.modules.setdefault("lerobot.datasets.lerobot_dataset", _lerobot_datasets_lerobot_dataset)

# ---------------------------------------------------------------------------
# Stub xr_teleoperate dependencies for E2E recording tests
# ---------------------------------------------------------------------------
_logging_mp = types.ModuleType("logging_mp")
_logging_mp.getLogger = lambda name=None: __import__("logging").getLogger(name)
sys.modules.setdefault("logging_mp", _logging_mp)

_rerun_sdk = types.ModuleType("rerun")
sys.modules.setdefault("rerun", _rerun_sdk)

# Stub the rerun_visualizer module that EpisodeWriter imports
_rerun_viz = types.ModuleType("teleop.utils.rerun_visualizer")
_rerun_viz.RerunLogger = MagicMock
sys.modules.setdefault("teleop.utils.rerun_visualizer", _rerun_viz)
