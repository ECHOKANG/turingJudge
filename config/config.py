import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000


def _load_models_config() -> dict:
    models_path = PROJECT_ROOT / "models.json"
    if models_path.exists():
        with open(models_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_MODELS = _load_models_config()

DEFAULT_MODEL = _MODELS.get("agent_default", "doubao-seed-2-0-mini-260428")
SIMULATOR_MODEL = _MODELS.get("simulator", "doubao-seed-character-251128")
JUDGE_MODEL = _MODELS.get("judge", "doubao-seed-2-0-mini-260428")
ALLOCATOR_MODEL = _MODELS.get("allocator", "doubao-seed-2-0-mini-260428")
REFINER_MODEL = _MODELS.get("refiner", "doubao-seed-2-0-mini-260428")
SCENARIO_GENERATOR_MODEL = _MODELS.get("scenario_generator", "doubao-seed-2-0-mini-260428")
