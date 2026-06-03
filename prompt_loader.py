import json
from pathlib import Path
from typing import Any, Dict

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_cache: Dict[str, Any] = {}


def load_prompt(name: str) -> Dict[str, Any]:
    path = _PROMPTS_DIR / f"{name}.json"
    if name not in _cache:
        with open(path, "r", encoding="utf-8") as f:
            _cache[name] = json.load(f)
    return _cache[name]


def load_md_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if name not in _cache:
        with open(path, "r", encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]


def clear_cache() -> None:
    _cache.clear()
