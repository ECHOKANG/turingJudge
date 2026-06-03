"""
统一处理 OpenAI / Doubao(Volcengine Ark) Chat Completions 返回的 usage 字段，
重点是提取「上下文缓存命中」信息并输出可观测日志。

火山方舟 / OpenAI 都遵循同一字段:
usage = {
  "prompt_tokens": int,
  "completion_tokens": int,
  "total_tokens": int,
  "prompt_tokens_details": {"cached_tokens": int},     # 缓存命中 token 数
}

参考: https://www.volcengine.com/docs/82379/1396491

【日志去向】
- 始终写 stderr
- 同时追加到 <PROJECT_ROOT>/output/llm_usage.log，方便事后排查
- 通过 USAGE_LOG_ENABLED=0 关闭全部日志
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


USAGE_LOG_ENABLED = os.getenv("USAGE_LOG_ENABLED", "1") not in ("0", "false", "False", "")

# 日志文件路径：默认放到 <project_root>/output/llm_usage.log
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_LOG_FILE = _PROJECT_ROOT / "output" / "llm_usage.log"
USAGE_LOG_FILE = os.getenv("USAGE_LOG_FILE") or str(_DEFAULT_LOG_FILE)


def _ensure_log_dir() -> None:
    try:
        Path(USAGE_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


_ensure_log_dir()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_usage(response: Any) -> Dict[str, Any]:
    """
    从 chat.completions.create 的返回里抽出标准化 usage dict，包含 cached_tokens
    与命中比。即使字段缺失也总能返回一个 dict（数值为 0）。
    """
    usage = _get(response, "usage")
    prompt_tokens = int(_get(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get(usage, "completion_tokens", 0) or 0)
    total_tokens = int(_get(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    details = _get(usage, "prompt_tokens_details") or {}
    cached_tokens = int(_get(details, "cached_tokens", 0) or 0)

    cache_hit_ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": round(cache_hit_ratio, 4),
    }


def format_usage_line(label: str, model: str, usage: Dict[str, Any]) -> str:
    return (
        f"[USAGE] {label} model={model} "
        f"prompt={usage['prompt_tokens']} "
        f"completion={usage['completion_tokens']} "
        f"total={usage['total_tokens']} "
        f"cached={usage['cached_tokens']} "
        f"cache_hit={usage['cache_hit_ratio'] * 100:.2f}%"
    )


def _emit(line: str) -> None:
    """把日志同时写到 stderr 和文件。"""
    if not USAGE_LOG_ENABLED:
        return
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} {line}\n")
    except Exception:
        pass


def log_usage(
    label: str,
    model: str,
    response: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    打印 usage 日志（含缓存命中），并返回标准化 usage dict 供调用方继续使用。
    任何异常都吞掉并返回空 dict，保证不影响主流程。
    """
    try:
        usage = extract_usage(response)
        line = format_usage_line(label, model, usage)
        if extra:
            line += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        _emit(line)
        return usage
    except Exception as e:
        _emit(f"[USAGE] {label} model={model} extract_failed err={e}")
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_hit_ratio": 0.0,
        }


def log_event(event_label: str, message: str, **kwargs: Any) -> None:
    """通用诊断日志：写 stderr + 文件，前缀 [EVENT]。"""
    extra = ""
    if kwargs:
        extra = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    _emit(f"[EVENT] {event_label} {message}{extra}")


def usage_to_langfuse(usage: Dict[str, Any]) -> Dict[str, int]:
    return {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
        "input_cached_tokens": usage.get("cached_tokens", 0),
    }
