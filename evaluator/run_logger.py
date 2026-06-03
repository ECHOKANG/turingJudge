"""run_logger.py — 评估执行日志收集与持久化。

输出结构: output/run_logs/<run_id>.json

每个 run 包含:
- run_id, started_at, finished_at, duration
- agent_name, total_n, max_turns, case_mode, allocation_mode
- module_timings: {模块名: 累计耗时}
- case_logs: [
    {
        "index", "case_id", "duration",
        "timings": {"agent_call_total": float, "simulator_call_total": float,
                    "judge_call_total": float, "turn_count": int, ...},
        "rule_flags": [...],
        "overall_score": float,
        "error": str or null
    }
]
"""
import json
import time
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

from config import OUTPUT_DIR

RUN_LOGS_DIR = OUTPUT_DIR / "run_logs"
RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)


class RunLogger:
    """单次批量评估的日志收集器，并发安全。"""

    def __init__(self, run_meta: Optional[Dict[str, Any]] = None):
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.time()
        self.meta = run_meta or {}
        self.case_logs: List[Dict[str, Any]] = []
        self.stage_timings: Dict[str, float] = {}  # 阶段耗时（如 allocate / dispatch / orchestrate / report）
        self._lock = Lock()

    def stage(self, name: str, duration: float):
        """记录阶段耗时（如配比、扩写、报告等）。"""
        with self._lock:
            self.stage_timings[name] = self.stage_timings.get(name, 0.0) + duration

    def add_case(self, case_log: Dict[str, Any]):
        """添加单 case 执行日志。"""
        with self._lock:
            self.case_logs.append(case_log)

    def finalize(self) -> Dict[str, Any]:
        """完成评估后落盘并返回完整 log dict。"""
        finished_at = datetime.now(timezone.utc).isoformat()
        duration = round(time.time() - self._t0, 3)

        # 聚合模块耗时
        module_totals: Dict[str, float] = {}
        for cl in self.case_logs:
            for k, v in (cl.get("timings") or {}).items():
                if isinstance(v, (int, float)):
                    module_totals[k] = round(module_totals.get(k, 0.0) + v, 3)

        doc = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration": duration,
            "meta": self.meta,
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "module_timings_total": module_totals,
            "case_count": len(self.case_logs),
            "case_logs": self.case_logs,
        }

        log_path = RUN_LOGS_DIR / f"{self.run_id}.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            doc["_write_error"] = str(e)
        doc["_log_path"] = str(log_path)
        return doc


class CaseTimer:
    """单 case 内部各步骤的计时器。用法:

        timer = CaseTimer()
        with timer.measure('agent_call'):
            ...
        with timer.measure('judge_call'):
            ...
        result = timer.snapshot()  # {"agent_call_total": 1.23, "agent_call_count": 3, ...}
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def measure(self, name: str):
        return _MeasureContext(self, name)

    def record(self, name: str, duration: float):
        self.timings[name] = self.timings.get(name, 0.0) + duration
        self.counts[name] = self.counts.get(name, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.timings.items():
            out[f"{k}_total"] = round(v, 3)
            out[f"{k}_count"] = self.counts.get(k, 0)
        return out


class _MeasureContext:
    def __init__(self, timer: CaseTimer, name: str):
        self.timer = timer
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.record(self.name, time.time() - self.t0)


def list_run_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """列出最近的 run logs 摘要。"""
    results = []
    files = sorted(RUN_LOGS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for fp in files[:limit]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
            results.append({
                "run_id": doc.get("run_id"),
                "started_at": doc.get("started_at"),
                "duration": doc.get("duration"),
                "case_count": doc.get("case_count"),
                "meta": doc.get("meta", {}),
            })
        except Exception:
            continue
    return results


def get_run_log(run_id: str) -> Optional[Dict[str, Any]]:
    p = RUN_LOGS_DIR / f"{run_id}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
