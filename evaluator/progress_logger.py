"""实时批量评估进度日志。

每条事件以 JSONL 逐行写入文件并立即 flush，确保：
- 卡住时也能看到哪些 case 已完成、哪些正在执行
- 超时/异常 case 有完整记录
- 可通过 /api/batch_progress?run_id=xxx 实时查看
"""
from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR = Path("output/progress_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ProgressLogger:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.log_path = LOG_DIR / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._completed = 0
        self._failed = 0
        self._running = 0
        self._total = 0
        self._event("batch_start", {"message": "batch started"})

    def _event(self, event: str, data: Dict[str, Any]):
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(time.time() - self._started_at, 1),
            "event": event,
            **data,
        }
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def set_total(self, total: int):
        self._total = total
        self._event("batch_total", {"total": total})

    def _progress_pct(self) -> float:
        return round((self._completed + self._failed) / max(self._total, 1) * 100, 1)

    def case_start(self, index: int, case_type: str, identity: str = ""):
        self._running += 1
        self._event("case_start", {
            "index": index,
            "case_type": case_type,
            "identity": identity[:50] if identity else "",
            "running": self._running,
            "completed": self._completed,
            "failed": self._failed,
        })

    def case_done(self, index: int, case_type: str, score: float, duration: float):
        self._completed += 1
        self._running = max(0, self._running - 1)
        self._event("case_done", {
            "index": index,
            "case_type": case_type,
            "score": round(score, 2),
            "duration": round(duration, 1),
            "completed": self._completed,
            "failed": self._failed,
            "total": self._total,
            "running": self._running,
            "progress_pct": self._progress_pct(),
        })

    def case_fail(self, index: int, case_type: str, error: str, duration: float):
        self._failed += 1
        self._running = max(0, self._running - 1)
        self._event("case_fail", {
            "index": index,
            "case_type": case_type,
            "error": error[:200],
            "duration": round(duration, 1),
            "completed": self._completed,
            "failed": self._failed,
            "total": self._total,
            "running": self._running,
            "progress_pct": self._progress_pct(),
        })

    def case_timeout(self, index: int, case_type: str, timeout: float):
        self._failed += 1
        self._running = max(0, self._running - 1)
        self._event("case_timeout", {
            "index": index,
            "case_type": case_type,
            "timeout_sec": timeout,
            "completed": self._completed,
            "failed": self._failed,
            "total": self._total,
            "running": self._running,
            "progress_pct": self._progress_pct(),
        })

    def batch_done(self, stats: Dict[str, Any]):
        self._event("batch_done", {
            "completed": self._completed,
            "failed": self._failed,
            "total": self._total,
            "running": self._running,
            "progress_pct": 100.0,
            "elapsed": round(time.time() - self._started_at, 1),
            **stats,
        })

    def read_log(self) -> str:
        if self.log_path.exists():
            return self.log_path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def list_runs() -> list:
        if not LOG_DIR.exists():
            return []
        runs = []
        for p in sorted(LOG_DIR.glob("*.jsonl"), reverse=True):
            runs.append({"run_id": p.stem, "size": p.stat().st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))})
        return runs[:20]
