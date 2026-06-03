"""case_source_store.py — 文件持久化层，管理 ref 模式的种子 case 数据源。

存储结构: data/case_sources/<source_id>.json
每个文件内容:
{
    "source_id": "...",
    "name": "用户自定义名称",
    "created_at": "ISO timestamp",
    "cases": [
        {
            "goal": "对话目标",
            "messages": [{"role":"user","content":"..."},...]  // 可选
        },
        ...
    ]
}
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import DATA_DIR

CASE_SOURCES_DIR = DATA_DIR / "case_sources"
CASE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)


def _source_path(source_id: str) -> Path:
    return CASE_SOURCES_DIR / f"{source_id}.json"


def create_source(name: str, cases: List[Dict]) -> Dict:
    """创建一个新的 case source，返回完整元数据。"""
    source_id = uuid.uuid4().hex[:12]
    doc = {
        "source_id": source_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    with open(_source_path(source_id), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc


def get_source(source_id: str) -> Optional[Dict]:
    """读取指定 source，不存在返回 None。"""
    p = _source_path(source_id)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sources() -> List[Dict]:
    """列出所有 source 的摘要信息（不含完整 cases 内容）。"""
    results = []
    for fp in sorted(CASE_SOURCES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
            results.append({
                "source_id": doc["source_id"],
                "name": doc["name"],
                "created_at": doc["created_at"],
                "case_count": len(doc.get("cases", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def delete_source(source_id: str) -> bool:
    """删除 source，成功返回 True。"""
    p = _source_path(source_id)
    if p.exists():
        p.unlink()
        return True
    return False
