"""
5 大生成多样性维度的默认描述 + merge 工具函数。

独立于 evaluator / simulator 双方，避免循环导入。
"""
from typing import Dict, Optional


# 5 大生成多样性维度的默认描述（与原 refiner_multi.md / allocator.md 中的硬编码一致）
DEFAULT_GENERATION_REQUIREMENTS: Dict[str, str] = {
    "req_identity": "覆盖不同年龄、职业、技术水平、地域背景的用户",
    "req_scenario": "包含正常咨询、投诉、紧急求助、模糊提问、刁难质疑等不同情境",
    "req_difficulty": "从简单到困难，包含边界 case（如信息不完整、情绪激动、反复追问）",
    "req_goal": "有的用户目标明确，有的目标模糊，有的目标不合理",
    "req_style": "有的用户表达清晰，有的啰嗦，有的简短，有的方言口音",
}


def merge_generation_requirements(
    user_reqs: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """合并用户传入的 5 维度生成要求 与默认值。空值/缺失项回落到默认值。

    支持 key 别名（前端可能传 identity/scenario/difficulty/goal/style 不带 req_ 前缀）。
    """
    aliases = {
        "identity": "req_identity",
        "scenario": "req_scenario",
        "difficulty": "req_difficulty",
        "goal": "req_goal",
        "style": "req_style",
    }
    merged = dict(DEFAULT_GENERATION_REQUIREMENTS)
    if isinstance(user_reqs, dict):
        for k, v in user_reqs.items():
            if v is None:
                continue
            v_str = str(v).strip()
            if not v_str:
                continue
            key = aliases.get(k, k)
            if key in DEFAULT_GENERATION_REQUIREMENTS:
                merged[key] = v_str
    return merged
