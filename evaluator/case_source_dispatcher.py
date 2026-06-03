"""case_source_dispatcher.py — 根据前端选择的 case_mode 分发构造 persona_list。

支持两种模式:
- llm: 走原来的 CaseAllocator / PersonaPoolAllocator 路径（不变）
- ref: 从 case_source_store 拿种子 → ReferenceExpander 扩写 → 生成 persona_list

返回统一格式 List[Tuple[Persona, Dict]] 供 BatchOrchestrator 消费。
"""
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from simulator import Persona
from evaluator.case_source_store import get_source
from evaluator.reference_expander import ReferenceExpander


class CaseSourceDispatcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        expander_model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.expander_model = expander_model

    def dispatch_ref(
        self,
        agent_system_prompt: str,
        case_source_id: str,
        total_n: int,
        conversation_goals: Optional[List[str]] = None,
    ) -> List[Tuple[Persona, Dict[str, Any]]]:
        """ref 模式：从 source 加载种子 → 扩写 → 输出 persona_list。

        Args:
            agent_system_prompt: 被测 agent 的 system prompt
            case_source_id: case source 的 ID
            total_n: 总用例数
            conversation_goals: 前端传入的对话目标列表（可选，如果有则补充到种子中）

        Returns:
            List[Tuple[Persona, Dict]] 供 BatchOrchestrator 使用
        """
        source = get_source(case_source_id)
        if source is None:
            raise ValueError(f"Case source '{case_source_id}' not found.")

        seed_cases = source.get("cases", [])
        if not seed_cases:
            raise ValueError(f"Case source '{case_source_id}' has no seed cases.")

        # 如果前端还填了额外 goals，也作为种子补充
        if conversation_goals:
            for g in conversation_goals:
                g = g.strip()
                if g:
                    seed_cases.append({"goal": g})

        # 调用 LLM 扩写
        expander = ReferenceExpander(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.expander_model,
        )
        expanded = expander.expand(
            agent_system_prompt=agent_system_prompt,
            seed_cases=seed_cases,
            total_n=total_n,
        )

        # 转换为 (Persona, metadata) 格式
        persona_list: List[Tuple[Persona, Dict[str, Any]]] = []
        for i, item in enumerate(expanded[:total_n]):
            p_data = item.get("persona", {})
            persona = Persona(
                name=f"ref_{uuid.uuid4().hex[:8]}",
                personality_traits=p_data.get("personality_traits", ["友好"]),
                speaking_style=p_data.get("speaking_style", "自然"),
                goal=item.get("goal", ""),
                constraints={},
                background=p_data.get("background", ""),
                identity=p_data.get("identity", "普通用户"),
                scene_premise=p_data.get("scene_premise", ""),
                agent_business_context=agent_system_prompt[:200],
                user_role_hint=p_data.get("identity", ""),
                plot_hooks=[],
            )
            metadata = {
                "case_type": "reference_expansion",
                "source_id": case_source_id,
                "seed_index": i % len(source.get("cases", [1])),
            }
            persona_list.append((persona, metadata))

        return persona_list
