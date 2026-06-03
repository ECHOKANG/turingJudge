"""reference_expander.py — 基于种子 case 做同分布扩写，生成 N 个 Persona + Goal。

输入: seed cases (来自 case_source_store) + agent_system_prompt + total_n
输出: List[Dict] 每条含 goal + persona 信息，可直接传入 BatchOrchestrator
"""
import json
import os
import re
from typing import Dict, List, Optional

from openai import OpenAI
from prompt_loader import load_md_prompt
from usage_logger import log_usage


class ReferenceExpander:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("EXPANDER_MODEL", "doubao-seed-2-0-mini-260428")
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def expand(
        self,
        agent_system_prompt: str,
        seed_cases: List[Dict],
        total_n: int,
    ) -> List[Dict]:
        """调用 LLM 扩写种子 case 到 total_n 条。

        返回:
        [
            {
                "goal": "...",
                "persona": {
                    "identity": "...",
                    "personality_traits": [...],
                    "speaking_style": "...",
                    "background": "...",
                    "scene_premise": "..."
                }
            },
            ...
        ]
        """
        template = load_md_prompt("reference_expander")

        # 构造种子文本
        seed_lines = []
        for i, sc in enumerate(seed_cases, 1):
            goal = sc.get("goal", "")
            msgs = sc.get("messages", [])
            line = f"{i}. 目标: {goal}"
            if msgs:
                snippet = " | ".join(
                    f"[{m.get('role', '?')}] {m.get('content', '')[:60]}"
                    for m in msgs[:4]
                )
                line += f"\n   对话片段: {snippet}"
            seed_lines.append(line)

        seed_cases_text = "\n".join(seed_lines)

        user_message = template.format(
            agent_system_prompt=agent_system_prompt,
            seed_count=len(seed_cases),
            seed_cases_text=seed_cases_text,
            total_n=total_n,
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.8,
        )

        result_text = response.choices[0].message.content or ""
        log_usage(label="ReferenceExpander", model=self.model, response=response)
        return self._parse_result(result_text, total_n, seed_cases, agent_system_prompt)

    def _parse_result(
        self, text: str, total_n: int, seed_cases: List[Dict], agent_system_prompt: str
    ) -> List[Dict]:
        """尝试解析 JSON 数组，fallback 到正则提取。"""
        # 尝试直接 parse
        try:
            items = json.loads(text)
            if isinstance(items, list):
                return self._validate_items(items)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 数组
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                items = json.loads(match.group())
                if isinstance(items, list):
                    return self._validate_items(items)
            except json.JSONDecodeError:
                pass

        # Fallback: 直接从种子复制并变体
        return self._fallback_expand(seed_cases, total_n)

    def _validate_items(self, items: List[Dict]) -> List[Dict]:
        """确保每条有 goal 和 persona 字段。"""
        valid = []
        for item in items:
            if not isinstance(item, dict):
                continue
            goal = item.get("goal", "")
            if not goal:
                continue
            persona = item.get("persona", {})
            if not isinstance(persona, dict):
                persona = {}
            valid.append({
                "goal": goal,
                "persona": {
                    "identity": persona.get("identity", "普通用户"),
                    "personality_traits": persona.get("personality_traits", ["友好"]),
                    "speaking_style": persona.get("speaking_style", "自然"),
                    "background": persona.get("background", ""),
                    "scene_premise": persona.get("scene_premise", ""),
                },
            })
        return valid

    def _fallback_expand(self, seed_cases: List[Dict], total_n: int) -> List[Dict]:
        """LLM 解析失败时的降级逻辑：循环复用种子的 goal。"""
        results = []
        for i in range(total_n):
            seed = seed_cases[i % len(seed_cases)]
            results.append({
                "goal": seed.get("goal", "通用测试目标"),
                "persona": {
                    "identity": "普通用户",
                    "personality_traits": ["友好", "好奇"],
                    "speaking_style": "自然流畅",
                    "background": "",
                    "scene_premise": "",
                },
            })
        return results
