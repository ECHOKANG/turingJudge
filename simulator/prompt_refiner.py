import json
import os
import re
import hashlib
from typing import Dict, List, Optional, Any
from openai import OpenAI
from prompt_loader import load_md_prompt
from langfuse import observe, Langfuse
from config import REFINER_MODEL
from usage_logger import log_usage


class PromptRefiner:
    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("REFINER_MODEL", REFINER_MODEL)
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("Missing API key for PromptRefiner.")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _cache_key(agent_system_prompt: str, conversation_goal: str) -> str:
        raw = f"{agent_system_prompt}||{conversation_goal}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @observe(name="prompt_refiner", as_type="generation")
    def refine(
        self,
        agent_system_prompt: str,
        identity: str,
        scene_premise: str,
        conversation_goal: str,
        trace_collector: Optional[List] = None
    ) -> Dict[str, Any]:
        key = self._cache_key(agent_system_prompt, conversation_goal)
        if key in self._cache:
            if trace_collector is not None:
                trace_collector.append({
                    "caller": "PromptRefiner",
                    "model": self.model,
                    "input_messages": [],
                    "output": json.dumps(self._cache[key], ensure_ascii=False),
                    "cached": True
                })
            return self._cache[key]

        template = load_md_prompt("refiner")
        user_message = template.format(
            agent_system_prompt=agent_system_prompt,
            identity=identity or "普通用户",
            scene_premise=scene_premise or "（无特定场景）",
            conversation_goal=conversation_goal or "（无特定目标）"
        )

        messages = [
            {"role": "user", "content": user_message}
        ]

        Langfuse().update_current_trace(
            metadata={
                "model": self.model,
                "temperature": 0.7,
                "messages": messages,
                "cached": False,
            }
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )

        result_text = response.choices[0].message.content or ""
        log_usage(label="PromptRefiner", model=self.model, response=response)

        if trace_collector is not None:
            trace_collector.append({
                "caller": "PromptRefiner",
                "model": self.model,
                "input_messages": messages,
                "output": result_text,
                "cached": False
            })

        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result_json = {}
            else:
                result_json = {}

        refined = {
            "inferred_identity": result_json.get("inferred_identity", identity or "普通用户"),
            "personality_traits": result_json.get("personality_traits", ["友好", "好奇"]),
            "speaking_style": result_json.get("speaking_style", "自然流畅"),
            "background": result_json.get("background", ""),
            "constraints": result_json.get("constraints", {}),
            "refined_goal": result_json.get("refined_goal", conversation_goal or "")
        }

        self._cache[key] = refined
        return refined
