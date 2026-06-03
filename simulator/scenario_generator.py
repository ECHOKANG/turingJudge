import json
import os
import re
import hashlib
from typing import Dict, List, Optional, Any
from openai import OpenAI
from prompt_loader import load_md_prompt
from langfuse import observe, Langfuse
from simulator import Persona
from generation_requirements import merge_generation_requirements
from config import SCENARIO_GENERATOR_MODEL
from usage_logger import log_usage


class ScenarioGenerator:
    _cache: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("REFINER_MODEL", SCENARIO_GENERATOR_MODEL)
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("Missing API key for ScenarioGenerator.")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _cache_key(agent_system_prompt: str, num_scenarios: int, gen_reqs_repr: str = "") -> str:
        raw = f"{agent_system_prompt}||{num_scenarios}||{gen_reqs_repr}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @observe(name="scenario_generator", as_type="generation")
    def generate(
        self,
        agent_system_prompt: str,
        num_scenarios: int = 5,
        trace_collector: Optional[List] = None,
        generation_requirements: Optional[Dict[str, str]] = None,
    ) -> List[Persona]:
        gen_reqs = merge_generation_requirements(generation_requirements)
        gen_reqs_repr = json.dumps(gen_reqs, ensure_ascii=False, sort_keys=True)
        key = self._cache_key(agent_system_prompt, num_scenarios, gen_reqs_repr)
        if key in self._cache:
            cached = self._cache[key]
            personas = [self._dict_to_persona(s, i) for i, s in enumerate(cached)]
            if trace_collector is not None:
                trace_collector.append({
                    "caller": "ScenarioGenerator",
                    "model": self.model,
                    "output": json.dumps(cached, ensure_ascii=False),
                    "cached": True
                })
            return personas

        template = load_md_prompt("refiner_multi")
        user_message = template.format(
            agent_system_prompt=agent_system_prompt,
            num_scenarios=num_scenarios,
            **gen_reqs,
        )

        messages = [{"role": "user", "content": user_message}]

        Langfuse().update_current_trace(
            metadata={
                "model": self.model,
                "temperature": 0.8,
                "messages": messages,
            }
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8
        )

        result_text = response.choices[0].message.content or ""
        log_usage(label="ScenarioGenerator", model=self.model, response=response)

        if trace_collector is not None:
            trace_collector.append({
                "caller": "ScenarioGenerator",
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

        scenarios = result_json.get("scenarios", [])
        self._cache[key] = scenarios

        return [self._dict_to_persona(s, i) for i, s in enumerate(scenarios)]

    @staticmethod
    def _dict_to_persona(scenario: Dict[str, Any], index: int) -> Persona:
        return Persona(
            name=f"scenario_{index}",
            personality_traits=scenario.get("personality_traits", ["理性", "配合"]),
            speaking_style=scenario.get("speaking_style", "自然流畅"),
            goal=scenario.get("conversation_goal", ""),
            constraints={},
            background=scenario.get("background", ""),
            identity=scenario.get("identity", "普通用户"),
            scene_premise=scenario.get("scene_premise", ""),
        )
