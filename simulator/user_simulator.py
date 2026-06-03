from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import os
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from prompt_loader import load_md_prompt
from langfuse import observe, Langfuse
from usage_logger import log_usage


@dataclass
class Persona:
    name: str
    personality_traits: List[str]
    speaking_style: str
    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    background: Optional[str] = None
    identity: Optional[str] = None
    scene_premise: Optional[str] = None
    # —— 新增：业务上下文驱动 ——
    # Agent 业务上下文摘要（来自 system_prompt 节选），让 user 知道对方是干嘛的
    agent_business_context: Optional[str] = None
    # 推导出的"用户真实角色"提示，例如"你是骑手张师傅，今天因为天气不好接单意愿低"
    user_role_hint: Optional[str] = None
    # 剧情钩子：本 case 必须制造的张力点（列表，每条一个具体诉求/障碍/质疑）
    plot_hooks: List[str] = field(default_factory=list)


class UserSimulator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self._client: Optional[OpenAI] = None
        self._ark: Optional[Ark] = None
        self.personas: Dict[str, Persona] = {}
        self._prefix_cache_id: Optional[str] = None
        self._cached_system_prompt: Optional[str] = None

    def register_persona(self, persona: Persona) -> None:
        self.personas[persona.name] = persona

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("Missing API key. Please provide api_key in constructor or set OPENAI_API_KEY environment variable.")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _get_ark(self) -> Ark:
        if self._ark is None:
            self._ark = Ark(api_key=self.api_key, base_url=self.base_url, timeout=30.0)
        return self._ark

    def create_user_prompt(self, persona: Persona, conversation_history: Optional[List[Dict]] = None) -> str:
        template = load_md_prompt("simulator")

        identity = persona.identity or persona.background or "普通用户"
        scene_premise = persona.scene_premise or "（无特定场景，按常规对话处理）"
        background = persona.background or "（无补充背景）"
        agent_business_context = (persona.agent_business_context or "").strip() \
            or "（无额外业务上下文，按场景前提推断对方职责）"
        user_role_hint = (persona.user_role_hint or "").strip() \
            or "请根据『对方身份』和『场景前提』，自行推导出你最合理的真实身份并锁死。"
        if persona.plot_hooks:
            plot_hooks = "\n".join(f"- {h}" for h in persona.plot_hooks if h)
        else:
            plot_hooks = ("- 至少提出 1 个具体诉求或疑虑（不要只附和）\n"
                          "- 当对方解释规则时，要追问『对我来说意味着什么』或『如果我做不到呢』")

        history_section = ""
        if conversation_history:
            history_section = "【已发生的对话】\n"
            for msg in conversation_history:
                role = "你（用户）" if msg["role"] == "user" else "对方"
                history_section += f"{role}: {msg['content']}\n"

        constraints_text = "\n".join(f"- {k}: {v}" for k, v in persona.constraints.items()) if persona.constraints else "（无特殊约束）"

        return template.format(
            identity=identity,
            scene_premise=scene_premise,
            background=background,
            personality_traits=', '.join(persona.personality_traits),
            speaking_style=persona.speaking_style,
            goal=persona.goal,
            constraints=constraints_text,
            conversation_history=history_section,
            agent_business_context=agent_business_context,
            user_role_hint=user_role_hint,
            plot_hooks=plot_hooks,
        )

    def create_prefix_cache(self, persona_name: str) -> str:
        if persona_name not in self.personas:
            raise ValueError(f"Persona '{persona_name}' not registered")
        self._prefix_cache_id = None
        self._cached_system_prompt = None
        return ""

    def clear_prefix_cache(self) -> None:
        if self._prefix_cache_id:
            try:
                ark = self._get_ark()
                ark.responses.delete(self._prefix_cache_id)
            except Exception:
                pass
            self._prefix_cache_id = None
            self._cached_system_prompt = None

    @observe(name="user_simulator", as_type="generation")
    def generate_response(
        self,
        persona_name: str,
        agent_message: str,
        conversation_history: Optional[List[Dict]] = None,
        trace_collector: Optional[List] = None
    ) -> str:
        if persona_name not in self.personas:
            raise ValueError(f"Persona '{persona_name}' not registered")

        persona = self.personas[persona_name]

        if self._prefix_cache_id and self._cached_system_prompt:
            return self._generate_with_cache(
                persona_name, agent_message, conversation_history, trace_collector
            )

        user_prompt = self.create_user_prompt(persona, conversation_history)

        messages = [
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": agent_message}
        ]

        Langfuse().update_current_trace(
            metadata={
                "model": self.model,
                "temperature": DEFAULT_TEMPERATURE,
                "messages": messages,
            }
        )

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=int(os.getenv("SIMULATOR_MAX_TOKENS", "300")),
            )
        except Exception as e:
            print(f"[Simulator ERROR] model={self.model}, messages_count={len(messages)}, "
                  f"system_len={len(user_prompt)}, user_len={len(agent_message)}, error={e}")
            raise

        content = response.choices[0].message.content or ""
        log_usage(
            label=f"UserSimulator[{persona_name}]",
            model=self.model,
            response=response,
        )

        if trace_collector is not None:
            trace_collector.append({
                "caller": "UserSimulator",
                "persona_name": persona_name,
                "model": self.model,
                "input_messages": messages,
                "output": content,
                "cached": False
            })

        return content

    @observe(name="user_simulator_cache", as_type="generation")
    def _generate_with_cache(
        self,
        persona_name: str,
        agent_message: str,
        conversation_history: Optional[List[Dict]] = None,
        trace_collector: Optional[List] = None
    ) -> str:
        persona = self.personas[persona_name]
        ark = self._get_ark()

        if self._prefix_cache_id is None:
            system_prompt = self.create_user_prompt(persona)
            self._cached_system_prompt = system_prompt
            resp = ark.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": agent_message},
                ],
                caching={"type": "enabled"},
                thinking={"type": "disabled"},
            )
            self._prefix_cache_id = resp.id
        else:
            resp = ark.responses.create(
                model=self.model,
                previous_response_id=self._prefix_cache_id,
                input=[{"role": "user", "content": agent_message}],
                caching={"type": "enabled"},
                thinking={"type": "disabled"},
            )
            self._prefix_cache_id = resp.id

        content = resp.output[0].content[0].text if resp.output else ""
        cached_tokens = resp.usage.input_tokens_details.cached_tokens if resp.usage and resp.usage.input_tokens_details else 0
        is_cache_hit = self._cached_system_prompt is not None and cached_tokens > 0

        if is_cache_hit:
            Langfuse().update_current_trace(
                output=content,
                metadata={
                    "model": self.model,
                    "cached": True,
                    "cache_id": self._prefix_cache_id,
                    "cached_tokens": cached_tokens,
                    "input_tokens": resp.usage.input_tokens if resp.usage else 0,
                    "output_tokens": resp.usage.output_tokens if resp.usage else 0,
                }
            )

        if trace_collector is not None:
            trace_collector.append({
                "caller": "UserSimulator",
                "persona_name": persona_name,
                "model": self.model,
                "input_messages": [{"role": "user", "content": agent_message}],
                "output": content,
                "cached": is_cache_hit,
                "cache_id": self._prefix_cache_id,
                "cached_tokens": cached_tokens
            })

        return content

    def sample_persona_from_distribution(self, persona_distribution: Dict[str, float]) -> Persona:
        if not persona_distribution:
            if self.personas:
                return random.choice(list(self.personas.values()))
            raise ValueError("No personas registered and no distribution provided")

        total = sum(persona_distribution.values())
        rand = random.uniform(0, total)
        cumulative = 0

        for name, weight in persona_distribution.items():
            cumulative += weight
            if rand <= cumulative:
                if name in self.personas:
                    return self.personas[name]

        return random.choice(list(self.personas.values()))