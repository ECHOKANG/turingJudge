from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import os
import re
from string import Template
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from prompt_loader import load_prompt
from langfuse import observe, Langfuse
from usage_logger import log_usage


# 占位符扫描规则:
# 1. ${var} / {var} / {{var}}  -> 模板变量未填充
# 2. <var> / [var]              -> 占位标记
# 3. 单独大写字母 X/Y/Z/W/N （非英文单词的一部分）-> 数值占位
def _load_placeholder_patterns():
    cfg = load_prompt("evaluation")
    return [(p[0], p[1]) for p in cfg["placeholder_patterns"]]

PLACEHOLDER_PATTERNS = _load_placeholder_patterns()


def scan_prompt_placeholders(prompt: str) -> List[Tuple[str, str]]:
    """扫描 prompt 中的未填充占位符。

    Returns:
        List of (matched_text, pattern_description)
    """
    findings: List[Tuple[str, str]] = []
    for pattern, desc in PLACEHOLDER_PATTERNS:
        for m in re.finditer(pattern, prompt):
            findings.append((m.group(), desc))
    return findings


# 上下文变量名提取:从占位符匹配文本里抽出变量名(去掉 ${} / {} / {{}} / <> / [] 等装饰)
_VAR_NAME_RE = re.compile(r"[\w]+")


def extract_var_name(token: str) -> Optional[str]:
    """从一个占位符 token (如 '${user_name}'、'{order_id}'、'<city>') 抽出纯变量名。"""
    m = _VAR_NAME_RE.search(token)
    return m.group() if m else None


def render_prompt_with_context(
    prompt: str,
    context_variables: Optional[Dict[str, str]] = None,
) -> str:
    """用 context_variables 替换 prompt 中的占位符。

    支持的占位符语法:
      - ${var}    (Python Template 风格,优先匹配)
      - {{var}}   (Jinja 风格)
      - {var}     (单括号,需独立词)
      - <var>     (尖括号)
      - [VAR]     (大写方括号)

    未在 context_variables 中的占位符保持原样,不抛错。
    """
    if not context_variables:
        return prompt

    rendered = prompt

    # 1) ${var}  →  Python Template safe_substitute
    try:
        rendered = Template(rendered).safe_substitute(context_variables)
    except Exception:
        pass

    # 2) {{var}}
    def _double_brace(m):
        name = m.group(1).strip()
        return str(context_variables.get(name, m.group(0)))
    rendered = re.sub(r"\{\{([\w\s]+)\}\}", _double_brace, rendered)

    # 3) {var} (独立)
    def _single_brace(m):
        name = m.group(1)
        return str(context_variables.get(name, m.group(0)))
    rendered = re.sub(r"(?<![\w])\{([\w]+)\}(?![\w])", _single_brace, rendered)

    # 4) <var>
    def _angle(m):
        name = m.group(1)
        return str(context_variables.get(name, m.group(0)))
    rendered = re.sub(r"<([A-Za-z_][\w]*)>", _angle, rendered)

    # 5) [VAR]
    def _bracket(m):
        name = m.group(1)
        # 同时尝试小写匹配,方便用户填 user_id 既能命中 [USER_ID] 也命中 [user_id]
        return str(context_variables.get(name, context_variables.get(name.lower(), m.group(0))))
    rendered = re.sub(r"(?<![\w])\[([A-Z_][A-Z0-9_]{1,30})\](?![\w])", _bracket, rendered)

    return rendered


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    context_window: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 上下文变量(用户在 V1 配置的 ${var} / {var} / <var> 等占位符的实际值)
    context_variables: Dict[str, str] = field(default_factory=dict)

    @property
    def effective_system_prompt(self) -> str:
        """返回应用 context_variables 后的真实 system prompt。"""
        return render_prompt_with_context(self.system_prompt, self.context_variables)


class AgentContainer:
    def __init__(self, strict_placeholder_check: bool = False):
        self.agents: Dict[str, AgentConfig] = {}
        self._clients: Dict[str, OpenAI] = {}
        self._ark_clients: Dict[str, Ark] = {}
        self.strict_placeholder_check = strict_placeholder_check
        self.placeholder_warnings: Dict[str, List[Tuple[str, str]]] = {}
        self._prefix_cache_ids: Dict[str, str] = {}

    def register_agent(self, config: AgentConfig) -> None:
        # 用渲染后的 prompt 做占位符告警检查,避免把已经被 context_variables 填充的变量误报
        rendered = config.effective_system_prompt
        findings = scan_prompt_placeholders(rendered)
        if findings:
            self.placeholder_warnings[config.name] = findings
            if self.strict_placeholder_check:
                msg_parts = [f"'{txt}' ({desc})" for txt, desc in findings[:5]]
                raise ValueError(
                    f"Agent '{config.name}' 的 system_prompt 中检测到未填充占位符: "
                    + ", ".join(msg_parts)
                    + ("，..." if len(findings) > 5 else "")
                    + "。请先填充后再保存（或将 strict_placeholder_check 设为 False）。"
                )
        else:
            self.placeholder_warnings.pop(config.name, None)

        self.agents[config.name] = config

    def get_placeholder_warnings(self, agent_name: str) -> List[Tuple[str, str]]:
        return self.placeholder_warnings.get(agent_name, [])

    def _create_client(self, config: AgentConfig) -> OpenAI:
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("Missing API key. Please provide api_key in config or set OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)

    def _get_ark(self, config: AgentConfig) -> Ark:
        if config.name not in self._ark_clients:
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
            self._ark_clients[config.name] = Ark(api_key=api_key, base_url=base_url, timeout=30.0)
        return self._ark_clients[config.name]

    def create_prefix_cache(self, agent_name: str) -> str:
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")
        self._prefix_cache_ids[agent_name] = None
        return ""

    def clear_prefix_cache(self, agent_name: str) -> None:
        cache_id = self._prefix_cache_ids.pop(agent_name, None)
        if cache_id and agent_name in self.agents:
            try:
                config = self.agents[agent_name]
                ark = self._get_ark(config)
                ark.responses.delete(cache_id)
            except Exception:
                pass

    def clear_all_prefix_caches(self) -> None:
        for agent_name in list(self._prefix_cache_ids.keys()):
            self.clear_prefix_cache(agent_name)

    @observe(name="agent_container", as_type="generation")
    def generate_response(
        self,
        agent_name: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        trace_collector: Optional[List] = None
    ) -> str:
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")

        config = self.agents[agent_name]

        if agent_name in self._prefix_cache_ids:
            return self._generate_with_cache(
                agent_name, user_message, trace_collector
            )

        if agent_name not in self._clients:
            self._clients[agent_name] = self._create_client(config)

        client = self._clients[agent_name]

        messages = [{"role": "system", "content": config.effective_system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        Langfuse().update_current_trace(
            metadata={
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "messages": messages,
            }
        )

        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        content = response.choices[0].message.content or ""
        log_usage(
            label=f"AgentContainer[{agent_name}]",
            model=config.model,
            response=response,
        )

        if trace_collector is not None:
            trace_collector.append({
                "caller": "AgentContainer",
                "agent_name": agent_name,
                "model": config.model,
                "input_messages": messages,
                "output": content,
                "cached": False
            })

        return content

    @observe(name="agent_container_cache", as_type="generation")
    def _generate_with_cache(
        self,
        agent_name: str,
        user_message: str,
        trace_collector: Optional[List] = None
    ) -> str:
        config = self.agents[agent_name]
        ark = self._get_ark(config)
        cache_id = self._prefix_cache_ids.get(agent_name)

        if cache_id is None:
            resp = ark.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": config.effective_system_prompt},
                    {"role": "user", "content": user_message},
                ],
                caching={"type": "enabled"},
                thinking={"type": "disabled"},
            )
            self._prefix_cache_ids[agent_name] = resp.id
        else:
            resp = ark.responses.create(
                model=config.model,
                previous_response_id=cache_id,
                input=[{"role": "user", "content": user_message}],
                caching={"type": "enabled"},
                thinking={"type": "disabled"},
            )
            self._prefix_cache_ids[agent_name] = resp.id

        content = resp.output[0].content[0].text if resp.output else ""
        cached_tokens = resp.usage.input_tokens_details.cached_tokens if resp.usage and resp.usage.input_tokens_details else 0
        is_cache_hit = cache_id is not None and cached_tokens > 0

        if is_cache_hit:
            Langfuse().update_current_trace(
                output=content,
                metadata={
                    "model": config.model,
                    "cached": True,
                    "cache_id": self._prefix_cache_ids[agent_name],
                    "cached_tokens": cached_tokens,
                    "input_tokens": resp.usage.input_tokens if resp.usage else 0,
                    "output_tokens": resp.usage.output_tokens if resp.usage else 0,
                }
            )

        if trace_collector is not None:
            trace_collector.append({
                "caller": "AgentContainer",
                "agent_name": agent_name,
                "model": config.model,
                "input_messages": [{"role": "user", "content": user_message}],
                "output": content,
                "cached": is_cache_hit,
                "cache_id": self._prefix_cache_ids[agent_name],
                "cached_tokens": cached_tokens
            })

        return content

    def get_agent(self, agent_name: str) -> Optional[AgentConfig]:
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())
